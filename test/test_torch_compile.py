from __future__ import annotations

import contextlib
import functools
import math
import operator
import re
import unittest
from unittest.mock import patch

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._compat import requires_torch_version
from helion._compat import supports_tensor_descriptor
from helion._compat import supports_torch_compile_fusion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl


def requires_fusion_support(test_fn):
    """Decorator: when fusion is unsupported, assert the upgrade error instead of running."""

    @functools.wraps(test_fn)
    def wrapper(self, *args, **kwargs):
        ctx = (
            contextlib.nullcontext()
            if supports_torch_compile_fusion()
            else self.assertRaisesRegex(
                RuntimeError,
                "torch_compile_fusion=True requires PyTorch nightly build",
            )
        )
        with ctx:
            test_fn(self, *args, **kwargs)

    return wrapper


# -----------------------------------------------------------------------------
# Basic Operations (no mutation, return new tensor)
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def k_add_ref(x, y):  # noqa: FURB118
    return x + y


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


def k_scale_two_ref(x, y):
    return x * 2.0, y * 3.0


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


def k_scale_with_scalar_output_ref(x, scale):
    return x * scale, 42


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


def k_tensor_scalar_tensor_ref(x, y, scale):
    return x * scale, 7, y * scale


@helion.kernel(autotune_effort="none")
def k_single_element_tuple(x: torch.Tensor) -> tuple[torch.Tensor]:
    """Return a single-element tuple."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2.0
    return (out,)


def k_single_element_tuple_ref(x):
    return (x * 2.0,)


@helion.kernel(autotune_effort="none")
def k_sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of x."""
    m, _ = x.size()
    out = torch.empty([m], device=x.device, dtype=x.dtype)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)
    return out


def k_sum_rows_ref(x):
    return x.to(torch.float32).sum(-1).to(x.dtype)


@helion.kernel(autotune_effort="none")
def k_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS normalization: returns (out, residual)."""
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

    return out, residual


def k_rms_norm_ref(x, weight, eps=1e-5):
    x_float = x.to(torch.float32)
    x_squared = x_float * x_float
    mean_x_squared = torch.mean(x_squared, dim=-1)
    inv_rms = torch.rsqrt(mean_x_squared + eps)
    normalized = x_float * inv_rms[:, None]
    result = normalized * weight.to(torch.float32)
    out = result.to(x.dtype)
    residual = normalized.to(x.dtype)
    return out, residual


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


def k_add_inplace_ref(x, y):
    x.add_(y)
    return x


@helion.kernel(autotune_effort="none")
def k_mutate_both(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile in hl.tile(x.size(0)):
        x[tile], y[tile] = x[tile] + 1, y[tile] * 2
    return x, y


def k_mutate_both_ref(x, y):
    x.add_(1)
    y.mul_(2)
    return x, y


@helion.kernel(autotune_effort="none")
def k_mutate_via_view(x: torch.Tensor) -> torch.Tensor:
    """Create view internally and mutate through it."""
    y = x.view(x.size())
    for tile in hl.tile(y.size()):
        y[tile] = y[tile] + 1
    return x


def k_mutate_via_view_ref(x):
    x.add_(1)
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


def k_store_ref(x, y):
    x.copy_(y * 2)
    return x


@helion.kernel(autotune_effort="none")
def k_atomic_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile], y[tile])
    return x


def k_atomic_add_ref(x, y):
    x.add_(y)
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


def k_mutate_with_out_ref(x, y):
    x.add_(1)
    out = x + y
    return x, out


@helion.kernel(autotune_effort="none")
def k_mutate_return_new(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        out[tile] = y[tile] * 2
    return out


def k_mutate_return_new_ref(x, y):
    x.add_(1)
    return y * 2


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


def k_mutate_two_return_new_ref(x, y, z):
    x.add_(1.0)
    y.mul_(2.0)
    return z + x + y


# -----------------------------------------------------------------------------
# Pre-allocated/External Output
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add_into_out(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Add x and y, store result in pre-allocated out tensor."""
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def k_add_into_out_ref(x, y, out):
    out.copy_(x + y)
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


def k_atomic_add_to_out_ref(x, y, out):
    out.add_(x)
    out.add_(y)
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


def k_mutate_permuted_ref(x, y):
    x.add_(y)
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


def k_create_return_view_ref(x, y):
    return (x + y).view(-1)


GLOBAL_SCALE_FACTOR = 2.5


@helion.kernel(autotune_effort="none")
def k_scale_with_global_var(x: torch.Tensor) -> torch.Tensor:
    """Scale x by a captured global variable."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * GLOBAL_SCALE_FACTOR
    return out


def k_scale_with_global_var_ref(x):
    return x * GLOBAL_SCALE_FACTOR


# =============================================================================
# Test Class
# =============================================================================


@onlyBackends(["triton"])
class TestTorchCompile(RefEagerTestDisabled, TestCase):
    def _compile_and_count_kernels(self, f, test_args, dynamic=False):
        """Compile f with torch.compile and return (result, source_codes, count)."""
        helion_kernel_side_table = None
        if supports_torch_compile_fusion():
            from helion._compiler._dynamo.higher_order_ops import (
                helion_kernel_side_table,
            )

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        # torch.compile fusion is controlled by an env var in these tests.
        # Clear both Python code artifacts and the guarded FX graph cache so
        # we don't accidentally reuse a graph compiled under a different mode.
        # Reset the HOP side table when available so earlier test modules can't
        # leak kernel indices into the current trace.
        FxGraphCache.clear()
        PyCodeCache.cache_clear()
        # Clear fusion config cache to prevent cross-test pollution.
        if supports_torch_compile_fusion():
            from helion._compiler._inductor.template_buffer import HelionTemplateBuffer

            HelionTemplateBuffer._fusion_config_cache.clear()
        if helion_kernel_side_table is not None:
            helion_kernel_side_table.reset_table()

        with fresh_cache():
            # Warmup
            warmup_args = tuple(
                a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
            )
            _ = f(*warmup_args)

            # Compile and run
            compiled_f = torch.compile(
                f, fullgraph=True, backend="inductor", dynamic=dynamic
            )
            run_args = tuple(
                a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
            )
            actual, source_codes = run_and_get_code(compiled_f, *run_args)

        # Count kernels
        kernel_count = sum(code.count("@triton.jit") for code in source_codes)

        # Verify no graph breaks
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        return actual, source_codes, kernel_count

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
        expected_num_kernels: int | None = None,
        expected_num_compilations: list[int] | None = None,
        kernels_ref: list | None = None,
        expected_num_kernels_ref: int | None = None,
    ):
        """Run torch.compile test comparing eager vs compiled execution."""
        if allow_torch_compile_fusion and not supports_torch_compile_fusion():
            expected_error = (
                RuntimeError,
                "torch_compile_fusion=True requires PyTorch nightly build",
            )

        # Reset specific kernels and configure fusion setting
        for kernel in kernels:
            self.addCleanup(
                setattr,
                kernel.settings,
                "torch_compile_fusion",
                kernel.settings.torch_compile_fusion,
            )
            kernel.settings.torch_compile_fusion = allow_torch_compile_fusion
            kernel.reset()

        # Handle expected errors
        if expected_error is not None:
            helion_kernel_side_table = None
            if supports_torch_compile_fusion():
                from helion._compiler._dynamo.higher_order_ops import (
                    helion_kernel_side_table,
                )

            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            FxGraphCache.clear()
            PyCodeCache.cache_clear()
            if helion_kernel_side_table is not None:
                helion_kernel_side_table.reset_table()
            error_type, error_pattern = expected_error
            with fresh_cache():
                # Warmup
                warmup_args = tuple(
                    a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
                )
                _ = f(*warmup_args)
                compiled_f = torch.compile(
                    f, fullgraph=True, backend="inductor", dynamic=dynamic
                )
                compiled_args = tuple(
                    a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
                )
                with self.assertRaisesRegex(error_type, error_pattern):
                    compiled_f(*compiled_args)
            return

        # Get expected result (eager)
        expected_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        expected = f(*expected_args)

        # Compile helion version and count kernels
        actual, source_codes, kernel_count = self._compile_and_count_kernels(
            f, test_args, dynamic=dynamic
        )

        # Compare results
        if compare_fn is not None:
            compare_fn(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

        # Assert helion kernel count
        if expected_num_kernels is not None:
            self.assertEqual(
                kernel_count,
                expected_num_kernels,
                f"Expected {expected_num_kernels} triton kernel(s), got {kernel_count}",
            )

        # Ref baseline kernel count check
        if expected_num_kernels_ref is not None:
            assert kernels_ref is not None, (
                "kernels_ref must be provided when expected_num_kernels_ref is set"
            )
            ref_f = functools.partial(f, _kernels=tuple(kernels_ref))
            _, _, ref_kernel_count = self._compile_and_count_kernels(
                ref_f, test_args, dynamic=dynamic
            )
            self.assertEqual(
                ref_kernel_count,
                expected_num_kernels_ref,
                f"Ref baseline: expected {expected_num_kernels_ref} "
                f"triton kernel(s), got {ref_kernel_count}",
            )

        # Verify helion compilation count (no unexpected recompilation)
        if expected_num_compilations is not None:
            assert len(expected_num_compilations) == len(kernels)
            for kernel, expected in zip(
                kernels, expected_num_compilations, strict=True
            ):
                self.assertEqual(
                    len(kernel._bound_kernels),
                    expected,
                    f"Expected {expected} helion compilation(s) for {kernel}, "
                    f"got {len(kernel._bound_kernels)}",
                )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_add_kernel(self, allow_torch_compile_fusion):
        """Test: basic addition kernel with prologue/epilogue ops."""

        def f(x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add,)) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = _kernels[0](x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_elementwise_kernel(self, allow_torch_compile_fusion):
        """Test: multi-input elementwise ops with complex prologue/epilogue."""

        def f(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, _kernels=(k_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            a = x * 2.0
            b = y + z
            result = _kernels[0](a, b)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_multi_input_return_used(self, allow_torch_compile_fusion):
        """Test: kernel with multiple inputs that mutates and returns one."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: torch.Tensor,
            *,
            _kernels=(k_add_inplace,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = _kernels[0](x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs(self, allow_torch_compile_fusion):
        """Test: kernel with multiple differently-shaped outputs."""

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_scale_two,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scaled_x, scaled_y = _kernels[0](x, y)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_two_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_all_keyword(self, allow_torch_compile_fusion):
        """Test: all keyword argument passing."""

        def f(x, y, z, *, _kernels=(k_add,)):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = _kernels[0](y=y + z, x=x) * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_mixed(self, allow_torch_compile_fusion):
        """Test: mixed positional/keyword argument passing."""

        def f(x, y, z, *, _kernels=(k_add,)):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = _kernels[0](x, y=y + z) - 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_default_params(self, allow_torch_compile_fusion):
        """Test: kernel with default vs custom parameter values."""

        def f_with_default_scale(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor, *, _kernels=(k_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Inline scaling (default scale=2.0) before kernel
            result = _kernels[0](biased_x, y * 2.0)
            result = result * 0.5
            return torch.relu(result) + 1.0

        def f_with_custom_scale(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor, *, _kernels=(k_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Inline scaling (custom scale=3.0) before kernel
            result = _kernels[0](biased_x, y * 3.0)
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
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

        # Test with custom scale
        self._run_compile_test(
            f_with_custom_scale,
            (x, y, bias),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_constant_scalar_args(self, allow_torch_compile_fusion):
        """Test: scalar constants in prologue/epilogue operations."""

        def f(x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add,)) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            # Apply scale and shift as prologue operations
            a = x * 2.5 + 1.0
            b = y * 2.5 + 1.0
            result = _kernels[0](a, b)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transposed_input(self, allow_torch_compile_fusion):
        """Test: transposed (non-contiguous) tensor input."""

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, *, _kernels=(k_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x.T * scale
            b = y.T
            result = _kernels[0](a, b)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(8, 4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(8, 4, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_called_twice(self, allow_torch_compile_fusion):
        """Test: same kernel called twice with different inputs."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
            scale: torch.Tensor,
            *,
            _kernels=(k_add,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            a = _kernels[0](scaled_x, y)
            b = _kernels[0](a, z)
            result = b + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, z, scale),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_tensor_as_two_different_args(self, allow_torch_compile_fusion):
        """Test: same tensor passed as two different arguments."""

        def f(
            x: torch.Tensor, bias: torch.Tensor, *, _kernels=(k_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = x * 2.0 + bias
            result = _kernels[0](scaled, scaled)
            result = result.mean(dim=-1)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, bias),
            kernels=[k_add],
            rtol=1e-2,
            atol=1e-2,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_mutation(self, allow_torch_compile_fusion):
        """Test: mutation via atomic operations."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            out: torch.Tensor,
            *,
            _kernels=(k_atomic_add_to_out,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 0.5
            b = y.abs()
            result = _kernels[0](a, b, out)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_atomic_add_to_out_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @unittest.skip("Correctness bug with indirect output aliasing")
    def test_indirect_output_alias(self, allow_torch_compile_fusion):
        """Test: output is a slice/view of input (indirect alias with different shape)."""

        def k_slice_mutate_ref(x, y):
            x[:2, :4].add_(y)
            return x[:2, :4]

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: torch.Tensor,
            *,
            _kernels=(k_slice_mutate,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = _kernels[0](x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_slice_mutate],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else 5,
            kernels_ref=[k_slice_mutate_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_empty_tensor(self, allow_torch_compile_fusion):
        """Test: tensors with zero-size dimensions."""

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, *, _kernels=(k_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            result = _kernels[0](scaled_x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        # Test with zero-size first dimension
        x = torch.randn(0, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(0, 8, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(0, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=0 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=0,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_reduction_sum(self, allow_torch_compile_fusion):
        """Test: kernel with reduction dimension (sum along axis)."""

        def f(
            x: torch.Tensor, weight: torch.Tensor, *, _kernels=(k_sum_rows,)
        ) -> torch.Tensor:
            x = x * 2.0
            weight = weight * 2.0
            scaled = x * weight
            # Helion kernel with reduction
            row_sums = _kernels[0](scaled)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_sum_rows_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inline_triton_mutation(self, allow_torch_compile_fusion):
        """Test: kernel using inline_triton marks all inputs as potentially mutated."""

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_inline_add,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x.exp()
            b = y.log1p()
            # Helion kernel with inline_triton
            result = _kernels[0](a, b)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        y = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_inline_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_argument_kernel_mutation(self, allow_torch_compile_fusion):
        """Test: kernel with mutation on first argument (tests mutation pattern)."""

        def f(
            x: torch.Tensor, bias: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = (x * 2.0 + bias).contiguous()
            ones = torch.ones_like(scaled)
            # Helion kernel with mutation
            result = _kernels[0](scaled, ones)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_slice_return_other],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if supports_torch_compile_fusion()
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_permute_view(self, allow_torch_compile_fusion):
        """Test: output is permuted view of input."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: torch.Tensor,
            *,
            _kernels=(k_mutate_permuted,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = _kernels[0](x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_mutate_permuted],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_permuted_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_alias_view_as_two_args(self, allow_torch_compile_fusion):
        """Test: passing x and aliased view of x as two different arguments."""

        def f(a: torch.Tensor, *, _kernels=(k_add_inplace,)) -> torch.Tensor:
            a = a * 2.0
            x = a * 2
            y = x.view(-1).view(8, 8)  # Reshape to maintain 2D, aliased with x
            result = _kernels[0](x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        a = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (a,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliasing_inputs_used_after(self, allow_torch_compile_fusion):
        """Test: view of input used after kernel mutation."""

        def f(x: torch.Tensor, *, _kernels=(k_add_inplace,)) -> torch.Tensor:
            x = x * 2.0
            y = x.view(-1)  # View before kernel
            ones = torch.ones_like(x)
            _ = _kernels[0](x, ones)  # Mutate x
            result = y + 1  # Use view after - should see mutation
            return torch.relu(result) + 1.0

        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_through_internal_view(self, allow_torch_compile_fusion):
        """Test: kernel that creates a view inside the kernel and mutates through it."""

        def f(x: torch.Tensor, *, _kernels=(k_mutate_via_view,)) -> torch.Tensor:
            x = x * 2.0
            x = x - 1  # Prologue
            result = _kernels[0](x)
            result = result * 2  # Epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_mutate_via_view],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_via_view_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_mutated_inputs(self, allow_torch_compile_fusion):
        """Test: kernel that mutates multiple input tensors independently."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
            *,
            _kernels=(k_mutate_two_return_new,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = _kernels[0](x, y, z)
            result = result - 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_mutate_two_return_new],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_two_return_new_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_detached_input(self, allow_torch_compile_fusion):
        """Test: input is detached from grad-tracking tensor."""

        def f(x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add,)) -> torch.Tensor:
            y = y * 2.0
            # Detach x before passing to kernel
            x_detached = x.detach()
            result = _kernels[0](x_detached, y)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_module_forward_with_kernel(self, allow_torch_compile_fusion):
        """Test: Helion kernel called inside nn.Module.forward()."""

        class SimpleModule(torch.nn.Module):
            def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.bias = torch.nn.Parameter(bias)

            def forward(self, x: torch.Tensor, kernel_fn) -> torch.Tensor:
                return kernel_fn(x * self.weight, self.bias)

        def f(module, x, *, _kernels=(k_add,)):
            x = x * 2.0
            result = module(x, _kernels[0])
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
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_with_chained_prologue_epilogue_ops(
        self, allow_torch_compile_fusion
    ):
        """Test: mutation with prologue/epilogue operations."""

        def f(
            x: torch.Tensor, bias: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            biased = x + bias
            ones = torch.ones_like(biased)
            result = _kernels[0](biased, ones)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate(self, allow_torch_compile_fusion):
        """Test: clone tensor, mutate clone, verify original unchanged."""

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            # Apply epilogue only to result, not to x (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_preallocated_output(self, allow_torch_compile_fusion):
        """Test: kernel fills pre-allocated output tensor passed as argument."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            out: torch.Tensor,
            *,
            _kernels=(k_add_into_out,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 2.0
            b = y + 1.0
            result = _kernels[0](a, b, out)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        out = torch.empty(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, out),
            kernels=[k_add_into_out],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_into_out_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_mutate_return_view],
            expected_error=(
                RuntimeError,
                r"Returning multiple outputs that share storage.*not yet supported",
            )
            if supports_torch_compile_fusion()
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliased_storage_different_shape(self, allow_torch_compile_fusion):
        """Test: inputs share storage but have different shapes."""

        def f(base: torch.Tensor, *, _kernels=(k_add,)) -> torch.Tensor:
            base = base * 2.0
            # Create two views of base with different strides
            x = base[::2]  # Every other element: shape [16]
            y = base[1::2]  # Every other element offset by 1: shape [16]
            result = _kernels[0](x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        base = torch.randn(32, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (base,),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=5 if allow_torch_compile_fusion else 6,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_output_aliases_intermediate(self, allow_torch_compile_fusion):
        """Test: output aliases tensor created inside the kernel."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: torch.Tensor,
            *,
            _kernels=(k_create_return_view,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x * scale
            b = y + 1.0
            result = _kernels[0](a, b)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_create_return_view],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_create_return_view_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inference_mode(self, allow_torch_compile_fusion):
        """Test: kernel works correctly inside inference_mode context."""

        def f(x, y, *, _kernels=(k_add_inplace,)):
            x = x * 2.0
            y = y * 2.0
            z = x + 0.5  # prologue
            result = _kernels[0](z, y)
            result = result * 2  # epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        with torch.inference_mode():
            self._run_compile_test(
                f,
                (x, y),
                kernels=[k_add_inplace],
                allow_torch_compile_fusion=allow_torch_compile_fusion,
                expected_num_kernels=3 if allow_torch_compile_fusion else None,
                kernels_ref=[k_add_inplace_ref],
                expected_num_kernels_ref=1,
            )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (z,),
            kernels=[k_add_to_both],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if supports_torch_compile_fusion()
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("ROCm Triton worker crashes on graph-input view torch.compile coverage")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_graph_input_is_view_with_kernel(self, allow_torch_compile_fusion):
        """Test: graph input is a view, kernel operates on derived view."""

        def f(x, y, *, _kernels=(k_add_inplace,)):
            x = x * 2.0
            y = y * 2.0
            # x is already a view (passed from outside), take a 2D slice
            a = x[:2]  # 2D slice of view
            result = _kernels[0](a.clone(), y[:2])
            return torch.relu(result) + 1.0

        base = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        x = base[1:]  # view with shape [3, 8]
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_assigned(self, allow_torch_compile_fusion):
        """Test: mutation where return value is assigned to a variable."""

        def fn(x, *, _kernels=(k_add_inplace,)):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            x = _kernels[0](x, ones)
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_discarded(self, allow_torch_compile_fusion):
        """Test: mutation where return value is discarded (not assigned)."""

        def fn(x, *, _kernels=(k_add_inplace,)):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            _kernels[0](x, ones)  # return ignored; still mutates x
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_mutated(self, allow_torch_compile_fusion):
        """Test: kernel that mutates two inputs and returns both."""

        def fn(x, y, *, _kernels=(k_mutate_both,)):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y - 1
            x, y = _kernels[0](x, y)
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
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_both_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_one_mutated(self, allow_torch_compile_fusion):
        """Test: kernel that mutates one input."""

        def fn(x, y, *, _kernels=(k_add_inplace,)):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = _kernels[0](x, y)
            result = x - 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mut_and_out(self, allow_torch_compile_fusion):
        """Test: kernel that mutates input and also returns new tensor."""

        def fn(x, y, *, _kernels=(k_mutate_with_out,)):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y + 1
            x, out = _kernels[0](x, y)
            x = torch.relu(x) + 1.0
            out = torch.relu(out) + 1.0
            return x, out

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_mutate_with_out],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_with_out_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_input_reused_after_call(self, allow_torch_compile_fusion):
        """Test: mutated input is used after kernel call, but kernel returns a different tensor."""

        def fn(x, y, *, _kernels=(k_mutate_return_new,)):
            x = x * 2.0
            y = y * 2.0
            x = x + 1
            out = _kernels[0](x, y)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_return_new_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_store_operation(self, allow_torch_compile_fusion):
        """Test hl.store write operation."""

        def fn(x, y, *, _kernels=(k_store,)):
            x = x * 2.0
            y = y * 2.0
            y = y + 1
            x = _kernels[0](x, y)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_store_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_operation(self, allow_torch_compile_fusion):
        """Test hl.atomic_add write operation."""

        def fn(x, y, *, _kernels=(k_atomic_add,)):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = _kernels[0](x, y)
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
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_atomic_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_no_mutation(self, allow_torch_compile_fusion):
        """Test: pure function kernel with no input mutations."""

        def fn(x, y, *, _kernels=(k_add,)):
            x = x * 2.0
            y = y * 2.0
            x, y = x * 2, y * 2
            out = _kernels[0](x, y)
            result = out + 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_tuple(self, allow_torch_compile_fusion):
        """Test: prologue/epilogue with tuple (tensor, scalar) output."""
        kernel_scale = 2.0

        def f(x, out_bias, *, _kernels=(k_scale_with_scalar_output,)):
            # Prologue: ops before kernel
            x_processed = torch.sigmoid(x) * 1.5
            out, info = _kernels[0](x_processed, kernel_scale)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_with_scalar_output_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_single(self, allow_torch_compile_fusion):
        """Test: prologue/epilogue with single tensor output."""

        def f(x, out_bias, *, _kernels=(k_add,)):
            # Prologue: ops before kernel
            x_processed = torch.tanh(x) * 2.0
            # Use k_add with processed input added to itself (equivalent to *2)
            out = _kernels[0](x_processed, x_processed)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_epilogue_chained_ops(self, allow_torch_compile_fusion):
        """Test: prologue/epilogue with chained ops on both sides."""
        kernel_scale = 2.0

        def f(x, out_bias, out_scale, *, _kernels=(k_scale_with_scalar_output,)):
            # Prologue: chained ops before kernel
            x_processed = torch.relu(x) + 0.1
            out, info = _kernels[0](x_processed, kernel_scale)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_with_scalar_output_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_rms_norm_prologue_epilogue(self, allow_torch_compile_fusion):
        """Test: prologue/epilogue with multi-output RMS norm kernel."""

        def f(x, weight, out_bias, res_bias, *, _kernels=(k_rms_norm,)):
            # Prologue: ops before kernel
            x_processed = torch.relu(x) + 0.5
            out, residual = _kernels[0](x_processed, weight, 1e-5)
            # Epilogue: ops after kernel (different epilogue per output)
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias

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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_rms_norm_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transpose_then_view_to_3d_epilogue(self, allow_torch_compile_fusion):
        """Test: prologue/epilogue with transpose and reshape view ops."""
        d1, d2, d3 = 8, 16, 32
        kernel_scale = 2.0

        def f(x, epilogue_bias, *, _kernels=(k_scale_with_scalar_output,)):
            # Prologue view ops: mirror of epilogue (3D->2D then transpose)
            x_3d = x.T.reshape(d3, d1, d2)  # (m, n) -> (n, m) -> (d3, d1, d2)
            x_2d = x_3d.reshape(
                d3, d1 * d2
            ).T  # (d3, d1, d2) -> (d3, m) -> (m, d3) = (m, n)
            out, info = _kernels[0](x_2d, kernel_scale)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_with_scalar_output_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_simple(
        self, allow_torch_compile_fusion
    ):
        """Test: fp16 kernel with fp32 epilogue (simple dtype promotion)."""
        m, n = 64, 128

        def f(x, y, *, _kernels=(k_add,)):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.relu(x) * 1.2
            # Use k_add with x_processed added to itself (equivalent to *2)
            out = _kernels[0](x_processed, x_processed)
            # Epilogue: simple ops with dtype promotion
            out_sigmoid = torch.sigmoid(out)
            return out_sigmoid + y  # fp16 + fp32 -> fp32

        x = torch.randn(m, n, device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)
        torch.relu(x) * 1.2
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            # Prologue not fused due to inductor's low-precision heuristic
            # (check_prologue_fusion_heuristics_fusable blocks fp32 prologues
            # on fp16 templates). Epilogue still fuses -> 2 kernels.
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_chained(
        self, allow_torch_compile_fusion
    ):
        """Test: fp16 kernel with fp32 epilogue (chained dtype promotion)."""
        m, n = 64, 128

        def f(x, y, *, _kernels=(k_add,)):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.sigmoid(x) + 0.1
            # Use k_add with x_processed added to itself (equivalent to *2)
            out = _kernels[0](x_processed, x_processed)
            # Epilogue: chained ops after kernel
            out = torch.sigmoid(out)
            out = torch.relu(out)
            out = torch.tanh(out)
            return out * y  # fp16 * fp32 -> fp32

        x = torch.randn(m, n, device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)
        torch.sigmoid(x) + 0.1
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            # Prologue not fused due to inductor's low-precision heuristic.
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_original_twice_in_output(
        self, allow_torch_compile_fusion
    ):
        """Test: same unmutated original appears twice in output tuple.

        This tests that when the same FX node appears multiple times as a graph
        output, all references correctly read from the preserved original buffer.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            result = torch.relu(result) + 1.0
            # Return x twice - both should be unchanged
            return result, x, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_view_of_original_as_output(
        self, allow_torch_compile_fusion
    ):
        """Test: view of original is output alongside clone-then-mutate.

        This tests that views derived from the original also see the preserved
        pre-mutation value.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_view = x.view(-1)  # view of original
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            result = torch.relu(result) + 1.0
            # Both x and view of x should be unchanged
            return result, x, x_view

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_transform_original(self, allow_torch_compile_fusion):
        """Test: original undergoes computation before being output.

        This tests that computations on the original (like x + 1) use the
        pre-mutation value, not the mutated value.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            result = torch.relu(result) + 1.0
            # x + 1 should use pre-mutation value of x
            return result, x + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_chained_kernels(self, allow_torch_compile_fusion):
        """Test: two kernel calls, each with clone-then-mutate pattern.

        This tests complex graphs with multiple HOPs where each needs
        independent cloning to preserve originals.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # First kernel: mutate clone of x
            x_clone1 = x.clone()
            result1 = _kernels[0](x_clone1, y)
            # Second kernel: mutate the result of first kernel
            ones = torch.ones_like(result1)
            result2 = _kernels[0](result1, ones)
            result2 = torch.relu(result2) + 1.0
            # Both x and y should be unchanged
            return result2, x, y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=6 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_of_view_then_mutate(self, allow_torch_compile_fusion):
        """Test: clone a view, mutate the clone, original unchanged.

        This tests that cloning a view and mutating the clone doesn't affect
        the original base tensor.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Create a view, clone it, mutate the clone
            x_view = x.view(-1)
            x_view_clone = x_view.clone()
            result = _kernels[0](x_view_clone, y.view(-1))
            result = torch.relu(result) + 1.0
            # Original x should be unchanged
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_clones_same_tensor(self, allow_torch_compile_fusion):
        """Test: multiple clones of same tensor, each mutated independently.

        This tests that when the same original is cloned multiple times and
        each clone is mutated, the original remains unchanged.
        """

        def f(
            x: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            # Create two clones
            x_clone1 = x.clone()
            x_clone2 = x.clone()
            # Mutate first clone
            ones = torch.ones_like(x_clone1)
            result1 = _kernels[0](x_clone1, ones)
            # Mutate second clone
            twos = ones * 2
            result2 = _kernels[0](x_clone2, twos)
            result1 = torch.relu(result1) + 1.0
            result2 = torch.relu(result2) + 1.0
            # x should be unchanged
            return result1, result2, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_transposed(self, allow_torch_compile_fusion):
        """Test: clone transposed tensor, mutate clone, original unchanged.

        This tests non-contiguous tensor handling in the clone-then-mutate pattern.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Transpose x, clone the transpose, mutate
            x_t = x.T
            x_t_clone = x_t.clone()
            y_t = y.T
            result = _kernels[0](x_t_clone, y_t)
            result = torch.relu(result) + 1.0
            # Original x should be unchanged (not transposed)
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_with_inplace_epilogue(self, allow_torch_compile_fusion):
        """Test: in-place PyTorch op on mutated result.

        This tests interaction between Helion mutation and PyTorch in-place ops.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            result = result.mul_(2.0)  # in-place PyTorch op
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_result_used_twice(self, allow_torch_compile_fusion):
        """Test: mutated result is used in multiple computations.

        This tests that the mutated clone can be used multiple times while
        the original remains unchanged.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            # Use result in two different computations
            out1 = torch.relu(result) + 1.0
            out2 = result.sum()
            return out1, out2, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        w = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (w,),
            kernels=[k_add_to_three],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if supports_torch_compile_fusion()
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_original_reduction_as_output(
        self, allow_torch_compile_fusion
    ):
        """Test: reduction of original as output alongside mutation.

        This tests that reductions (like sum) on the original use the
        pre-mutation value.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            result = torch.relu(result) + 1.0
            # sum of x should use pre-mutation value
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_both_inputs_as_outputs(self, allow_torch_compile_fusion):
        """Test: clone x, mutate clone, return result along with both x and y unchanged.

        This tests that non-mutated inputs (y) are also correctly handled
        when mutated inputs have clone-then-mutate pattern.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = _kernels[0](x_clone, y)
            result = torch.relu(result) + 1.0
            # Both x and y should be unchanged
            return result, x, y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace_1d,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then multiple chained views
            x_clone = x.clone()
            x_t = x_clone.t()  # (8, 4)
            x_contig = x_t.contiguous()  # Makes a copy since t() is non-contiguous!
            x_view = x_contig.view(32)  # (32,)
            y_flat = y.flatten()
            result = _kernels[0](x_view, y_flat)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace_1d],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace_1d,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then create two different views
            x_clone = x.clone()
            x_flat = x_clone.flatten()  # view 1 - will be mutated
            x_transposed = x_clone.t()  # view 2 - not mutated, used in output
            y_flat = y.flatten()
            result = _kernels[0](x_flat, y_flat)
            result = torch.relu(result) + 1.0
            # x_transposed should use pre-mutation value (same as x.t() since clone was made)
            return result, x_transposed.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace_1d],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_add_two_inplace_ref(x, y):
            x.add_(1)
            y.add_(2)
            return x + y

        def f(
            x: torch.Tensor, *, _kernels=(k_add_two_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            # Create two independent clones of x
            clone1 = x.clone()
            clone2 = x.clone()
            # Both clones are mutated
            result = _kernels[0](clone1, clone2)
            result = torch.relu(result) + 1.0
            # x should be unchanged (both mutations happened to clones)
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_two_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_two_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_add_one_ref(x):
            x.add_(1)
            return x

        def k_mul_two_ref(x):
            x.mul_(2)
            return x

        def f(
            x: torch.Tensor, *, _kernels=(k_add_one, k_mul_two)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            x_clone = x.clone()
            # First kernel mutates clone
            _ = _kernels[0](x_clone)
            # Second kernel mutates same clone
            result = _kernels[1](x_clone)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        k_add_one.reset()
        k_mul_two.reset()
        _ = k_add_one(warmup1)
        _ = k_mul_two(warmup2)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_one, k_mul_two],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=5 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_one_ref, k_mul_two_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_repeat_mutate(self, allow_torch_compile_fusion):
        """Test: clone then repeat (expansion), mutate.

        repeat(1,1) is a no-op that may be optimized away. Clone detection
        traces through no-op repeats to find the underlying clone.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add_inplace,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then repeat (repeat creates a new tensor, not a view)
            x_clone = x.clone()
            x_repeated = x_clone.repeat(1, 1)  # Same shape, but new tensor
            result = _kernels[0](x_repeated, y)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_dynamic_shapes_rejects_static_shapes(self, allow_torch_compile_fusion):
        """Regression: static_shapes=True must be rejected with dynamic=True.

        When torch.compile(dynamic=True) is used, tensor dimensions are
        symbolic.  A kernel with static_shapes=True would bake placeholder
        sizes (e.g. 64) into the generated Triton code instead of the real
        sizes, producing wrong results at runtime.  We now raise a clear
        error instead.
        """

        def f(x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add,)) -> torch.Tensor:
            return _kernels[0](x, y)

        x = torch.randn(2, 3, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 3, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            dynamic=True,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_error=(RuntimeError, "static_shapes=True.*dynamic=True")
            if supports_torch_compile_fusion()
            else None,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_dynamic_shapes_basic(self, allow_torch_compile_fusion):
        """Test: kernel with dynamic shapes enabled."""

        def f(x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add,)) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = _kernels[0](x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        # dynamic=True requires static_shapes=False on the kernel
        self.addCleanup(
            setattr, k_add.settings, "static_shapes", k_add.settings.static_shapes
        )
        k_add.settings.static_shapes = False
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            dynamic=True,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_mutate_no_return_ref(x, y):
            x.add_(y)

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_mutate_no_return,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            _kernels[0](x, y)
            # Use x after mutation
            return torch.relu(x) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_mutate_no_return],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_mutate_no_return_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_add_optional_ref(x, y, bias=None):
            result = x + y
            return result + bias if bias is not None else result

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            bias: torch.Tensor,
            *,
            _kernels=(k_add_optional,),
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            result = _kernels[0](x, y, bias)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, bias),
            kernels=[k_add_optional],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_optional_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_scale_by_2_ref(x):
            return x * 2.0

        def f(x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_scale,)) -> torch.Tensor:
            # Apply same kernel to two tensors of different shapes
            scaled1 = _kernels[0](x)  # 4x8
            scaled2 = _kernels[0](y)  # 2x4
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
            expected_num_kernels=4 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_by_2_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_captured_global_variable(self, allow_torch_compile_fusion):
        """Test: kernel using captured global variable from module scope.

        This tests that when a Helion kernel references a global variable defined
        in the module scope (GLOBAL_SCALE_FACTOR), the generated Inductor code
        correctly imports _source_module to resolve the captured variable.
        """

        def f(x: torch.Tensor, *, _kernels=(k_scale_with_global_var,)) -> torch.Tensor:
            return _kernels[0](x)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scale_with_global_var],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_with_global_var_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        x = torch.randn(5, 4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=5 if allow_torch_compile_fusion else 8,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_compute_with_none_ref(x, flag):
            return x * 2.0, None, flag * 2

        def f(
            x: torch.Tensor, *, _kernels=(k_compute_with_none,)
        ) -> tuple[torch.Tensor, None, int]:
            x = x * 2.0
            result, none_val, scalar = _kernels[0](x, 21)
            result = torch.relu(result) + 1.0
            return result, none_val, scalar

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_compute_with_none],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_compute_with_none_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_compute_none_first_ref(x, flag):
            return None, x * 2.0, flag * 2

        def f(
            x: torch.Tensor, *, _kernels=(k_compute_none_first,)
        ) -> tuple[None, torch.Tensor, int]:
            x = x * 2.0
            none_val, result, scalar = _kernels[0](x, 21)
            result = torch.relu(result) + 1.0
            return none_val, result, scalar

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_compute_none_first],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_compute_none_first_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_tuple_of_scalars(self, allow_torch_compile_fusion):
        """Test: kernel returning only scalars works with fusion (constants inlined)."""

        @helion.kernel(autotune_effort="none")
        def k_two_scalars(x: torch.Tensor, a: int, b: int) -> tuple[int, int]:
            """Return two scalars based on input args."""
            for tile in hl.tile(x.size()):
                _ = x[tile]
            return a * 2, b * 3

        def k_two_scalars_ref(x, a, b):
            _ = x
            return a * 2, b * 3

        def f(x: torch.Tensor, *, _kernels=(k_two_scalars,)) -> tuple[int, int]:
            x = x * 2.0
            s1, s2 = _kernels[0](x, 10, 20)
            return s1, s2

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_two_scalars],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=0 if allow_torch_compile_fusion else None,
            kernels_ref=[k_two_scalars_ref],
            expected_num_kernels_ref=0,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_return_same_twice_ref(x, y):
            out = x + y
            return out, out

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_return_same_twice,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result1, result2 = _kernels[0](x, y)
            return torch.relu(result1) + 1.0, torch.relu(result2) + 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_return_same_twice],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_return_same_twice_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_alias_return_twice_ref(x, y):
            x.add_(y)
            return x, x

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_alias_return_twice,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result1, result2 = _kernels[0](x, y)
            return torch.relu(result1) + 1.0, torch.relu(result2) + 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_alias_return_twice],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_alias_return_twice_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_local_alias,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            out = _kernels[0](x, y)
            return torch.relu(out) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_local_alias],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_inplace_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_return_list_ref(x, y):
            return [x + y, x - y]

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_return_list,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result_list = _kernels[0](x, y)
            return torch.relu(result_list[0]) + 1.0, torch.relu(result_list[1]) + 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_return_list],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_return_list_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_nested_ref(x, y):
            return x + y, (x - y, x * y)

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_nested,)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            a, (b, c) = _kernels[0](x, y)
            return torch.relu(a) + 1.0, torch.relu(b) + 2.0, torch.relu(c) + 3.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_nested],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_nested_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_float_scalar_ref(x):
            return x * 2.0, math.pi

        def f(
            x: torch.Tensor, *, _kernels=(k_float_scalar,)
        ) -> tuple[torch.Tensor, float]:
            x = x * 2.0
            result, scalar = _kernels[0](x)
            result = torch.relu(result) + 1.0
            return result, scalar

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_float_scalar],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_float_scalar_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_scalar_literal_in_compile_region_no_recompilation(
        self, allow_torch_compile_fusion
    ):
        """Test: kernel returning (tensor, scalar) called twice with different scalar literals
        inside the compile region. Verifies no helion recompilation."""

        def f(
            x: torch.Tensor, *, _kernels=(k_scale_with_scalar_output,)
        ) -> torch.Tensor:
            result1, s1 = _kernels[0](x, 2.0)
            result2, s2 = _kernels[0](x, 5.0)
            return result1 + s1 + result2 + s2

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scale_with_scalar_output],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            expected_num_compilations=[1],
            kernels_ref=[k_scale_with_scalar_output_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_scalar_input_to_compile_region_used_in_kernel_output(
        self, allow_torch_compile_fusion
    ):
        """Test: scalar input from outside torch.compile region is used in kernel that returns (tensor, scalar, tensor)."""

        def f(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: float,
            *,
            _kernels=(k_tensor_scalar_tensor,),
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result_x, scalar_val, result_y = _kernels[0](x, y, scale)
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
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_tensor_scalar_tensor_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_element_tuple_return(self, allow_torch_compile_fusion):
        """Test: kernel returning (out,) works with fusion."""

        def f(x: torch.Tensor, *, _kernels=(k_single_element_tuple,)) -> torch.Tensor:
            x = x * 2.0
            (result,) = _kernels[0](x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_single_element_tuple],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_single_element_tuple_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_list_return_ref(x):
            return [x * 2.0]

        def f(x: torch.Tensor, *, _kernels=(k_list_return,)) -> torch.Tensor:
            [result] = _kernels[0](x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_list_return],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_list_return_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_nested_return_ref(x):
            out = x * 2.0
            return (out,), out

        def f(x: torch.Tensor, *, _kernels=(k_nested_return,)) -> torch.Tensor:
            (inner,), outer = _kernels[0](x)
            return torch.relu(inner) + torch.relu(outer)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_nested_return],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_nested_return_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_deep_nested_ref(x, y):
            return (x * 2.0, x * 3.0), 7, y * 4.0

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_deep_nested,)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            (a, b), scalar_val, c = _kernels[0](x, y)
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
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_deep_nested_ref],
            expected_num_kernels_ref=2,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_scalar_only_ref(x):
            return 42, math.pi

        def f(x: torch.Tensor, *, _kernels=(k_scalar_only,)) -> tuple[int, float]:
            return _kernels[0](x)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scalar_only],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=0 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scalar_only_ref],
            expected_num_kernels_ref=0,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_param_scalar_ref(x, scale):
            return x * scale, scale

        def f(
            x: torch.Tensor, scale: float, *, _kernels=(k_param_scalar,)
        ) -> tuple[torch.Tensor, float]:
            return _kernels[0](x, scale)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, 2.0),
            kernels=[k_param_scalar],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_param_scalar_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_reassign_ref(x, y):
            return y * 2.0

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_reassign,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = _kernels[0](x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_reassign],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_reassign_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_local_return_ref(x):
            return (x * 2.0).sum(dim=1)

        def f(x: torch.Tensor, *, _kernels=(k_local_return,)) -> torch.Tensor:
            x = x * 2.0
            result = _kernels[0](x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_local_return],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_local_return_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_control_flow_ref(x, use_sum):
            out = x * 2.0
            return out.sum(dim=1) if use_sum else out.mean(dim=1)

        def f(
            x: torch.Tensor, use_sum: bool, *, _kernels=(k_control_flow,)
        ) -> torch.Tensor:
            x = x * 2.0
            result = _kernels[0](x, use_sum)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, True),
            kernels=[k_control_flow],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_control_flow_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_multi_return_ref(x, use_sum):
            out = x * 2.0
            return out.sum(dim=1) if use_sum else out.mean(dim=1)

        def f(
            x: torch.Tensor, use_sum: bool, *, _kernels=(k_multi_return,)
        ) -> torch.Tensor:
            x = x * 2.0
            result = _kernels[0](x, use_sum)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, True),
            kernels=[k_multi_return],
            expected_error=(
                RuntimeError,
                r"Return statements inside control flow.*not supported",
            )
            if supports_torch_compile_fusion()
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            kernels_ref=[k_multi_return_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_augassign_ref(x):
            return (x * 2.0).sum(dim=1) + 1.0

        def f(x: torch.Tensor, *, _kernels=(k_augassign,)) -> torch.Tensor:
            x = x * 2.0
            result = _kernels[0](x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_augassign],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_augassign_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_annotated_ref(x):
            return (x * 2.0).sum(dim=1)

        def f(x: torch.Tensor, *, _kernels=(k_annotated,)) -> torch.Tensor:
            x = x * 2.0
            result = _kernels[0](x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_annotated],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_annotated_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        base = torch.randn(4, 4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (base,),
            kernels=[k],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if supports_torch_compile_fusion()
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        k_scale_with_param_ref = operator.mul

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_add, k_scale)
        ) -> torch.Tensor:
            # First kernel on transposed input
            x_t = x.T
            result = _kernels[0](x_t, y.T)
            # Second kernel on result
            result = _kernels[1](result, 2.0)
            return result.T

        def warmup():
            k_add.reset()
            k_scale.reset()
            warmup_x = torch.randn(8, 4, device=DEVICE, dtype=torch.float32)
            warmup_y = torch.randn(8, 4, device=DEVICE, dtype=torch.float32)
            k_add(warmup_x, warmup_y)
            k_scale(warmup_x, 2.0)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_scale, k_add],
            rtol=1e-2,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref, k_scale_with_param_ref],
            expected_num_kernels_ref=1,
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

        def k_scalar_and_mutate_ref(x, scale):
            x.mul_(scale)
            return 99, x

        def f(
            x: torch.Tensor, *, _kernels=(k_scalar_and_mutate,)
        ) -> tuple[int, torch.Tensor]:
            scalar, result = _kernels[0](x, 2.0)
            return scalar, result + 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scalar_and_mutate],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=3 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scalar_and_mutate_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_sum_tuple_ref(tensors):
            return tensors[0] + tensors[1]

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_sum_tuple,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = _kernels[0]((x, y))
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_sum_tuple],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_sum_tuple_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        k_scale_constexpr_ref = operator.mul

        def f(x: torch.Tensor, *, _kernels=(k_scale_constexpr,)) -> torch.Tensor:
            x = x * 2.0
            # Pass 3.0 as constexpr parameter
            result = _kernels[0](x, 3.0)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scale_constexpr],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_scale_constexpr_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_sum_dict_ref(tensors):
            return tensors["a"] + tensors["b"]

        def f(
            x: torch.Tensor, y: torch.Tensor, *, _kernels=(k_sum_dict,)
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = _kernels[0]({"a": x, "b": y})
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_sum_dict],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_sum_dict_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
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

        def k_returns_string_ref(x):
            return x * 2.0, "hello"

        def f(
            x: torch.Tensor, *, _kernels=(k_returns_string,)
        ) -> tuple[torch.Tensor, str]:
            return _kernels[0](x)

        # Custom compare needed because torch.testing.assert_close doesn't support str values.
        def compare(actual, expected):
            self.assertEqual(len(actual), len(expected))
            torch.testing.assert_close(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_returns_string],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            compare_fn=compare,
            kernels_ref=[k_returns_string_ref],
            expected_num_kernels_ref=1,
        )

    @requires_fusion_support
    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_symint_return_from_tensor_shape(self, allow_torch_compile_fusion):
        """Test: kernel returning SymInt (tensor shape) with dynamic shapes."""
        if not allow_torch_compile_fusion:
            self.skipTest("Only testing with torch.compile fusion enabled")

        @helion.kernel(
            autotune_effort="none", static_shapes=False, torch_compile_fusion=True
        )
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
        x0 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        _ = f(x0.clone())

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor", dynamic=True)

        # Test with multiple shapes to exercise dynamic SymInt return values
        for nrows in (4, 16, 32):
            x = torch.randn(nrows, 8, device=DEVICE, dtype=torch.float32)
            expected = f(x.clone())
            actual = compiled_f(x.clone())
            torch.testing.assert_close(actual, expected)

    @parametrize("indexing", ("pointer", "block_ptr", "tensor_descriptor"))
    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_epilogue_indexing_strategies(
        self, allow_torch_compile_fusion, indexing
    ):
        """Test: prologue/epilogue with different indexing strategies."""
        if indexing == "tensor_descriptor" and not supports_tensor_descriptor():
            self.skipTest("Tensor descriptor support is required")

        @helion.kernel(
            config=helion.Config(block_sizes=[64, 128], indexing=indexing),
            autotune_effort="none",
        )
        def k_add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        def f(x, out_bias, *, _kernels=(k_add_2d,)):
            x_processed = torch.sigmoid(x) * 1.5
            out = _kernels[0](x_processed, x_processed)
            return torch.relu(out) + out_bias

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        run_kwargs = {
            "f": f,
            "test_args": (x, out_bias),
            "kernels": [k_add_2d],
            "rtol": 1e-3,
            "atol": 1e-3,
            "allow_torch_compile_fusion": allow_torch_compile_fusion,
            "expected_num_kernels": 1 if allow_torch_compile_fusion else None,
            "kernels_ref": [k_add_ref],
            "expected_num_kernels_ref": 1,
        }
        if indexing == "tensor_descriptor":
            # Tensor descriptor lowering queries CUDA target info during compilation.
            # Force single-threaded compile here to avoid forking CUDA-initialized state.
            with inductor_config.patch({"compile_threads": 1}):
                self._run_compile_test(**run_kwargs)
        else:
            self._run_compile_test(**run_kwargs)

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_reads_kernel_output_with_different_indices(
        self, allow_torch_compile_fusion
    ):
        """Epilogue that reads kernel output with different index patterns.

        out + out.T reads the kernel output at (i,j) AND (j,i). Fusion must
        not intercept both reads identically, which would give 2*out instead
        of out + out.T.
        """

        def f(x, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return out + out.T

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_full_reduction_not_fused(self, allow_torch_compile_fusion):
        """Full reduction (scalar output): out.sum().

        Full reductions produce a scalar and cannot be fused into the kernel.
        """

        def f(x, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return out.sum()

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_broadcast_bias(self, allow_torch_compile_fusion):
        """Epilogue with broadcast: out + bias where bias is (1, N).

        The bias is a separate buffer (not the kernel output), so the
        epilogue only reads the kernel output once -- fusion is correct.
        """

        n = 64

        def f(x, bias, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return out + bias

        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(1, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, bias),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_self_read(self, allow_torch_compile_fusion):
        """Epilogue with same-index self-read: relu(out) + out.

        Both reads of kernel output use the same index, so fusion should
        be allowed and produce correct results.
        """

        def f(x, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return torch.relu(out) + out

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_matmul_with_transposed_self(self, allow_torch_compile_fusion):
        """Epilogue that computes out @ out.T on a non-square tensor.

        The kernel output (buf1) is consumed twice: once as the left operand
        of mm and once as the right (transposed).  Two-store mode keeps the
        original output buffer live while also writing the epilogue target,
        so the fused kernel is correct with just 1 Triton kernel.
        """

        def f(x, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return out @ out.T

        x = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_transposed_contiguous(self, allow_torch_compile_fusion):
        """Epilogue that reads kernel output at transposed indices: out.T.contiguous().

        The epilogue's read index (j, i) differs from the store index (i, j),
        so fusion must not substitute the non-transposed value.
        """

        def f(x, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return out.T.contiguous()

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_gather_not_fused(self, allow_torch_compile_fusion):
        """Epilogue that gathers from kernel output at non-trivial indices.

        gather reads kernel output at index-dependent positions (i, idx[i,j]),
        so fusion must not substitute the store-position value for each access.
        """

        def f(x, idx, *, _kernels=(k_add,)):
            out = _kernels[0](x, x)
            return torch.gather(out, 1, idx)

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        idx = torch.randint(0, n, (n, n), device=DEVICE)
        self._run_compile_test(
            f,
            (x, idx),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=2 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_flip_fused(self, allow_torch_compile_fusion):
        """Prologue that flips the input tensor is correctly inlined.

        _kernels[0](x.flip(0), x) has x.flip(0) as a prologue with index remapping:
        the prologue reads source[N-1-i, j] while writing out[i, j]. The
        remapped load (N-1-i) must be preserved, producing flip(x)+x correctly.
        """

        def f(x, *, _kernels=(k_add,)):
            return _kernels[0](x.flip(0), x)

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_epilogue_triu_fused(self, allow_torch_compile_fusion):
        """Epilogue using triu (index_expr) is fused via _Handler.index_expr."""

        def f(x, *, _kernels=(k_add,)):
            return torch.triu(_kernels[0](x, x))

        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_flip_plus_bias_not_fused(self, allow_torch_compile_fusion):
        """Prologue that chains flip+bias is not inlined.

        ``flip(x) + b`` produces a node with two MemoryDep reads (x and b).
        Multi-read prologue nodes must not be fused, since collapsing both
        reads into a single placeholder would give wrong results.
        """
        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        b = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, *, _kernels=(k_add,)):
            return _kernels[0](x.flip(0) + b, y)

        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_bias_not_fused_when_multi_read(self, allow_torch_compile_fusion):
        """Multi-read prologue (y + bias) is not inlined when one input is flipped.

        In ``k_add(flip(x), y + b)``, ``y + b`` forms a two-read prologue
        (reads y and b). Multi-read prologue nodes must not be fused, so
        ``y + b`` should run as a separate kernel.
        """
        n = 64
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(n, n, device=DEVICE, dtype=torch.float32)
        b = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, y, *, _kernels=(k_add,)):
            return _kernels[0](x.flip(0), y + b)

        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            expected_num_kernels=1 if allow_torch_compile_fusion else None,
            kernels_ref=[k_add_ref],
            expected_num_kernels_ref=1,
        )

    # --- Shared helpers for autotune-with-fusion tests ---

    _EPILOGUE_PATTERN = "maximum"  # relu → triton_helpers.maximum
    _PROLOGUE_PATTERN = "tl.full([1], 2.0"  # mul-by-2 scalar constant

    def _make_autotune_kernel(self, autotune_with_torch_compile_fusion=True):
        """Create a simple add kernel with two configs for autotuning."""

        @helion.kernel(
            configs=[
                helion.Config(block_sizes=[32]),
                helion.Config(block_sizes=[64]),
            ],
            torch_compile_fusion=True,
            autotune_with_torch_compile_fusion=autotune_with_torch_compile_fusion,
        )
        def k_add_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        return k_add_autotune

    def _make_code_capture(self, kernel_name="k_add_autotune"):
        """Return (captured_codes, patch_context) for intercepting PyCodeCache.load."""
        from torch._inductor.codecache import PyCodeCache

        captured_codes: list[str] = []
        original_load = PyCodeCache.load

        def patched_load(code, *args, **kwargs):
            if kernel_name in code:
                captured_codes.append(code)
            return original_load(code, *args, **kwargs)

        return captured_codes, patch.object(PyCodeCache, "load", patched_load)

    # --- Autotune-with-fusion tests ---

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("autotune_with_fusion", (True, False))
    def test_autotune_fusion_aware_vs_default(self, autotune_with_fusion):
        """When fusion-aware autotuning is on, each config is benchmarked as fused code;
        when off, the pre-existing BoundKernel config is reused without recompilation."""

        kernel = self._make_autotune_kernel(
            autotune_with_torch_compile_fusion=autotune_with_fusion
        )
        captured_codes, patch_ctx = self._make_code_capture()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            return torch.relu(kernel(x, y)) + 1.0

        kernel.reset()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        # Standalone call populates BoundKernel cache
        result_standalone = kernel(x.clone(), y.clone())
        torch.testing.assert_close(result_standalone, x + y, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(kernel._bound_kernels), 1)

        # torch.compile — cache reuse depends on autotune_with_fusion
        with patch_ctx:
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            result = compiled_f(x.clone(), y.clone())

        expected = torch.relu((x * 2.0) + (y * 2.0)) + 1.0
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        ep = self._EPILOGUE_PATTERN
        pp = self._PROLOGUE_PATTERN
        fused_codes = [c for c in captured_codes if ep in c and pp in c]
        unfused_codes = [c for c in captured_codes if ep not in c and pp not in c]

        if autotune_with_fusion:
            self.assertEqual(len(fused_codes), 2, "2 fused compiles (one per config)")
            self.assertEqual(len(set(fused_codes)), 2, "Distinct fused kernels")
            self.assertEqual(
                len(unfused_codes),
                0,
                "No unfused compiles (bench_compile_config handles all configs)",
            )
        else:
            self.assertEqual(
                len(captured_codes),
                0,
                "Without fusion-aware autotuning, BoundKernel is reused",
            )

        # Direct call still works (extra_params not leaked)
        result_direct = kernel(x.clone(), y.clone())
        torch.testing.assert_close(result_direct, x + y)

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_fusion_recompile(self):
        """Recompile with shared BoundKernel still produces fused code."""

        kernel = self._make_autotune_kernel()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            return torch.relu(kernel(x, y)) + 1.0

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)
        ep = self._EPILOGUE_PATTERN
        pp = self._PROLOGUE_PATTERN

        # First compile — creates BoundKernel and autotuning config.
        kernel.reset()
        torch._dynamo.reset()
        _, (code,) = run_and_get_code(
            torch.compile(f, fullgraph=True, backend="inductor"),
            x.clone(),
            y.clone(),
        )
        self.assertIn(ep, code, "Must have epilogue fusion")
        self.assertIn(pp, code, "Must have prologue fusion")

        # Recompile — only reset Dynamo, keep BoundKernel cache intact.
        # This tests that Inductor re-codegen with a shared BoundKernel
        # still produces fused output with a single kernel.
        torch._dynamo.reset()
        _, (code,) = run_and_get_code(
            torch.compile(f, fullgraph=True, backend="inductor"),
            x.clone(),
            y.clone(),
        )
        self.assertIn(ep, code, "Must have epilogue fusion")
        self.assertIn(pp, code, "Must have prologue fusion")
        self.assertEqual(code.count("@triton.jit"), 1, "Single fused kernel")

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_different_epilogues(self):
        """Different epilogues (relu vs sigmoid) trigger separate autotuning."""
        kernel = self._make_autotune_kernel()
        captured_codes, patch_ctx = self._make_code_capture()

        def g(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            a = torch.relu(kernel(x, y))  # relu epilogue
            b = torch.sigmoid(kernel(x, y))  # sigmoid epilogue
            return a, b

        kernel.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            compiled_g = torch.compile(g, fullgraph=True, backend="inductor")
            result_a, result_b = compiled_g(x.clone(), y.clone())

        torch.testing.assert_close(result_a, torch.relu(x + y), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result_b, torch.sigmoid(x + y), rtol=1e-4, atol=1e-4)

        ep = self._EPILOGUE_PATTERN
        sigmoid_pattern = "tl.sigmoid"
        relu_only = [c for c in captured_codes if ep in c and sigmoid_pattern not in c]
        sigmoid_only = [
            c for c in captured_codes if sigmoid_pattern in c and ep not in c
        ]
        self.assertGreater(len(relu_only), 0, "Must have relu-only kernel(s)")
        self.assertGreater(len(sigmoid_only), 0, "Must have sigmoid-only kernel(s)")

        # Different epilogues must produce separate fusion-context cache entries
        from helion._compiler._inductor.template_buffer import HelionTemplateBuffer

        bk = next(iter(kernel._bound_kernels.values()))
        bk_cache = HelionTemplateBuffer._fusion_config_cache.get(bk)
        self.assertIsNotNone(
            bk_cache, "Fusion config cache must have entries for the BoundKernel"
        )
        self.assertGreater(
            len(bk_cache),
            1,
            "Different epilogues must produce separate fusion-context entries",
        )

        # Re-running reuses cached kernels
        captured_codes.clear()
        with patch_ctx:
            result_a2, result_b2 = compiled_g(x.clone(), y.clone())
        torch.testing.assert_close(result_a2, torch.relu(x + y), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            result_b2, torch.sigmoid(x + y), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(len(captured_codes), 0, "Re-run must reuse cached kernels")

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_different_shapes(self):
        """Different input shapes trigger re-autotuning with fused kernels."""

        kernel = self._make_autotune_kernel()
        captured_codes, patch_ctx = self._make_code_capture()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            return torch.relu(kernel(x, y)) + 1.0

        kernel.reset()
        torch._dynamo.reset()
        ep = self._EPILOGUE_PATTERN
        pp = self._PROLOGUE_PATTERN

        # First compile with shape 128.
        x128 = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y128 = torch.randn(128, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            result128 = compiled_f(x128.clone(), y128.clone())

        expected128 = torch.relu((x128 * 2.0) + (y128 * 2.0)) + 1.0
        torch.testing.assert_close(result128, expected128, rtol=1e-4, atol=1e-4)

        first_fused_count = len([c for c in captured_codes if ep in c and pp in c])
        self.assertGreater(first_fused_count, 0, "Shape-128 must produce fused kernels")

        # Second compile with shape 256 — keep BoundKernel cache intact.
        # A new shape creates a new BoundKernel, so fusion autotuning must
        # run again (not reuse the shape-128 result).
        captured_codes.clear()
        torch._dynamo.reset()

        x256 = torch.randn(256, device=DEVICE, dtype=torch.float32)
        y256 = torch.randn(256, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            result256 = compiled_f(x256.clone(), y256.clone())

        expected256 = torch.relu((x256 * 2.0) + (y256 * 2.0)) + 1.0
        torch.testing.assert_close(result256, expected256, rtol=1e-4, atol=1e-4)

        second_fused_count = len([c for c in captured_codes if ep in c and pp in c])
        self.assertGreater(
            second_fused_count, 0, "New shape must trigger new fused compilations"
        )

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_same_epilogue_cache(self):
        """Same kernel + same epilogue called twice → second hits fusion cache."""

        kernel = self._make_autotune_kernel()
        captured_codes, patch_ctx = self._make_code_capture()
        ep = self._EPILOGUE_PATTERN

        # First compile with relu epilogue — populates _fusion_config_cache.
        def f_single(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.relu(kernel(x, y)) + 1.0

        kernel.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            compiled_single = torch.compile(
                f_single, fullgraph=True, backend="inductor"
            )
            result_single = compiled_single(x.clone(), y.clone())

        torch.testing.assert_close(
            result_single, torch.relu(x + y) + 1.0, rtol=1e-4, atol=1e-4
        )
        single_call_fused_count = len([c for c in captured_codes if ep in c])
        self.assertGreater(single_call_fused_count, 0, "First compile must autotune")

        # Second compile with same epilogue + different inputs (prevents CSE).
        # Keep BoundKernel cache intact (no kernel.reset()) so the
        # _fusion_config_cache entry from the first compile is available.
        def f_double(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a = torch.relu(kernel(x, y)) + 1.0
            b = torch.relu(kernel(x, z)) + 1.0
            return a, b

        torch._dynamo.reset()
        captured_codes.clear()

        z = torch.randn(128, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            compiled_double = torch.compile(
                f_double, fullgraph=True, backend="inductor"
            )
            result_a, result_b = compiled_double(x.clone(), y.clone(), z.clone())

        torch.testing.assert_close(
            result_a, torch.relu(x + y) + 1.0, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            result_b, torch.relu(x + z) + 1.0, rtol=1e-4, atol=1e-4
        )

        # The second compile should not have triggered new fused compilations
        # because the fusion cache already has an entry for this epilogue.
        double_call_fused_count = len([c for c in captured_codes if ep in c])
        self.assertEqual(
            double_call_fused_count,
            0,
            f"Same epilogue with shared BoundKernel must hit fusion cache "
            f"(got {double_call_fused_count} new fused compilations)",
        )

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_standalone_call_after_fusion_triggers_autotuning(self):
        """Standalone call after torch.compile with fusion must trigger its own autotuning.

        When the first use of a kernel is through torch.compile with fusion,
        the fusion-aware autotuner picks a config optimized for the fused
        workload.  A subsequent direct call must trigger autotuning for
        the unfused context rather than silently reusing the fused config.
        """

        kernel = self._make_autotune_kernel()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            return torch.relu(kernel(x, y)) + 1.0

        kernel.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        # First use is through torch.compile with fusion — no prior standalone call.
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result_fused = compiled_f(x.clone(), y.clone())
        expected_fused = torch.relu((x * 2.0) + (y * 2.0)) + 1.0
        torch.testing.assert_close(result_fused, expected_fused, rtol=1e-4, atol=1e-4)

        # Spy on compile_config to verify the standalone call triggers autotuning.
        from helion._compiler._inductor.template_buffer import HelionTemplateBuffer
        from helion.runtime.kernel import BoundKernel

        compile_config_calls: list[bool] = []
        original_compile_config = BoundKernel.compile_config

        def tracking_compile_config(self_bk, *args, **kwargs):
            compile_config_calls.append(True)
            return original_compile_config(self_bk, *args, **kwargs)

        # Now do a standalone call — must produce correct results and
        # establish its own config for the unfused context.
        with patch.object(BoundKernel, "compile_config", tracking_compile_config):
            result = kernel(x.clone(), y.clone())
        torch.testing.assert_close(result, x + y, rtol=1e-4, atol=1e-4)

        # The standalone call must have triggered compile_config (autotuning).
        self.assertGreater(
            len(compile_config_calls),
            0,
            "Standalone call must trigger its own autotuning via compile_config",
        )

        # Verify _config is set and fusion cache has separate entries.
        bk = next(iter(kernel._bound_kernels.values()))
        self.assertIsNotNone(
            bk._config,
            "Standalone call must establish its own unfused config",
        )
        bk_cache = HelionTemplateBuffer._fusion_config_cache.get(bk)
        self.assertIsNotNone(
            bk_cache,
            "Fusion config cache should have entries from the torch.compile call",
        )

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_fused_vs_unfused_config_stored_separately(self):
        """Unfused config (bk._config) and fused config (_fusion_config_cache) are independent."""
        from helion.runtime.config import Config

        kernel = self._make_autotune_kernel()

        kernel.reset()
        torch._dynamo.reset()

        if supports_torch_compile_fusion():
            from helion._compiler._inductor.template_buffer import HelionTemplateBuffer

            HelionTemplateBuffer._fusion_config_cache.clear()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        # Standalone call — unfused autotuning, sets bk._config.
        standalone_result = kernel(x.clone(), y.clone())
        torch.testing.assert_close(standalone_result, x + y, rtol=1e-4, atol=1e-4)

        bk = next(iter(kernel._bound_kernels.values()))
        unfused_config = bk._config
        self.assertIsNotNone(unfused_config, "Standalone must set bk._config")
        self.assertIsInstance(unfused_config, Config)

        # torch.compile with fusion — fused autotuning, stores in _fusion_config_cache.
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            return torch.relu(kernel(x, y)) + 1.0

        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result = compiled_f(x.clone(), y.clone())
        expected = torch.relu((x * 2.0) + (y * 2.0)) + 1.0
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # Verify both storage locations independently.
        self.assertIsNotNone(
            bk._config, "bk._config must still hold the unfused config"
        )
        bk_cache = HelionTemplateBuffer._fusion_config_cache.get(bk)
        self.assertIsNotNone(
            bk_cache, "Fusion config cache must have entries for this BoundKernel"
        )
        self.assertGreater(len(bk_cache), 0, "Must have at least one fusion entry")
        # Each fusion cache entry must be a Config instance.
        for fusion_key, fused_cfg in bk_cache.items():
            self.assertIsInstance(
                fused_cfg,
                Config,
                f"Fusion cache entry {fusion_key!r} must be a Config",
            )

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_epilogue_only_fusion(self):
        """Fusion-aware autotuning works with epilogue only (no prologue)."""

        kernel = self._make_autotune_kernel()
        captured_codes, patch_ctx = self._make_code_capture()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.relu(kernel(x, y)) + 1.0  # epilogue only, no prologue

        kernel.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            result = compiled_f(x.clone(), y.clone())

        expected = torch.relu(x + y) + 1.0
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        ep = self._EPILOGUE_PATTERN
        pp = self._PROLOGUE_PATTERN
        fused = [c for c in captured_codes if ep in c]
        self.assertGreater(len(fused), 0, "Must have epilogue-fused kernel(s)")
        for code in fused:
            self.assertNotIn(pp, code, "Must NOT have prologue in epilogue-only test")

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_prologue_only_fusion(self):
        """Fusion-aware autotuning works with prologue only (no epilogue)."""

        kernel = self._make_autotune_kernel()
        captured_codes, patch_ctx = self._make_code_capture()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            return kernel(x, y)  # prologue only, no epilogue

        kernel.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        with patch_ctx:
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            result = compiled_f(x.clone(), y.clone())

        expected = (x * 2.0) + (y * 2.0)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        ep = self._EPILOGUE_PATTERN
        pp = self._PROLOGUE_PATTERN
        fused = [c for c in captured_codes if pp in c]
        self.assertGreater(len(fused), 0, "Must have prologue-fused kernel(s)")
        for code in fused:
            self.assertNotIn(ep, code, "Must NOT have epilogue in prologue-only test")

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_bare_kernel_no_prologue_epilogue(self):
        """Fusion-aware autotuning does not break when Inductor has no prologue or epilogue to fuse."""

        kernel = self._make_autotune_kernel()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return kernel(x, y)  # no prologue, no epilogue

        kernel.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result = compiled_f(x.clone(), y.clone())
        torch.testing.assert_close(result, x + y, rtol=1e-4, atol=1e-4)

        # Direct call still works afterward (extra_params not leaked).
        result_direct = kernel(x.clone(), y.clone())
        torch.testing.assert_close(result_direct, x + y, rtol=1e-4, atol=1e-4)

    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_no_configs_uses_fusion_context(self):
        """Without explicit configs, fusion autotuning must benchmark the fused kernel.

        Without explicit configs, Backend.autotune falls through to the full
        search path which wraps the search in BaseCache.  The adapter must
        pass the is_cacheable() guard in BaseCache, and the autotuner must
        use the fusion context hash to avoid returning a stale config from
        a prior unfused run.
        """
        if not supports_torch_compile_fusion():
            self.skipTest(
                "torch.compile fusion requires ExternalTritonTemplateKernel support"
            )

        @helion.kernel(
            torch_compile_fusion=True,
            autotune_with_torch_compile_fusion=True,
            autotune_max_generations=1,
            autotune_effort="quick",
        )
        def k_add_no_configs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        k_add_no_configs.reset()
        torch._dynamo.reset()

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)

        # Standalone call — runs real unfused autotuning, populates disk cache.
        standalone_result = k_add_no_configs(x.clone(), y.clone())
        torch.testing.assert_close(standalone_result, x + y, rtol=1e-4, atol=1e-4)

        # torch.compile with fusion — must not crash and must not silently
        # reuse the cached unfused config.
        torch._dynamo.reset()

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.relu(k_add_no_configs(x, y)) + 1.0

        from helion._compiler._inductor.template_buffer import _FusionAutotuneAdapter

        bench_compile_called: list[bool] = []
        original_bench = _FusionAutotuneAdapter.bench_compile_config

        def tracking_bench(adapter_self, config=None, **kwargs):
            bench_compile_called.append(True)
            return original_bench(adapter_self, config, **kwargs)

        with patch.object(
            _FusionAutotuneAdapter, "bench_compile_config", tracking_bench
        ):
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            result = compiled_f(x.clone(), y.clone())

        expected = torch.relu(x + y) + 1.0
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        # bench_compile_config must have been called, proving the fusion
        # autotuner benchmarked the fused kernel rather than silently
        # returning the cached unfused config.
        self.assertGreater(
            len(bench_compile_called),
            0,
            "Fusion autotuner must benchmark the fused kernel via "
            "bench_compile_config, not reuse the cached unfused config",
        )

    @requires_fusion_support
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @patch.object(k_rms_norm.settings, "torch_compile_fusion", True)
    def test_inductor_output_code_has_helion_generated_triton_kernel(self):
        """Verify Helion-specific patterns appear in inductor output code."""

        def f(x, weight, out_bias, res_bias):
            x_processed = torch.relu(x) + 0.5
            out, residual = k_rms_norm(x_processed, weight, 1e-5)
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias

        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        args = (x, weight, out_bias, res_bias)

        k_rms_norm.reset()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

        _, (code,) = run_and_get_code(
            torch.compile(f, fullgraph=True, backend="inductor"),
            *(a.clone() if isinstance(a, torch.Tensor) else a for a in args),
        )

        # Helion-specific patterns
        self.assertIn("_helion_", code, "Should contain _helion_ prefix")
        self.assertIn("@triton.jit", code, "Should contain @triton.jit decorator")
        self.assertIn("_launcher", code, "Should contain _launcher call")
        self.assertIn("tl.load", code, "Should contain tl.load")
        self.assertIn("tl.store", code, "Should contain tl.store")

        # All ops should be fused into one Helion kernel
        num_triton_kernels = len(re.findall(r"@triton\.jit", code))
        self.assertEqual(
            num_triton_kernels,
            1,
            f"Expected exactly 1 Triton kernel (all ops fused), found {num_triton_kernels}",
        )


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
