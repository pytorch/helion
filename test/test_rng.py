from __future__ import annotations

import importlib
import os
from typing import Callable
import unittest
from unittest.mock import patch

import torch
from torch._inductor import inductor_prims

import helion
from helion._compiler.backend import PallasBackend
from helion._compiler.rng_utils import philox_rand_ref
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
from helion._testing import skipUnlessCuteAvailable
from helion._testing import xfailIfCute
from helion._testing import xfailIfPallas
import helion.language as hl
from helion.runtime.config import Config
from helion.runtime.ref_mode import is_ref_mode_enabled
from helion.runtime.settings import _get_backend

try:
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
except ModuleNotFoundError:
    triton = None
    tl = None


def _assert_uses_philox(testcase: TestCase, code: str) -> None:
    if os.environ.get("HELION_INTERPRET") == "1":
        return
    testcase.assertTrue(
        ("3528531795" in code and "3449720151" in code)
        or ("36183" in code and "52638" in code and "8019" in code and "53841" in code),
        "Philox round constants not found in generated code",
    )
    testcase.assertTrue(
        ("2654435769" in code or "-1640531527" in code)
        and ("3144134277" in code or "-1150833019" in code),
        "Philox key schedule constants not found in generated code",
    )


def _assert_bitwise_equal_float(
    testcase: TestCase, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    testcase.assertTrue(
        torch.equal(
            actual.detach().cpu().view(torch.int32),
            expected.detach().cpu().view(torch.int32),
        )
    )


def _rng_2d_block_sizes() -> list[int]:
    if _get_backend() == "cute":
        return [32, 32]
    return [64, 64]


def _rng_heavy_2d_block_sizes() -> list[int]:
    if _get_backend() == "cute":
        return [16, 16]
    return [16, 16]


def _rng_3d_block_sizes() -> list[int]:
    if _get_backend() == "cute":
        return [4, 8, 32]
    return [8, 8, 64]


def _compile_once(
    fn: helion.Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> tuple[str, object]:
    bound = fn.bind(args)
    if kwargs:
        config = Config(
            # pyrefly: ignore [bad-argument-type]
            **kwargs
        )
    elif fn.configs:
        (config,) = fn.configs
    else:
        config = bound.config_spec.default_config()
    for key in bound.config_spec.unsupported_config_keys(config.config):
        config.config.pop(key, None)
    code = bound.to_triton_code(config)
    compiled = bound.compile_config(config)
    return code, compiled


def _compile_only(
    fn: helion.Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> object:
    bound = fn.bind(args)
    if kwargs:
        config = Config(
            # pyrefly: ignore [bad-argument-type]
            **kwargs
        )
    elif fn.configs:
        (config,) = fn.configs
    else:
        config = bound.config_spec.default_config()
    for key in bound.config_spec.unsupported_config_keys(config.config):
        config.config.pop(key, None)
    if is_ref_mode_enabled(bound.kernel.settings):
        bound._config = config
        return bound
    return bound.compile_config(config)


if triton is not None and tl is not None:

    @triton.jit
    def _triton_rand_from_offsets(
        seed, offsets_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        offsets = tl.load(offsets_ptr + idx, mask=mask, other=0)
        values = tl.rand(seed, offsets)
        tl.store(out_ptr + idx, values, mask=mask)

    def _triton_rand_reference(seed: int, offsets: torch.Tensor) -> torch.Tensor:
        out = torch.empty(offsets.numel(), device=offsets.device, dtype=torch.float32)
        _triton_rand_from_offsets[(triton.cdiv(offsets.numel(), 256),)](
            seed,
            offsets,
            out,
            offsets.numel(),
            BLOCK=256,
        )
        return out

    @triton.jit
    def _triton_randn_from_offsets(
        seed, offsets_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        offsets = tl.load(offsets_ptr + idx, mask=mask, other=0)
        values = tl.randn(seed, offsets)
        tl.store(out_ptr + idx, values, mask=mask)

    def _triton_randn_reference(seed: int, offsets: torch.Tensor) -> torch.Tensor:
        out = torch.empty(offsets.numel(), device=offsets.device, dtype=torch.float32)
        _triton_randn_from_offsets[(triton.cdiv(offsets.numel(), 256),)](
            seed,
            offsets,
            out,
            offsets.numel(),
            BLOCK=256,
        )
        return out

else:

    def _triton_rand_reference(seed: int, offsets: torch.Tensor) -> torch.Tensor:
        raise unittest.SkipTest("requires Triton")

    def _triton_randn_reference(seed: int, offsets: torch.Tensor) -> torch.Tensor:
        raise unittest.SkipTest("requires Triton")


def _nested_broadcast_rand_expected(
    shape: tuple[int, int],
    block_sizes: tuple[int, int],
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    m, n = shape
    block_m, block_n = block_sizes
    expected = torch.empty((m, n), device=device, dtype=dtype)
    for m_begin in range(0, m, block_m):
        tile_m = min(block_m, m - m_begin)
        values = philox_rand_ref(
            seed,
            torch.arange(m_begin, m_begin + tile_m, device=device, dtype=torch.int64),
        ).to(dtype)
        for n_begin in range(0, n, block_n):
            tile_n = min(block_n, n - n_begin)
            expected[m_begin : m_begin + tile_m, n_begin : n_begin + tile_n] = values[
                :, None
            ].expand(tile_m, tile_n)
    return expected


def _nested_broadcast_rand_expected_3d(
    shape: tuple[int, int, int],
    block_sizes: tuple[int, int, int],
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    b, m, n = shape
    block_b, block_m, block_n = block_sizes
    expected = torch.empty((b, m, n), device=device, dtype=dtype)
    for b_begin in range(0, b, block_b):
        tile_b = min(block_b, b - b_begin)
        for m_begin in range(0, m, block_m):
            tile_m = min(block_m, m - m_begin)
            base_offsets = (
                (torch.arange(b_begin, b_begin + tile_b, device=device)[:, None] * m)
                + torch.arange(m_begin, m_begin + tile_m, device=device)[None, :]
            ).to(torch.int64)
            for n_begin in range(0, n, block_n):
                tile_n = min(block_n, n - n_begin)
                values = philox_rand_ref(
                    seed,
                    base_offsets.reshape(-1),
                ).reshape(tile_b, tile_m)
                expected[
                    b_begin : b_begin + tile_b,
                    m_begin : m_begin + tile_m,
                    n_begin : n_begin + tile_n,
                ] = values[:, :, None].expand(tile_b, tile_m, tile_n).to(dtype)
    return expected


@onlyBackends(["triton", "pallas", "cute"])
class TestRNG(RefEagerTestBase, TestCase):
    @xfailIfPallas("implicit rand still hits TPU deferred buffer materialization")
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_rand(self):
        """Test RNG seeding behavior, reproducibility, output range, and distribution."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rand_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output

        # Test with different tensor sizes for different aspects
        x_small = torch.ones(128, 128, device=DEVICE)  # For distribution tests
        x_large = torch.ones(128, 128, device=DEVICE)  # For seeding tests
        block_sizes = _rng_2d_block_sizes()
        _code1, compiled = _compile_once(
            rand_kernel_tiled_2d, (x_large,), block_sizes=block_sizes
        )

        # Test 1: Different seeds produce different outputs
        torch.manual_seed(42)
        output1 = compiled(x_large)

        torch.manual_seed(123)
        output2 = compiled(x_large)

        self.assertFalse(
            torch.allclose(output1, output2),
            "Different seeds should produce different outputs",
        )

        # Test 2: Same seed produces identical outputs (reproducibility)
        torch.manual_seed(42)
        output3 = compiled(x_large)

        torch.testing.assert_close(
            output1, output3, msg="Same seed should produce identical outputs"
        )

        # Test 3: RNG state advances between calls
        torch.manual_seed(42)
        output4 = compiled(x_large)
        # No manual_seed here - RNG state should advance
        output5 = compiled(x_large)

        self.assertFalse(
            torch.allclose(output4, output5),
            "Sequential calls should produce different outputs (RNG state advanced)",
        )

        # Test 4: Output range and distribution properties
        torch.manual_seed(42)
        output6 = compiled(x_small)

        # All values should be in [0, 1) range
        self.assertTrue(torch.all(output6 >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output6 < 1.0), "All values should be < 1")

        # Check distribution properties
        mean_val = output6.mean().item()
        self.assertTrue(
            0.4 < mean_val < 0.6,
            f"Mean {mean_val:.3f} should be around 0.5 for uniform distribution",
        )

        # Check spread of values
        min_val = output6.min().item()
        max_val = output6.max().item()
        self.assertTrue(
            min_val < 0.2, f"Min value {min_val:.3f} should be < 0.2 for good spread"
        )
        self.assertTrue(
            max_val > 0.8, f"Max value {max_val:.3f} should be > 0.8 for good spread"
        )

    @xfailIfPallas("3D aten rand has low uniqueness with fold_in offset collisions")
    def test_rand_3d_tensor(self):
        """Test 3D RNG with tiled operations."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rand_kernel_3d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = torch.rand_like(
                    x[tile_b, tile_m, tile_n]
                )
            return output

        x = torch.ones(16, 32, 64, device=DEVICE)  # 3D tensor
        torch.manual_seed(77)
        block_sizes = _rng_3d_block_sizes()
        _code, output = code_and_output(rand_kernel_3d, (x,), block_sizes=block_sizes)

        # All values should be in [0, 1) range
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output < 1.0))

        # Check uniqueness - 3D should generate different values for each element
        unique_values = output.cpu().unique().numel()
        total_values = output.numel()

        # With a good RNG, we should have mostly unique values
        uniqueness_ratio = unique_values / total_values
        print(
            f"3D Unique values: {unique_values}, Total: {total_values}, Percentage: {uniqueness_ratio * 100:.2f}%"
        )

        # Expect at least 95% unique values for good 3D RNG
        self.assertGreater(uniqueness_ratio, 0.95)

        # Check distribution across dimensions
        # Mean should be around 0.5 for each 2D slice
        for b_idx in range(x.shape[0]):
            slice_mean = output[b_idx].mean().item()
            self.assertTrue(
                0.35 < slice_mean < 0.65,
                f"Slice {b_idx} mean {slice_mean} is not well distributed",
            )

        # Verify different seeds produce different results
        torch.manual_seed(88)
        _code2, output2 = code_and_output(rand_kernel_3d, (x,), block_sizes=block_sizes)
        self.assertFalse(torch.allclose(output, output2))

    @xfailIfPallas(
        "mixed explicit and implicit RNG tiles mis-handle partial tile shapes"
    )
    def test_explicit_seeded_rng_does_not_shift_implicit_seed_slots(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def implicit_only_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def explicit_then_implicit_kernel(
            x: torch.Tensor,
            explicit_seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            implicit = torch.zeros_like(x)
            explicit = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                explicit[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=explicit_seed)
                implicit[tile_m, tile_n] = torch.rand_like(tile)
            return implicit, explicit

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def implicit_then_explicit_kernel(
            x: torch.Tensor,
            explicit_seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            implicit = torch.zeros_like(x)
            explicit = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                implicit[tile_m, tile_n] = torch.rand_like(tile)
                explicit[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=explicit_seed)
            return implicit, explicit

        x = torch.ones((9, 11), device=DEVICE, dtype=torch.float32)
        explicit_seed = 0x1234_5678
        block_sizes = [4, 8]

        torch.manual_seed(2026)
        implicit_only = _compile_only(
            implicit_only_kernel,
            (x,),
            block_sizes=block_sizes,
        )(x)
        torch.manual_seed(2026)
        explicit_then_implicit, _explicit0 = _compile_only(
            explicit_then_implicit_kernel,
            (x, explicit_seed),
            block_sizes=block_sizes,
        )(x, explicit_seed)
        torch.manual_seed(2026)
        implicit_then_explicit, _explicit1 = _compile_only(
            implicit_then_explicit_kernel,
            (x, explicit_seed),
            block_sizes=block_sizes,
        )(x, explicit_seed)

        _assert_bitwise_equal_float(self, explicit_then_implicit, implicit_only)
        _assert_bitwise_equal_float(self, implicit_then_explicit, implicit_only)

    @xfailIfPallas(
        "multiple implicit RNG outputs hit TPU deferred buffer materialization"
    )
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_multiple_rng_ops(self):
        """Test multiple RNG operations: independence, reproducibility, mixed rand/randn."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def multiple_rng_ops_kernel(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # Two independent rand operations
            rand1 = torch.zeros_like(x)
            rand2 = torch.zeros_like(x)

            normal = torch.zeros_like(x)
            randn_sum = torch.zeros_like(x)

            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                # Two independent rand operations
                rand1[tile_m, tile_n] = torch.rand_like(tile)
                rand2[tile_m, tile_n] = torch.rand_like(tile)

                # Mixed rand and randn
                normal[tile_m, tile_n] = torch.randn_like(tile)

                # Multiple randn
                randn_sum[tile_m, tile_n] = (
                    torch.randn_like(tile)
                    + torch.randn_like(tile)
                    + torch.randn_like(tile)
                )

            return rand1, rand2, normal, randn_sum

        x = torch.ones(32, 32, device=DEVICE)
        block_sizes = _rng_heavy_2d_block_sizes()
        _code1, compiled = _compile_once(
            multiple_rng_ops_kernel, (x,), block_sizes=block_sizes
        )

        # Test 1: Independence and distribution properties
        torch.manual_seed(42)
        rand1, rand2, normal, randn_sum = compiled(x)

        # Check two independent rand operations
        self.assertTrue(
            torch.all(rand1 >= 0.0) and torch.all(rand1 < 1.0),
            "First rand output should be in [0, 1)",
        )
        self.assertTrue(
            torch.all(rand2 >= 0.0) and torch.all(rand2 < 1.0),
            "Second rand output should be in [0, 1)",
        )
        self.assertFalse(
            torch.allclose(rand1, rand2),
            "Two independent RNG ops should produce different outputs",
        )
        self.assertTrue(
            0.45 < rand1.mean().item() < 0.55,
            f"First rand mean {rand1.mean().item():.3f} should be ~0.5",
        )
        self.assertTrue(
            0.45 < rand2.mean().item() < 0.55,
            f"Second rand mean {rand2.mean().item():.3f} should be ~0.5",
        )

        # Check mixed rand and randn
        self.assertTrue(
            -0.2 < normal.mean().item() < 0.2,
            f"Normal mean {normal.mean().item():.3f} should be ~0",
        )
        self.assertTrue(
            0.9 < normal.cpu().std().item() < 1.1,
            f"Normal std {normal.cpu().std().item():.3f} should be ~1",
        )
        self.assertTrue(
            torch.any(normal < 0.0), "Normal distribution should have negative values"
        )
        self.assertFalse(
            torch.allclose(rand1, normal),
            "Uniform and normal distributions should be different",
        )

        # Check sum of multiple randn
        expected_std = 3**0.5
        mean = randn_sum.mean().item()
        std = randn_sum.cpu().std().item()
        self.assertTrue(-0.2 < mean < 0.2, f"Combined mean {mean:.3f} should be ~0")
        self.assertTrue(
            expected_std * 0.9 < std < expected_std * 1.1,
            f"Combined std {std:.3f} should be ~{expected_std:.3f}",
        )

        # Test 2: Reproducibility with same seed
        torch.manual_seed(42)
        outputs_a = compiled(x)

        torch.manual_seed(42)
        outputs_b = compiled(x)

        # All outputs should be identical with same seed
        for i, (a, b) in enumerate(zip(outputs_a, outputs_b, strict=False)):
            torch.testing.assert_close(
                a, b, msg=f"Output {i} should be identical with same seed"
            )

        _assert_uses_philox(self, _code1)

    @xfailIfPallas("implicit randn still hits TPU deferred buffer materialization")
    def test_randn_different_seeds_tiled(self):
        """Test that different torch.manual_seed values produce different outputs for randn."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
            return output

        x = torch.ones(128, 128, device=DEVICE)
        block_sizes = _rng_2d_block_sizes()

        torch.manual_seed(42)
        _code1, output1 = code_and_output(
            randn_kernel_tiled_2d, (x,), block_sizes=block_sizes
        )

        torch.manual_seed(123)
        _code2, output2 = code_and_output(
            randn_kernel_tiled_2d, (x,), block_sizes=block_sizes
        )

        # Different seeds should produce different outputs
        self.assertFalse(torch.allclose(output1, output2))

    @xfailIfPallas("implicit randn still hits TPU deferred buffer materialization")
    def test_randn_normal_distribution(self):
        """Test that torch.randn_like produces normal distribution (mean≈0, std≈1)."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
            return output

        x = torch.ones(128, 128, device=DEVICE)  # 16384 samples for better statistics
        torch.manual_seed(42)
        _code, output = code_and_output(
            randn_kernel_tiled_2d, (x,), block_sizes=_rng_2d_block_sizes()
        )

        # Check mean is close to 0
        mean = output.mean().item()
        self.assertTrue(-0.1 < mean < 0.1, f"Mean {mean} is not close to 0")

        # Check std is close to 1
        std = output.cpu().std().item()
        self.assertTrue(0.95 < std < 1.05, f"Std {std} is not close to 1")

        # Check we have values outside [-1, 1] (characteristic of normal distribution)
        self.assertTrue(torch.any(output < -1.0))
        self.assertTrue(torch.any(output > 1.0))

        # Roughly 68% should be within 1 std
        within_1_std = (
            torch.logical_and(output > -1.0, output < 1.0).float().mean().item()
        )
        self.assertTrue(
            0.63 < within_1_std < 0.73, f"Values within 1 std: {within_1_std}"
        )

    @xfailIfPallas("3D implicit randn still hits TPU materialization failure")
    def test_randn_3d_tensor(self):
        """Test 3D randn with tiled operations."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_kernel_3d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = torch.randn_like(
                    x[tile_b, tile_m, tile_n]
                )
            return output

        x = torch.ones(8, 32, 64, device=DEVICE)  # 3D tensor
        torch.manual_seed(77)
        _code, output = code_and_output(
            randn_kernel_3d, (x,), block_sizes=_rng_3d_block_sizes()
        )

        # Check overall distribution
        output_cpu = output.cpu()
        mean = output_cpu.mean().item()
        std = output_cpu.std().item()
        self.assertTrue(-0.1 < mean < 0.1, f"3D mean {mean} not close to 0")
        self.assertTrue(0.95 < std < 1.05, f"3D std {std} not close to 1")

        # Check distribution across dimensions
        for b_idx in range(x.shape[0]):
            slice_mean = output_cpu[b_idx].mean().item()
            slice_std = output_cpu[b_idx].std().item()
            self.assertTrue(
                -0.3 < slice_mean < 0.3,
                f"Slice {b_idx} mean {slice_mean} is not well distributed",
            )
            self.assertTrue(
                0.85 < slice_std < 1.15,
                f"Slice {b_idx} std {slice_std} is not well distributed",
            )

    def _test_rng_with_dynamic_tile_sizes(self, rng_func, is_uniform, rng_name):
        """Common test logic for RNG operations with dynamic tile sizes."""

        # Single kernel that takes an RNG callable as a parameter
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rng_kernel(
            x: torch.Tensor,
            rng_func: Callable[[int, int, torch.dtype], torch.Tensor],
        ) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = rng_func(tile_m, tile_n, x.dtype)
            return output

        x = torch.ones(48, 48, device=DEVICE)
        torch.manual_seed(42)
        block_sizes = _rng_2d_block_sizes()
        compiled = _compile_only(rng_kernel, (x, rng_func), block_sizes=block_sizes)
        output = compiled(x, rng_func)

        # Check distribution properties based on RNG type
        if is_uniform:
            # For rand: values in [0, 1), mean ~0.5
            self.assertTrue(
                torch.all(output >= 0.0), f"{rng_name}: All values should be >= 0"
            )
            self.assertTrue(
                torch.all(output < 1.0), f"{rng_name}: All values should be < 1"
            )
            mean_val = output.mean().item()
            self.assertTrue(
                0.4 < mean_val < 0.6,
                f"{rng_name}: Mean {mean_val:.3f} should be ~0.5",
            )
        else:
            # For randn: mean ~0, std ~1
            mean_val = output.mean().item()
            std_val = output.cpu().std().item()
            self.assertTrue(
                -0.15 < mean_val < 0.15, f"{rng_name}: Mean {mean_val:.3f} should be ~0"
            )
            self.assertTrue(
                0.9 < std_val < 1.1, f"{rng_name}: Std {std_val:.3f} should be ~1"
            )

        # Test reproducibility with same seed
        torch.manual_seed(42)
        output2 = compiled(x, rng_func)
        torch.testing.assert_close(
            output,
            output2,
            msg=f"{rng_name}: Same seed should produce identical outputs",
        )

        # Test that different seeds produce different outputs
        torch.manual_seed(99)
        output3 = compiled(x, rng_func)
        self.assertFalse(
            torch.allclose(output, output3),
            f"{rng_name}: Different seeds should produce different outputs",
        )

    @xfailIfPallas(
        "dynamic-shape implicit rand hits TPU deferred buffer materialization"
    )
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_rand_with_dynamic_tile_sizes(self):
        """Test torch.rand with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.rand(
                (tile_m, tile_n), dtype=dtype, device=DEVICE
            ),
            is_uniform=True,
            rng_name="rand",
        )

    @xfailIfPallas(
        "dynamic-shape implicit rand_like hits TPU deferred buffer materialization"
    )
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_rand_like_with_dynamic_tile_sizes(self):
        """Test torch.rand_like with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.rand_like(
                torch.ones((tile_m, tile_n), dtype=dtype, device=DEVICE)
            ),
            is_uniform=True,
            rng_name="rand_like",
        )

    @xfailIfPallas(
        "dynamic-shape implicit randn hits TPU deferred buffer materialization"
    )
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_randn_with_dynamic_tile_sizes(self):
        """Test torch.randn with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.randn(
                (tile_m, tile_n), dtype=dtype, device=DEVICE
            ),
            is_uniform=False,
            rng_name="randn",
        )

    @xfailIfPallas(
        "dynamic-shape implicit randn_like hits TPU deferred buffer materialization"
    )
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_randn_like_with_dynamic_tile_sizes(self):
        """Test torch.randn_like with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.randn_like(
                torch.ones((tile_m, tile_n), dtype=dtype, device=DEVICE)
            ),
            is_uniform=False,
            rng_name="randn_like",
        )

    @skipIfRefEager(
        "compiled implicit RNG validation is not applicable in ref eager mode"
    )
    def test_implicit_rand_rejects_non_floating_dtype(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def bad_rand_dtype(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x, dtype=torch.float32)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.rand(
                    (tile_m, tile_n), dtype=torch.int32, device=x.device
                ).to(torch.float32)
            return out

        x = torch.ones((13, 29), device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(
            Exception, "implicit RNG only supports floating-point dtypes"
        ):
            code_and_output(bad_rand_dtype, (x,), block_sizes=[8, 16])

    @skipIfRefEager(
        "compiled implicit RNG validation is not applicable in ref eager mode"
    )
    def test_implicit_rand_like_rejects_non_floating_dtype(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def bad_rand_like_dtype(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x, dtype=torch.float32)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n]).to(
                    torch.float32
                )
            return out

        x = torch.ones((13, 29), device=DEVICE, dtype=torch.int32)
        with self.assertRaisesRegex(
            Exception, "implicit RNG only supports floating-point dtypes"
        ):
            code_and_output(bad_rand_like_dtype, (x,), block_sizes=[8, 16])

    @skipIfRefEager(
        "compiled implicit RNG validation is not applicable in ref eager mode"
    )
    def test_raw_aten_rand_rejects_non_floating_dtype(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def bad_aten_rand_dtype(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x, dtype=torch.float32)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                out[tile_m, tile_n] = torch.ops.aten.rand.default(
                    [tile.shape[0], tile.shape[1]],
                    dtype=torch.int32,
                    device=x.device,
                ).to(torch.float32)
            return out

        x = torch.ones((13, 29), device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(
            Exception, "implicit RNG only supports floating-point dtypes"
        ):
            code_and_output(bad_aten_rand_dtype, (x,), block_sizes=[8, 16])

    @skipIfRefEager(
        "compiled implicit RNG validation is not applicable in ref eager mode"
    )
    def test_implicit_rng_rejects_wrong_device_index(self):
        if DEVICE.type == "cuda":
            if torch.cuda.device_count() < 2:
                self.skipTest("requires multiple CUDA devices")
        elif DEVICE.type == "xpu":
            if torch.xpu.device_count() < 2:
                self.skipTest("requires multiple XPU devices")
        else:
            self.skipTest("requires indexed accelerator devices")

        x = torch.ones((13, 29), device=DEVICE, dtype=torch.float32)
        requested_device = torch.device(
            x.device.type,
            1 if x.device.index == 0 else 0,
        )

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def wrong_device_rand(
            x: torch.Tensor,
            requested_device=requested_device,
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.rand(
                    (tile_m, tile_n), device=requested_device
                )
            return out

        with self.assertRaisesRegex(Exception, "expected .* got .*"):
            code_and_output(wrong_device_rand, (x,), block_sizes=[8, 16])

    @skipIfXPU("RNG with specialized dimensions not supported on XPU")
    @xfailIfPallas("specialized-dimension rand_like hits TPU MLIR refinement mismatch")
    @xfailIfCute(
        "CuTe matmul plus specialized-dimension rand_like still returns unstable NaNs"
    )
    @skipIfRefEager("compiled codegen inspection is not applicable in ref eager mode")
    @skipIfRocm("ROCm Triton worker crashes on specialized-dimension rand_like")
    def test_rand_like_with_specialized_dimension(self):
        """Test torch.rand_like with specialized (constant) dimensions."""

        @helion.kernel(config=helion.Config(block_sizes=[64, 128]))
        def matmul_with_rand(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            # Specialize n to make it a constant dimension
            n = hl.specialize(n)

            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    mm = torch.matmul(x[tile_m, tile_k], y[tile_k, :])
                    acc = acc + mm
                # This rand_like has shape [tile_m, n] where:
                # - tile_m is a block dimension
                # - n is a specialized (constant) dimension
                noise = torch.rand_like(acc, dtype=torch.float32)
                acc = acc + noise * 0.01  # Small noise
                out[tile_m, :] = acc.to(out.dtype)
            return out

        m, k, n = 256, 512, 64
        x = torch.randn(m, k, device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn(k, n, device=DEVICE, dtype=HALF_DTYPE)

        torch.manual_seed(42)
        code, result = code_and_output(matmul_with_rand, (x, y))

        # Verify the output shape
        self.assertEqual(result.shape, (m, n))

        # Verify reproducibility
        torch.manual_seed(42)
        _code2, result2 = code_and_output(matmul_with_rand, (x, y))
        torch.testing.assert_close(result, result2)

        # Verify different seeds produce different results
        torch.manual_seed(123)
        _code3, result3 = code_and_output(matmul_with_rand, (x, y))
        self.assertFalse(torch.allclose(result, result3))
        _assert_uses_philox(self, code)

    @xfailIfCute(
        "CuTe still rejects nested RNG tiles that require a fourth thread axis"
    )
    @xfailIfPallas("nested rand_like tiles hit TPU MLIR refinement mismatch")
    @skipIfRefEager("compiled codegen inspection is not applicable in ref eager mode")
    def test_rand_like_nested_tiles_issue_1208(self):
        """Test torch.rand_like with nested tiles (regression test for issue #1208).

        This test reproduces the bug where torch.rand_like() failed with nested tiles
        because the RNG codegen incorrectly used dimension indices instead of block_ids
        when constructing index variable names.
        """

        @helion.kernel(
            autotune_effort="none",
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def nested_tiles_rand(q: torch.Tensor) -> torch.Tensor:
            B, T, H = q.shape
            out = torch.empty((B, T, H), device=q.device, dtype=q.dtype)

            for tile_b, tile_q in hl.tile([B, T]):
                qs = q[tile_b, tile_q, :]
                for tile_k in hl.tile(T):
                    ks = q[tile_b, tile_k, :]
                    # logits has shape [tile_b, tile_q, tile_k]
                    # The third dimension uses indices_3 (from the inner loop)
                    # not indices_2 (from H dimension)
                    logits = qs @ ks.transpose(-1, -2)

                    # This used to fail because rand_like incorrectly used
                    # indices_2 (size H=32) instead of indices_3 (size tile_k=16)
                    rand = torch.rand_like(logits)

                    mask = ((logits + rand) > 0).float()
                    out[tile_b, tile_q, :] = torch.matmul(mask, q[tile_b, tile_q, :])

            return out

        q = torch.randn(2, 16, 32, device=DEVICE, dtype=torch.float32)
        torch.manual_seed(42)
        code, result = code_and_output(nested_tiles_rand, (q,), block_sizes=[2, 16, 16])

        # Verify output shape
        self.assertEqual(result.shape, (2, 16, 32))

        # Verify reproducibility
        torch.manual_seed(42)
        _code2, result2 = code_and_output(
            nested_tiles_rand, (q,), block_sizes=[2, 16, 16]
        )
        torch.testing.assert_close(result, result2)

        # Verify different seeds produce different results
        torch.manual_seed(123)
        _code3, result3 = code_and_output(
            nested_tiles_rand, (q,), block_sizes=[2, 16, 16]
        )
        self.assertFalse(torch.allclose(result, result3))
        _assert_uses_philox(self, code)


@onlyBackends(["triton", "cute"])
@skipIfRefEager("compiled RNG parity checks are not applicable in ref eager mode")
class TestRNGBitParity(TestCase):
    def test_run_ref_matches_compiled_for_multiple_implicit_rng_callsites(self):
        def implicit_rng_impl(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((2, *x.shape), device=x.device, dtype=x.dtype)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                out[0, tile_m, tile_n] = torch.rand(
                    (tile_m, tile_n), device=x.device, dtype=x.dtype
                )
                out[1, tile_m, tile_n] = torch.randn_like(tile)
            return out

        compiled_kernel = helion.kernel(
            static_shapes=True,
            autotune_effort="none",
        )(implicit_rng_impl)
        ref_kernel = helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            ref_mode=helion.RefMode.EAGER,
        )(implicit_rng_impl)

        x = torch.ones(17, 19, device=DEVICE)
        compiled_bound = compiled_kernel.bind((x,))
        ref_bound = ref_kernel.bind((x,))
        config = Config(block_sizes=[8, 16])

        torch.manual_seed(2026)
        compiled = compiled_bound.compile_config(config)(x)

        torch.manual_seed(2026)
        ref = ref_bound.run_ref(x)

        _assert_bitwise_equal_float(self, compiled[0], ref[0])
        torch.testing.assert_close(compiled[1], ref[1], rtol=1e-6, atol=1e-6)

    def test_rand_nested_tl_range_loops_match_philox(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def nested_rand_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b in hl.tile(b):
                for tile_m in hl.tile(m):
                    for tile_n in hl.tile(n):
                        noise = torch.rand(
                            (tile_b, tile_m), device=x.device, dtype=x.dtype
                        )
                        out[tile_b, tile_m, tile_n] = noise[:, :, None].expand(
                            tile_b, tile_m, tile_n
                        )
            return out

        x = torch.ones(3, 7, 9, device=DEVICE)
        torch.manual_seed(901)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(901)
        code, out = code_and_output(nested_rand_kernel, (x,), block_sizes=[2, 4, 8])
        expected = _nested_broadcast_rand_expected_3d(
            (3, 7, 9),
            (2, 4, 8),
            seed,
            device=DEVICE,
            dtype=x.dtype,
        )
        _assert_bitwise_equal_float(self, out, expected)
        if _get_backend() == "triton":
            self.assertTrue("tl.range" in code or "tl.static_range" in code)

    def test_rand_multi_axis_loop_offsets_match_philox(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def tiled_rand_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b in hl.tile(b):
                for tile_m, tile_n in hl.tile([m, n]):
                    noise = torch.rand((tile_b, tile_m), device=x.device, dtype=x.dtype)
                    out[tile_b, tile_m, tile_n] = noise[:, :, None].expand(
                        tile_b, tile_m, tile_n
                    )
            return out

        x = torch.ones(3, 7, 9, device=DEVICE)
        torch.manual_seed(902)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(902)
        _code, out = code_and_output(tiled_rand_kernel, (x,), block_sizes=[2, 4, 8])
        expected = _nested_broadcast_rand_expected_3d(
            (3, 7, 9),
            (2, 4, 8),
            seed,
            device=DEVICE,
            dtype=x.dtype,
        )
        _assert_bitwise_equal_float(self, out, expected)

    def test_rand_like_nested_loops_matches_philox(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def nested_rand_like_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b in hl.tile(b):
                for tile_m in hl.tile(m):
                    out[tile_b, tile_m, :] = torch.rand_like(x[tile_b, tile_m, :])
            return out

        x = torch.ones(7, 9, 11, device=DEVICE)
        torch.manual_seed(456)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(456)
        _code, out = code_and_output(nested_rand_like_kernel, (x,), block_sizes=[4, 4])
        expected = philox_rand_ref(
            seed,
            torch.arange(out.numel(), device=DEVICE, dtype=torch.int64),
        ).reshape_as(out)
        _assert_bitwise_equal_float(self, out, expected)

    def test_rand_with_specialized_dim_matches_philox(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def specialized_rand_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            n = hl.specialize(n)
            out = torch.zeros_like(x)
            for tile_m in hl.tile(m):
                out[tile_m, :] = torch.rand((tile_m, n), device=x.device, dtype=x.dtype)
            return out

        x = torch.ones(13, 16, device=DEVICE)
        torch.manual_seed(654)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(654)
        _code, out = code_and_output(specialized_rand_kernel, (x,), block_sizes=[8])
        expected = philox_rand_ref(
            seed,
            torch.arange(out.numel(), device=DEVICE, dtype=torch.int64),
        ).reshape_as(out)
        _assert_bitwise_equal_float(self, out, expected)

    def test_rand_with_literal_non_power_of_two_dim_matches_philox(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def literal_dim_rand_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            (m, _) = x.shape
            for tile_m in hl.tile(m):
                out[tile_m, :] = torch.rand(
                    (tile_m, 3),
                    device=x.device,
                    dtype=x.dtype,
                ) + torch.full((tile_m, 3), 1.0, device=x.device, dtype=x.dtype)
            return out

        x = torch.ones(13, 3, device=DEVICE)
        torch.manual_seed(314)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(314)
        _code, out = code_and_output(literal_dim_rand_kernel, (x,), block_sizes=[8])
        expected = (
            philox_rand_ref(
                seed,
                torch.arange(out.numel(), device=DEVICE, dtype=torch.int64),
            ).reshape_as(out)
            + 1.0
        )
        torch.testing.assert_close(out, expected, rtol=0.0, atol=2e-7)

    def test_rand_with_literal_non_power_of_two_dim_no_intermediate_matches_philox(
        self,
    ):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def literal_dim_rand_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            (m, _) = x.shape
            for tile_m in hl.tile(m):
                out[tile_m, :] = torch.rand(
                    (tile_m, 3),
                    device=x.device,
                    dtype=x.dtype,
                )
            return out

        x = torch.ones(13, 3, device=DEVICE)
        torch.manual_seed(2718)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(2718)
        _code, out = code_and_output(literal_dim_rand_kernel, (x,), block_sizes=[8])
        expected = philox_rand_ref(
            seed,
            torch.arange(out.numel(), device=DEVICE, dtype=torch.int64),
        ).reshape_as(out)
        _assert_bitwise_equal_float(self, out, expected)

    def test_rand_like_matches_triton_philox(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rand_like_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return out

        x = torch.ones(17, 19, device=DEVICE)
        torch.manual_seed(123)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(123)
        _code, out = code_and_output(rand_like_kernel, (x,), block_sizes=[8, 16])
        offsets = torch.arange(out.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_rand_reference(seed, offsets).reshape_as(out)
        _assert_bitwise_equal_float(self, out, expected)

    def test_rand_like_multiple_calls_match_triton_reference(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rand_like_twice_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((2, *x.shape), device=x.device, dtype=x.dtype)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                out[0, tile_m, tile_n] = torch.rand_like(tile)
                out[1, tile_m, tile_n] = torch.rand_like(tile)
            return out

        x = torch.ones(17, 19, device=DEVICE)
        torch.manual_seed(987)
        seeds = inductor_prims.seeds(2, torch.accelerator.current_accelerator())
        seed0 = int(seeds[0].item())
        seed1 = int(seeds[1].item())
        torch.manual_seed(987)
        _code, out = code_and_output(rand_like_twice_kernel, (x,), block_sizes=[8, 16])
        offsets = torch.arange(x.numel(), device=DEVICE, dtype=torch.int64)
        expected0 = _triton_rand_reference(seed0, offsets).reshape_as(x)
        expected1 = _triton_rand_reference(seed1, offsets).reshape_as(x)
        _assert_bitwise_equal_float(self, out[0], expected0)
        _assert_bitwise_equal_float(self, out[1], expected1)
        self.assertFalse(torch.equal(out[0].cpu(), out[1].cpu()))

    def test_randn_like_matches_triton_philox(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_like_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
            return out

        x = torch.ones(13, 29, device=DEVICE)
        torch.manual_seed(321)
        seed = int(
            inductor_prims.seeds(1, torch.accelerator.current_accelerator())[0].item()
        )
        torch.manual_seed(321)
        _code, out = code_and_output(randn_like_kernel, (x,), block_sizes=[8, 16])
        offsets = torch.arange(out.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_randn_reference(seed, offsets).reshape_as(out)
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


class TestPallasRNGRegression(TestCase):
    def test_pallas_seed_buffer_expr_preserves_high_32_bits(self):
        seeds64 = torch.tensor(
            [0x0000_0001_0000_0005, 0x0000_0002_0000_0005],
            dtype=torch.int64,
        )
        backend = PallasBackend()
        expr = backend.rng_seed_buffer_expr(len(seeds64))

        class FakeInductorPrims:
            @staticmethod
            def seeds(count: int, device: torch.device) -> torch.Tensor:
                self.assertEqual(count, len(seeds64))
                self.assertEqual(device, torch.device("cpu"))
                return seeds64

        with patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("cpu"),
        ):
            seed_buffer = eval(
                expr,
                {"inductor_prims": FakeInductorPrims, "torch": torch},
            )

        self.assertEqual(seed_buffer.dtype, torch.int64)
        self.assertTrue(torch.equal(seed_buffer.cpu(), seeds64))


@onlyBackends(["triton", "cute"])
@skipUnlessCuteAvailable("requires CUTLASS CuTe Python DSL")
@skipIfRefEager("compiled backend parity checks are not applicable in ref eager mode")
class TestRNGBackendParity(TestCase):
    def test_triton_and_cute_match_explicit_seeded_rand(self):
        @helion.kernel(backend="triton", static_shapes=False, autotune_effort="none")
        def rand_kernel_triton(
            x: torch.Tensor, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out0 = torch.zeros_like(x)
            out1 = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out0[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            for tile_m, tile_n in hl.tile([m, n]):
                out1[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return out0, out1

        @helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
        def rand_kernel_cute(
            x: torch.Tensor, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out0 = torch.zeros_like(x)
            out1 = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out0[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            for tile_m, tile_n in hl.tile([m, n]):
                out1[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return out0, out1

        x = torch.empty((11, 13), device=DEVICE, dtype=torch.float32)
        seed = 4096
        out_t = _compile_only(rand_kernel_triton, (x, seed), block_sizes=[4, 8, 4, 8])(
            x, seed
        )
        out_c = _compile_only(rand_kernel_cute, (x, seed), block_sizes=[4, 8, 4, 8])(
            x, seed
        )
        _assert_bitwise_equal_float(self, out_t[0], out_c[0])
        _assert_bitwise_equal_float(self, out_t[1], out_c[1])

    def test_triton_and_cute_match_explicit_seeded_randint(self):
        @helion.kernel(backend="triton", static_shapes=False, autotune_effort="none")
        def randint_kernel_triton(
            x: torch.Tensor, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out0 = torch.zeros_like(x)
            out1 = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out0[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-3, high=29, seed=seed
                )
            for tile_m, tile_n in hl.tile([m, n]):
                out1[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-3, high=29, seed=seed
                )
            return out0, out1

        @helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
        def randint_kernel_cute(
            x: torch.Tensor, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out0 = torch.zeros_like(x)
            out1 = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out0[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-3, high=29, seed=seed
                )
            for tile_m, tile_n in hl.tile([m, n]):
                out1[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-3, high=29, seed=seed
                )
            return out0, out1

        x = torch.empty((9, 15), device=DEVICE, dtype=torch.int32)
        seed = 5150
        out_t = _compile_only(
            randint_kernel_triton, (x, seed), block_sizes=[4, 8, 4, 8]
        )(x, seed)
        out_c = _compile_only(randint_kernel_cute, (x, seed), block_sizes=[4, 8, 4, 8])(
            x, seed
        )
        self.assertTrue(torch.equal(out_t[0].cpu(), out_c[0].cpu()))
        self.assertTrue(torch.equal(out_t[1].cpu(), out_c[1].cpu()))

    def test_triton_and_cute_match_raw_aten_rand_and_randn(self):
        @helion.kernel(backend="triton", static_shapes=False, autotune_effort="none")
        def aten_rng_kernel_triton(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            uniform = torch.zeros_like(x)
            normal = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                uniform[tile_m, tile_n] = torch.ops.aten.rand.default(
                    [tile.shape[0], tile.shape[1]], dtype=x.dtype, device=x.device
                )
                normal[tile_m, tile_n] = torch.ops.aten.randn.default(
                    [tile.shape[0], tile.shape[1]], dtype=x.dtype, device=x.device
                )
            return uniform, normal

        @helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
        def aten_rng_kernel_cute(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            uniform = torch.zeros_like(x)
            normal = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                tile = x[tile_m, tile_n]
                uniform[tile_m, tile_n] = torch.ops.aten.rand.default(
                    [tile.shape[0], tile.shape[1]], dtype=x.dtype, device=x.device
                )
                normal[tile_m, tile_n] = torch.ops.aten.randn.default(
                    [tile.shape[0], tile.shape[1]], dtype=x.dtype, device=x.device
                )
            return uniform, normal

        x = torch.empty((13, 29), device=DEVICE, dtype=torch.float32)
        torch.manual_seed(111)
        uniform_t, normal_t = _compile_only(
            aten_rng_kernel_triton, (x,), block_sizes=[4, 8]
        )(x)
        torch.manual_seed(111)
        uniform_c, normal_c = _compile_only(
            aten_rng_kernel_cute, (x,), block_sizes=[4, 8]
        )(x)
        _assert_bitwise_equal_float(self, uniform_t, uniform_c)
        _assert_bitwise_equal_float(self, normal_t, normal_c)


if __name__ == "__main__":
    unittest.main()
