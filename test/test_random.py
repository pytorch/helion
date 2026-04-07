from __future__ import annotations

import importlib
import os
import unittest

import torch

import helion
from helion._compiler.rng_utils import philox_rand_ref
from helion._compiler.rng_utils import philox_randint_ref
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfMTIA
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
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


def _rng_3d_block_sizes() -> list[int]:
    if _get_backend() == "cute":
        return [4, 8, 32]
    return [8, 8, 64]


def _rng_determinism_block_sizes() -> list[list[int]]:
    if _get_backend() == "cute":
        return [[8, 8], [16, 16], [32, 32]]
    return [[8, 8], [16, 16], [32, 32]]


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


def _helper_seeded_rand(tile_m, tile_n, seed: int) -> torch.Tensor:
    return hl.rand([tile_m, tile_n], seed=seed)


def _helper_seeded_randint(tile_m, tile_n, seed: int) -> torch.Tensor:
    return hl.randint([tile_m, tile_n], low=-5, high=17, seed=seed)


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

    @triton.jit
    def _triton_randint_from_offsets(
        seed,
        offsets_ptr,
        out_ptr,
        low,
        high,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        offsets = tl.load(offsets_ptr + idx, mask=mask, other=0)
        values = low + tl.abs(tl.randint(seed, offsets).to(tl.int32)) % (high - low)
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

    def _triton_randint_reference(
        seed: int,
        offsets: torch.Tensor,
        low: int,
        high: int,
    ) -> torch.Tensor:
        out = torch.empty(offsets.numel(), device=offsets.device, dtype=torch.int32)
        _triton_randint_from_offsets[(triton.cdiv(offsets.numel(), 256),)](
            seed,
            offsets,
            out,
            low,
            high,
            offsets.numel(),
            BLOCK=256,
        )
        return out

else:

    def _triton_rand_reference(seed: int, offsets: torch.Tensor) -> torch.Tensor:
        raise unittest.SkipTest("requires Triton")

    def _triton_randint_reference(
        seed: int,
        offsets: torch.Tensor,
        low: int,
        high: int,
    ) -> torch.Tensor:
        raise unittest.SkipTest("requires Triton")


def _hl_rand_outer_loop_expected(
    shape: tuple[int, int],
    block_sizes: tuple[int, int],
    seed: int,
) -> torch.Tensor:
    m, n = shape
    block_m, block_n = block_sizes
    expected = torch.empty(shape, device=DEVICE, dtype=torch.float32)
    for m_begin in range(0, m, block_m):
        tile_m = min(block_m, m - m_begin)
        row_offsets = torch.arange(
            m_begin, m_begin + tile_m, device=DEVICE, dtype=torch.int64
        )
        values = _triton_rand_reference(seed, row_offsets).reshape(tile_m, 1)
        for n_begin in range(0, n, block_n):
            tile_n = min(block_n, n - n_begin)
            expected[m_begin : m_begin + tile_m, n_begin : n_begin + tile_n] = values
    return expected


def _hl_randint_outer_loop_expected(
    shape: tuple[int, int],
    block_sizes: tuple[int, int],
    seed: int,
    low: int,
    high: int,
) -> torch.Tensor:
    m, n = shape
    block_m, block_n = block_sizes
    expected = torch.empty(shape, device=DEVICE, dtype=torch.int32)
    for m_begin in range(0, m, block_m):
        tile_m = min(block_m, m - m_begin)
        row_offsets = torch.arange(
            m_begin, m_begin + tile_m, device=DEVICE, dtype=torch.int64
        )
        values = _triton_randint_reference(
            seed,
            row_offsets,
            low,
            high,
        ).reshape(tile_m, 1)
        for n_begin in range(0, n, block_n):
            tile_n = min(block_n, n - n_begin)
            expected[m_begin : m_begin + tile_m, n_begin : n_begin + tile_n] = values
    return expected


@onlyBackends(["triton", "pallas", "cute"])
class TestRandom(RefEagerTestBase, TestCase):
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_rand_1d(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_kernel_tiled_1d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            (m,) = x.shape
            for tile_m in hl.tile(m):
                output[tile_m] = hl.rand([tile_m], seed=seed)
            return output

        x_small = torch.ones(1024, device=DEVICE)
        code3, compiled = _compile_once(
            rand_kernel_tiled_1d, (x_small, 42), block_sizes=[1024]
        )
        output = compiled(x_small, 42)
        output2 = compiled(x_small, 1337)

        self.assertFalse(
            torch.allclose(output, output2),
            "Different seeds should produce different outputs",
        )

        output3 = compiled(x_small, 42)
        self.assertTrue(
            torch.allclose(output, output3),
            "Same seed should produce identical outputs",
        )
        _assert_uses_philox(self, code3)

        # Check that all values are in [0, 1) range
        self.assertTrue(torch.all(output >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 1.0), "All values should be < 1")

    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_rand_2d(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_kernel_tiled_2d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return output

        x_small = torch.ones(1024, 1024, device=DEVICE)
        block_sizes = _rng_2d_block_sizes()
        code3, compiled = _compile_once(
            rand_kernel_tiled_2d, (x_small, 42), block_sizes=block_sizes
        )
        output = compiled(x_small, 42)
        output2 = compiled(x_small, 1337)

        self.assertFalse(
            torch.allclose(output, output2),
            "Different seeds should produce different outputs",
        )

        output3 = compiled(x_small, 42)
        self.assertTrue(
            torch.allclose(output, output3),
            "Same seed should produce identical outputs",
        )
        _assert_uses_philox(self, code3)

        self.assertTrue(torch.all(output >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 1.0), "All values should be < 1")

    @xfailIfPallas("3D hl.rand lowering hits TPU Mosaic unsupported shape cast")
    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_rand_3d(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_kernel_tiled_3d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = hl.rand(
                    [tile_b, tile_m, tile_n], seed=seed
                )
            return output

        x_small = torch.ones(16, 32, 64, device=DEVICE)
        block_sizes = _rng_3d_block_sizes()
        code3, compiled = _compile_once(
            rand_kernel_tiled_3d, (x_small, 42), block_sizes=block_sizes
        )
        output = compiled(x_small, 42)
        output2 = compiled(x_small, 1337)

        self.assertFalse(
            torch.allclose(output, output2),
            "Different seeds should produce different outputs",
        )

        output3 = compiled(x_small, 42)
        self.assertTrue(
            torch.allclose(output, output3),
            "Same seed should produce identical outputs",
        )
        _assert_uses_philox(self, code3)

        self.assertTrue(torch.all(output >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 1.0), "All values should be < 1")

        # Check distribution properties
        mean_val = output.mean().item()
        self.assertTrue(
            0.4 < mean_val < 0.6,
            f"Mean {mean_val:.3f} should be around 0.5 for uniform distribution",
        )

    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_rand_block_size_determinism(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_kernel_2d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return output

        x = torch.ones(64, 128, device=DEVICE)
        seed = 42

        block_sizes = _rng_determinism_block_sizes()
        compiled_kernels = [
            _compile_only(rand_kernel_2d, (x, seed), block_sizes=bs)
            for bs in block_sizes
        ]
        outputs = [compiled(x, seed) for compiled in compiled_kernels]

        for block_sizes_variant, output in zip(
            block_sizes[1:], outputs[1:], strict=True
        ):
            torch.testing.assert_close(
                outputs[0],
                output,
                msg=(
                    "rand should be deterministic across different block sizes "
                    f"({block_sizes[0]} vs {block_sizes_variant})"
                ),
            )

        self.assertTrue(torch.all(outputs[0] >= 0.0))
        self.assertTrue(torch.all(outputs[0] < 1.0))

    def test_hl_rand_uniqueness_distribution(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return output

        x = torch.ones(256, 256, device=DEVICE)
        seed = 1337

        _, output = code_and_output(
            rand_kernel, (x, seed), block_sizes=_rng_2d_block_sizes()
        )

        sorted_values = torch.sort(output.flatten()).values.cpu()

        unique_values = torch.unique(sorted_values)
        total_values = output.numel()
        uniqueness_ratio = len(unique_values) / total_values

        self.assertGreater(
            uniqueness_ratio,
            0.99,
            f"Expected >99% unique values, got {uniqueness_ratio:.4f}",
        )

        n_quartile = total_values // 4
        q1_val = sorted_values[n_quartile].item()
        q2_val = sorted_values[2 * n_quartile].item()
        q3_val = sorted_values[3 * n_quartile].item()

        self.assertTrue(
            0.2 < q1_val < 0.3, f"First quartile {q1_val:.3f} should be around 0.25"
        )
        self.assertTrue(
            0.45 < q2_val < 0.55, f"Median {q2_val:.3f} should be around 0.5"
        )
        self.assertTrue(
            0.7 < q3_val < 0.8, f"Third quartile {q3_val:.3f} should be around 0.75"
        )

    @xfailIfPallas(
        "specialized non-tiled hl.rand hits TPU Mosaic unsupported shape cast"
    )
    def test_hl_rand_non_tiled_dimensions(self):
        @helion.kernel(static_shapes=False)
        def rand_kernel_partial_tile(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n, k = x.shape
            k = hl.specialize(k)
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n, :] = hl.rand([tile_m, tile_n, k], seed=seed)
            return output

        x = torch.ones(64, 64, 8, device=DEVICE)
        seed = 1337

        _, output = code_and_output(
            rand_kernel_partial_tile, (x, seed), block_sizes=[8, 8]
        )

        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output < 1.0))

        code2, output2 = code_and_output(
            rand_kernel_partial_tile, (x, seed), block_sizes=[8, 8]
        )
        torch.testing.assert_close(output, output2, msg="it should deterministic")

    @skipIfMTIA("Skip on MTIA due to unaligned address crash")
    @xfailIfPallas("reordered tile dims cause BlockSpec axis mismatch")
    def test_hl_rand_mixed_argument_order(self):
        @helion.kernel(static_shapes=False)
        def rand_kernel_normal_order(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n, k = x.shape
            for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
                output[tile_m, tile_n, tile_k] = hl.rand(
                    [tile_m, tile_n, tile_k], seed=seed
                )
            return output

        @helion.kernel(static_shapes=False)
        def rand_kernel_mixed_order(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n, k = x.shape
            for tile_k, tile_m, tile_n in hl.tile([k, m, n]):
                output[tile_m, tile_n, tile_k] = hl.rand(
                    [tile_m, tile_n, tile_k], seed=seed
                )
            return output

        x = torch.ones(16, 32, 8, device=DEVICE)
        seed = 1337
        block_sizes = [4, 8, 8]

        output1 = _compile_only(
            rand_kernel_normal_order, (x, seed), block_sizes=block_sizes
        )(x, seed)
        output2 = _compile_only(
            rand_kernel_mixed_order, (x, seed), block_sizes=block_sizes
        )(x, seed)

        torch.testing.assert_close(
            output1,
            output2,
            msg="Mixed tile argument order should produce identical results",
        )

    @xfailIfPallas("rolled reductions not supported for pallas rand")
    @skipIfRocm("ROCm Triton worker crashes on rand with rolled reductions")
    def test_hl_rand_rolled_reductions(self):
        @helion.kernel(static_shapes=False)
        def rand_kernel_with_reduction(x: torch.Tensor, seed: int) -> torch.Tensor:
            m, n = x.shape
            output = torch.zeros([m], device=x.device)
            for tile_m in hl.tile(m):
                tile_values = x[tile_m, :]
                rand_values = hl.rand([tile_m], seed=seed)
                mean_val = tile_values.mean(-1)
                output[tile_m] = rand_values * mean_val
            return output

        x = torch.ones(64, 128, device=DEVICE)
        seed = 42

        code1, output_persistent = code_and_output(
            rand_kernel_with_reduction,
            (x, seed),
            block_sizes=[32],
            reduction_loops=[None],
        )
        code2, output_rolled = code_and_output(
            rand_kernel_with_reduction,
            (x, seed),
            block_sizes=[32],
            reduction_loops=[64],
        )

        torch.testing.assert_close(
            output_persistent,
            output_rolled,
            msg="Persistent and rolled reductions should produce identical results",
        )

    def test_hl_randint_1d(self):
        """Test hl.randint with 1D output."""

        @helion.kernel(static_shapes=False)
        def randint_kernel_1d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
            (m,) = x.shape
            for tile_m in hl.tile(m):
                output[tile_m] = hl.randint([tile_m], low=0, high=100, seed=seed)
            return output

        x = torch.ones(1024, device=DEVICE)
        _, output = code_and_output(randint_kernel_1d, (x, 42), block_sizes=[1024])
        _, output2 = code_and_output(randint_kernel_1d, (x, 1337), block_sizes=[1024])

        # Different seeds should produce different outputs
        self.assertFalse(
            torch.allclose(output.float(), output2.float()),
            "Different seeds should produce different outputs",
        )

        # Same seed should produce identical outputs
        code3, output3 = code_and_output(randint_kernel_1d, (x, 42), block_sizes=[1024])
        self.assertTrue(
            torch.allclose(output.float(), output3.float()),
            "Same seed should produce identical outputs",
        )
        _assert_uses_philox(self, code3)

        # Check that all values are in [0, 100) range
        self.assertTrue(torch.all(output >= 0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 100), "All values should be < 100")
        self.assertEqual(output.dtype, torch.int32, "Output dtype should be int32")

    def test_hl_randint_2d(self):
        """Test hl.randint with 2D output."""

        @helion.kernel(static_shapes=False)
        def randint_kernel_2d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=10, high=50, seed=seed
                )
            return output

        x = torch.ones(1024, 1024, device=DEVICE)
        block_sizes = _rng_2d_block_sizes()
        _, output = code_and_output(randint_kernel_2d, (x, 42), block_sizes=block_sizes)

        # Check that all values are in [10, 50) range
        self.assertTrue(torch.all(output >= 10), "All values should be >= 10")
        self.assertTrue(torch.all(output < 50), "All values should be < 50")

        code2, output2 = code_and_output(
            randint_kernel_2d, (x, 42), block_sizes=block_sizes
        )
        torch.testing.assert_close(
            output, output2, msg="Same seed should be deterministic"
        )

    def test_hl_randint_negative_range(self):
        """Test hl.randint with negative range."""

        @helion.kernel(static_shapes=False)
        def randint_kernel_neg(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
            (m,) = x.shape
            for tile_m in hl.tile(m):
                output[tile_m] = hl.randint([tile_m], low=-50, high=50, seed=seed)
            return output

        x = torch.ones(1024, device=DEVICE)
        code, output = code_and_output(randint_kernel_neg, (x, 42), block_sizes=[1024])

        # Check that all values are in [-50, 50) range
        self.assertTrue(torch.all(output >= -50), "All values should be >= -50")
        self.assertTrue(torch.all(output < 50), "All values should be < 50")

        # Check that we have both negative and positive values (statistically very likely)
        self.assertTrue(torch.any(output < 0), "Should have some negative values")
        self.assertTrue(torch.any(output >= 0), "Should have some non-negative values")

    def test_hl_rand_static_shapes(self):
        """Test hl.rand with static_shapes=True (default)."""

        @helion.kernel(static_shapes=True)
        def rand_kernel_static(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            (m,) = x.shape
            for tile_m in hl.tile(m):
                output[tile_m] = hl.rand([tile_m], seed=seed)
            return output

        x = torch.ones(128, device=DEVICE)
        _, output = code_and_output(rand_kernel_static, (x, 1337), block_sizes=[128])
        code, output2 = code_and_output(
            rand_kernel_static, (x, 1337), block_sizes=[128]
        )
        torch.testing.assert_close(
            output, output2, msg="Same seed should produce identical outputs"
        )

    def test_hl_randint_static_shapes(self):
        """Test hl.randint with static_shapes=True (default)."""

        @helion.kernel(static_shapes=True)
        def randint_kernel_static(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
            (m,) = x.shape
            for tile_m in hl.tile(m):
                output[tile_m] = hl.randint([tile_m], low=0, high=100, seed=seed)
            return output

        x = torch.ones(1024, device=DEVICE)
        _, output = code_and_output(randint_kernel_static, (x, 42), block_sizes=[1024])
        code, output2 = code_and_output(
            randint_kernel_static, (x, 42), block_sizes=[1024]
        )
        torch.testing.assert_close(
            output, output2, msg="Same seed should produce identical outputs"
        )

    @skipIfMTIA(
        "MTIA requires all tensor inputs to be aligned and/or padded according to the MTIA HW requirements"
    )
    def test_hl_rand_specialize(self):
        @helion.kernel()
        def fn(out: torch.Tensor, seed: int) -> torch.Tensor:
            m = out.size(0)
            n = hl.specialize(out.size(1))
            for tile_m in hl.tile(m):
                out[tile_m, :] = hl.rand([tile_m, n], seed=seed)

        out = torch.empty(128, 1, device=DEVICE)
        _, output = code_and_output(fn, (out, 1337), block_sizes=[128])
        code, output2 = code_and_output(fn, (out, 1337), block_sizes=[128])
        torch.testing.assert_close(
            output, output2, msg="Same seed should produce identical outputs"
        )

    @skipIfMTIA(
        "MTIA requires all tensor inputs to be aligned and/or padded according to the MTIA HW requirements"
    )
    def test_hl_randint_specialize(self):
        @helion.kernel()
        def fn(out: torch.Tensor, seed: int) -> torch.Tensor:
            m = out.size(0)
            n = hl.specialize(out.size(1))
            for tile_m in hl.tile(m):
                out[tile_m, :] = hl.randint([tile_m, n], low=15, high=75, seed=seed)

        out = torch.empty(128, 1, device=DEVICE)
        _, output = code_and_output(fn, (out, 1337), block_sizes=[128])
        code, output2 = code_and_output(fn, (out, 1337), block_sizes=[128])
        torch.testing.assert_close(
            output, output2, msg="Same seed should produce identical outputs"
        )


@onlyBackends(["triton", "cute"])
@skipIfRefEager("compiled RNG parity checks are not applicable in ref eager mode")
@skipIfMTIA(
    "Skip due to unaligned/unpadded tensor inputs that don't meet MTIA HW requirements"
)
class TestRandomPhiloxParity(TestCase):
    def test_reference_matches_triton(self):
        seed = 42
        offsets = torch.arange(257, device=DEVICE, dtype=torch.int64)

        triton_rand = _triton_rand_reference(seed, offsets)
        ref_rand = philox_rand_ref(seed, offsets)
        _assert_bitwise_equal_float(self, triton_rand, ref_rand)

        low, high = -17, 29
        triton_randint = _triton_randint_reference(seed, offsets, low, high)
        ref_randint = philox_randint_ref(seed, offsets, low, high)
        self.assertTrue(torch.equal(triton_randint.cpu(), ref_randint.cpu()))

    def test_hl_rand_matches_triton_reference(self):
        @helion.kernel(static_shapes=False)
        def rand_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return out

        shape = (17, 19)
        x = torch.empty(shape, device=DEVICE, dtype=torch.float32)
        seed = 1337
        _code, out = code_and_output(rand_kernel, (x, seed), block_sizes=[8, 16])
        offsets = torch.arange(out.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_rand_reference(seed, offsets).reshape(shape)
        _assert_bitwise_equal_float(self, out, expected)

    def test_hl_randint_matches_triton_reference(self):
        @helion.kernel(static_shapes=False)
        def randint_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-11, high=23, seed=seed
                )
            return out

        shape = (15, 21)
        x = torch.empty(shape, device=DEVICE, dtype=torch.int32)
        seed = 2024
        _code, out = code_and_output(randint_kernel, (x, seed), block_sizes=[8, 8])
        offsets = torch.arange(out.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_randint_reference(
            seed,
            offsets,
            -11,
            23,
        ).reshape(shape)
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_hl_rand_outer_loop_offsets_match_triton_reference(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_outer_loop_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_n in hl.tile(n):
                for tile_m in hl.tile(m):
                    values = hl.rand([tile_m], seed=seed)
                    out[tile_m, tile_n] = values[:, None].expand(tile_m, tile_n)
            return out

        shape = (17, 23)
        block_sizes = (8, 8)
        x = torch.empty(shape, device=DEVICE, dtype=torch.float32)
        seed = 314
        _code, out = code_and_output(
            rand_outer_loop_kernel,
            (x, seed),
            block_sizes=list(block_sizes),
        )
        expected = _hl_rand_outer_loop_expected(shape, block_sizes, seed)
        _assert_bitwise_equal_float(self, out, expected)

    def test_hl_randint_outer_loop_offsets_match_triton_reference(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def randint_outer_loop_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_n in hl.tile(n):
                for tile_m in hl.tile(m):
                    values = hl.randint([tile_m], low=-9, high=19, seed=seed)
                    out[tile_m, tile_n] = values[:, None].expand(tile_m, tile_n)
            return out

        shape = (15, 21)
        block_sizes = (8, 8)
        x = torch.empty(shape, device=DEVICE, dtype=torch.int32)
        seed = 2718
        _code, out = code_and_output(
            randint_outer_loop_kernel,
            (x, seed),
            block_sizes=list(block_sizes),
        )
        expected = _hl_randint_outer_loop_expected(shape, block_sizes, seed, -9, 19)
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_hl_rand_multiple_calls_reuse_explicit_seed_stream(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_twice_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                out[0, tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
                out[1, tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return out

        x = torch.empty((11, 13), device=DEVICE, dtype=torch.float32)
        seed = 777
        _code, out = code_and_output(rand_twice_kernel, (x, seed), block_sizes=[8, 8])
        offsets = torch.arange(x.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_rand_reference(seed, offsets).reshape_as(x)
        _assert_bitwise_equal_float(self, out[0], expected)
        _assert_bitwise_equal_float(self, out[1], expected)
        _assert_bitwise_equal_float(self, out[0], out[1])

    def test_hl_rand_helper_invocations_reuse_explicit_seed_stream(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_helper_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                out[0, tile_m, tile_n] = _helper_seeded_rand(tile_m, tile_n, seed)
                out[1, tile_m, tile_n] = _helper_seeded_rand(tile_m, tile_n, seed)
            return out

        x = torch.empty((11, 13), device=DEVICE, dtype=torch.float32)
        seed = 1776
        _code, out = code_and_output(rand_helper_kernel, (x, seed), block_sizes=[8, 8])
        offsets = torch.arange(x.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_rand_reference(seed, offsets).reshape_as(x)
        _assert_bitwise_equal_float(self, out[0], expected)
        _assert_bitwise_equal_float(self, out[1], expected)
        _assert_bitwise_equal_float(self, out[0], out[1])

    def test_hl_rand_sibling_loops_reuse_explicit_seed_stream(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_sibling_loops_kernel(
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
        seed = 1777
        _code, (out0, out1) = code_and_output(
            rand_sibling_loops_kernel, (x, seed), block_sizes=[8, 16, 8, 16]
        )
        offsets = torch.arange(x.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_rand_reference(seed, offsets).reshape_as(x)
        _assert_bitwise_equal_float(self, out0, expected)
        _assert_bitwise_equal_float(self, out1, expected)
        _assert_bitwise_equal_float(self, out0, out1)

    def test_hl_randint_sibling_loops_reuse_explicit_seed_stream(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def randint_sibling_loops_kernel(
            x: torch.Tensor, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out0 = torch.zeros_like(x)
            out1 = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out0[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-5, high=17, seed=seed
                )
            for tile_m, tile_n in hl.tile([m, n]):
                out1[tile_m, tile_n] = hl.randint(
                    [tile_m, tile_n], low=-5, high=17, seed=seed
                )
            return out0, out1

        x = torch.empty((9, 15), device=DEVICE, dtype=torch.int32)
        seed = 201
        _code, (out0, out1) = code_and_output(
            randint_sibling_loops_kernel, (x, seed), block_sizes=[8, 16, 8, 16]
        )
        offsets = torch.arange(x.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_randint_reference(seed, offsets, -5, 17).reshape_as(x)
        self.assertTrue(torch.equal(out0.cpu(), expected.cpu()))
        self.assertTrue(torch.equal(out1.cpu(), expected.cpu()))
        self.assertTrue(torch.equal(out0.cpu(), out1.cpu()))

    def test_hl_randint_helper_invocations_reuse_explicit_seed_stream(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def randint_helper_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                out[0, tile_m, tile_n] = _helper_seeded_randint(tile_m, tile_n, seed)
                out[1, tile_m, tile_n] = _helper_seeded_randint(tile_m, tile_n, seed)
            return out

        x = torch.empty((9, 15), device=DEVICE, dtype=torch.int32)
        seed = 202
        _code, out = code_and_output(
            randint_helper_kernel, (x, seed), block_sizes=[8, 16]
        )
        offsets = torch.arange(x.numel(), device=DEVICE, dtype=torch.int64)
        expected = _triton_randint_reference(seed, offsets, -5, 17).reshape_as(x)
        self.assertTrue(torch.equal(out[0].cpu(), expected.cpu()))
        self.assertTrue(torch.equal(out[1].cpu(), expected.cpu()))
        self.assertTrue(torch.equal(out[0].cpu(), out[1].cpu()))

    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_rand_ref_mode_matches_compiled(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_outer_loop_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_n in hl.tile(n):
                for tile_m in hl.tile(m):
                    values = hl.rand([tile_m], seed=seed)
                    out[tile_m, tile_n] = values[:, None].expand(tile_m, tile_n)
            return out

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            ref_mode=helion.RefMode.EAGER,
        )
        def rand_outer_loop_kernel_ref(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_n in hl.tile(n):
                for tile_m in hl.tile(m):
                    values = hl.rand([tile_m], seed=seed)
                    out[tile_m, tile_n] = values[:, None].expand(tile_m, tile_n)
            return out

        x = torch.empty((15, 21), device=DEVICE, dtype=torch.float32)
        seed = 31415
        _code, compiled = _compile_once(
            rand_outer_loop_kernel, (x, seed), block_sizes=[8, 8]
        )
        compiled_out = compiled(x, seed)
        ref_out = rand_outer_loop_kernel_ref(x, seed)
        _assert_bitwise_equal_float(self, compiled_out, ref_out)

    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_rand_helper_ref_mode_matches_compiled(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def rand_helper_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                out[0, tile_m, tile_n] = _helper_seeded_rand(tile_m, tile_n, seed)
                out[1, tile_m, tile_n] = _helper_seeded_rand(tile_m, tile_n, seed)
            return out

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            ref_mode=helion.RefMode.EAGER,
        )
        def rand_helper_kernel_ref(x: torch.Tensor, seed: int) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                out[0, tile_m, tile_n] = _helper_seeded_rand(tile_m, tile_n, seed)
                out[1, tile_m, tile_n] = _helper_seeded_rand(tile_m, tile_n, seed)
            return out

        x = torch.empty((11, 13), device=DEVICE, dtype=torch.float32)
        seed = 314159
        _code, compiled = _compile_once(
            rand_helper_kernel, (x, seed), block_sizes=[8, 8]
        )
        compiled_out = compiled(x, seed)
        ref_out = rand_helper_kernel_ref(x, seed)
        _assert_bitwise_equal_float(self, compiled_out, ref_out)

    @skipIfRefEager("compile_config is not supported in ref eager mode")
    def test_hl_randint_ref_mode_matches_compiled(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def randint_outer_loop_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_n in hl.tile(n):
                for tile_m in hl.tile(m):
                    values = hl.randint([tile_m], low=-7, high=23, seed=seed)
                    out[tile_m, tile_n] = values[:, None].expand(tile_m, tile_n)
            return out

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            ref_mode=helion.RefMode.EAGER,
        )
        def randint_outer_loop_kernel_ref(x: torch.Tensor, seed: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, n = x.shape
            for tile_n in hl.tile(n):
                for tile_m in hl.tile(m):
                    values = hl.randint([tile_m], low=-7, high=23, seed=seed)
                    out[tile_m, tile_n] = values[:, None].expand(tile_m, tile_n)
            return out

        x = torch.empty((15, 21), device=DEVICE, dtype=torch.int32)
        seed = 27182
        _code, compiled = _compile_once(
            randint_outer_loop_kernel, (x, seed), block_sizes=[8, 8]
        )
        compiled_out = compiled(x, seed)
        ref_out = randint_outer_loop_kernel_ref(x, seed)
        self.assertTrue(torch.equal(compiled_out.cpu(), ref_out.cpu()))


if __name__ == "__main__":
    unittest.main()
