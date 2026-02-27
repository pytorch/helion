from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
import helion.language as hl


@helion.kernel(backend="pallas", static_shapes=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.relu(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sin(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_pointwise_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(torch.sin(torch.relu(x[tile] * y[tile])))
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_affine_scalar_args(
    x: torch.Tensor,
    scale: int,
    bias: float,
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * scale + bias
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sum_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_max_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_tile_begin_end(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + tile.begin - tile.end
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inplace_add(x: torch.Tensor, y: torch.Tensor) -> None:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(out.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_arange_add(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        offsets = hl.arange(m)
        out[tile_n, :] = x[tile_n, :] + offsets[None, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inner_loop_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel with an outer grid loop and an inner device loop."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@onlyBackends(["triton", "pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallas(TestCase):
    def test_add_1d(self) -> None:
        args = (torch.randn(1024, device=DEVICE), torch.randn(1024, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=256)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_large(self) -> None:
        args = (torch.randn(4096, device=DEVICE), torch.randn(4096, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=512)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_2d(self) -> None:
        args = (
            torch.randn(64, 512, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 512, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(pallas_add_2d, args, block_sizes=[8, 512])
        torch.testing.assert_close(result, args[0] + args[1])

    def test_arange(self) -> None:
        x = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
        offsets = torch.arange(64, device=DEVICE, dtype=torch.int32).float()
        code, result = code_and_output(pallas_arange_add, (x,), block_size=8)
        torch.testing.assert_close(result, x + offsets[None, :])
        self.assertIn("jnp.arange", code)

    def test_inplace_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        expected = x + y
        # Use block_size=1024 so grid=1; with grid>1 the full-array
        # access pattern causes inplace mutations to accumulate.
        code, result = code_and_output(pallas_inplace_add, (x, y), block_size=1024)
        # x should be mutated in place
        torch.testing.assert_close(x, expected)

    def test_pointwise_mul(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(pallas_mul, args, block_size=256)
        x, y = args
        torch.testing.assert_close(out, x * y)

    def test_pointwise_relu(self) -> None:
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_relu, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.relu(x))

    def test_pointwise_sin(self) -> None:
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_sin, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.sin(x))

    def test_pointwise_sigmoid(self) -> None:
        # float16 is not supported by TPU Pallas Mosaic lowering
        # ("Not implemented: offset not aligned to sublanes")
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_sigmoid, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.sigmoid(x), rtol=1e-5, atol=1e-5)

    def test_pointwise_chain(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(pallas_pointwise_chain, args, block_size=256)
        x, y = args
        expected = torch.sigmoid(torch.sin(torch.relu(x * y)))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_args(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            3,
            1.25,
        )
        code, out = code_and_output(pallas_affine_scalar_args, args, block_size=256)
        x, scale, bias = args
        torch.testing.assert_close(out, x * scale + bias, rtol=1e-5, atol=1e-5)

    @unittest.expectedFailure  # inductor DeviceProperties.create() has no TPU interface
    def test_sum_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        from helion.runtime.config import Config

        bound = pallas_sum_reduction.bind((x,))
        code = bound.to_triton_code(Config(block_size=16))
        self.assertIn("jnp.sum", code)

    @unittest.expectedFailure  # inductor DeviceProperties.create() has no TPU interface
    def test_max_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        from helion.runtime.config import Config

        bound = pallas_max_reduction.bind((x,))
        code = bound.to_triton_code(Config(block_size=16))
        self.assertIn("jnp.max", code)

    def test_tile_begin_end(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        from helion.runtime.config import Config

        bound = pallas_tile_begin_end.bind((x,))
        code = bound.to_triton_code(Config(block_size=256))
        self.assertIn("pl.program_id", code)

    def test_dynamic_scalar_no_recompile(self) -> None:
        """Verify that changing dynamic scalar values does not trigger recompilation."""
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        pallas_affine_scalar_args.reset()

        # First call - triggers compilation
        result1 = pallas_affine_scalar_args(x, 3, 1.25)
        self.assertEqual(len(pallas_affine_scalar_args._bound_kernels), 1)

        # Second call with different scalar values - should NOT recompile
        result2 = pallas_affine_scalar_args(x, 5, 2.5)
        self.assertEqual(len(pallas_affine_scalar_args._bound_kernels), 1)

        # Verify correctness
        torch.testing.assert_close(result1, x * 3 + 1.25, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result2, x * 5 + 2.5, rtol=1e-5, atol=1e-5)

    def test_inner_loop_add(self) -> None:
        """Test kernel with outer grid loop and inner device loop."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add, args, block_sizes=[8, 128]
        )
        self.assertIn("for ", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_emit_pipeline_codegen(self) -> None:
        """Test that pallas_loop_type='emit_pipeline' generates correct emit_pipeline code."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("pl.BlockSpec", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_fori_loop_codegen(self) -> None:
        """Test that pallas_loop_type='fori_loop' generates correct fori_loop code."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn("pltpu.make_async_copy", code)
        self.assertNotIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(result, args[0] + args[1])


if __name__ == "__main__":
    unittest.main()
