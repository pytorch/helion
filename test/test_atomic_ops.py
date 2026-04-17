from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import xfailIfCute
from helion._testing import xfailIfPallas
import helion.language as hl
from helion.runtime.settings import _get_backend


@helion.kernel()
def atomic_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test basic atomic_add functionality."""
    for i in hl.tile(x.size(0)):
        hl.atomic_add(x, [i], y[i])
    return x


@helion.kernel(static_shapes=True)
def atomic_add_overlap_kernel(
    x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Test atomic_add with overlapping indices."""
    for i in hl.tile([y.size(0)]):
        idx = indices[i]
        hl.atomic_add(x, [idx], y[i])
    return x


@helion.kernel()
def atomic_add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add with 2D indexing."""
    for i, j in hl.tile([y.size(0), y.size(1)]):
        hl.atomic_add(x, [i, j], y[i, j])
    return x


@helion.kernel()
def atomic_add_float_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Test atomic_add with a float constant value and reading from lookup"""
    for i in hl.tile(indices.size(0)):
        idx = indices[i]
        hl.atomic_add(x, [idx], 2.0)
    return x


@helion.kernel(static_shapes=True)
def split_k_atomic_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Split-K matmul with atomic_add into a non-zero output."""
    m, k = x.size()
    k2, n = y.size()
    out = torch.ones([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(tile_k.begin, tile_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc.to(x.dtype))
    return out


@helion.kernel()
def atomic_add_f32_into_bf16_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where value dtype (float32) differs from output (bfloat16)."""
    m, n = x.size()
    out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = acc + x[tile_m, tile_n].to(torch.float32)
        acc = acc + y[tile_m, tile_n].to(torch.float32)
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


@helion.kernel()
def atomic_add_w_tile_attr(x: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where the index is a symbolic int"""
    y = torch.zeros_like(x, device=x.device, dtype=torch.int32)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(y, [tile.begin], 1)
    return y


@helion.kernel()
def atomic_add_tile_begin_reduce_other_axis(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0)], device=x.device, dtype=x.dtype)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_add(out, [tile_m.begin], x[tile_m, tile_n])
    return out


@helion.kernel()
def atomic_add_1d_tensor_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where the index is a 1D tensor"""
    m, n = x.shape
    n = hl.specialize(n)

    z = torch.zeros([n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        y_tile = y[tile_m, :].to(torch.float32)
        z_vec = torch.sum(x_tile * y_tile, dim=0).to(x.dtype)
        hl.atomic_add(z, [hl.arange(0, n)], z_vec)

    return z


# New kernels for other atomics


@helion.kernel()
def atomic_and_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_and(x, [i], y[i])
    return x


@helion.kernel()
def atomic_or_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_or(x, [i], y[i])
    return x


@helion.kernel()
def atomic_xor_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_xor(x, [i], y[i])
    return x


@helion.kernel()
def atomic_xchg_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_xchg(x, [i], y[i])
    return x


@helion.kernel()
def atomic_max_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_max(x, [i], y[i])
    return x


@helion.kernel()
def atomic_max_return_kernel(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        out[i] = hl.atomic_max(x, [i], y[i])
    return out


@helion.kernel()
def atomic_min_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_min(x, [i], y[i])
    return x


@helion.kernel()
def atomic_cas_kernel(
    x: torch.Tensor, y: torch.Tensor, expect: torch.Tensor
) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_cas(x, [i], expect[i], y[i])
    return x


# 2D kernels for tensor descriptor atomic tests (TD requires ndim >= 2 + static_shapes)


@helion.kernel(static_shapes=True)
def atomic_add_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_add(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_and_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_and(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_or_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_or(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_xor_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_xor(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_max_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_max(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_min_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_min(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_xchg_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_xchg(x, [i, j], y[i, j])
    return x


@onlyBackends(["triton", "cute", "pallas"])
class TestAtomicOperations(RefEagerTestBase, TestCase):
    def test_basic_atomic_add(self):
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_kernel,
            args,
            block_sizes=[32],
        )

        expected = torch.ones(10, device=DEVICE)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_add", code)

    @xfailIfPallas("view-backed atomic_add targets are not supported on Pallas")
    def test_basic_atomic_add_strided_target(self):
        x_base = torch.zeros(16, device=DEVICE)
        x = x_base[::2]
        y = torch.ones(8, device=DEVICE)

        code, result = code_and_output(
            atomic_add_kernel,
            (x, y),
            block_sizes=[32],
        )

        expected = torch.ones(8, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @xfailIfCute("cute: hl.arange atomic scatter requires an active non-reduction axis")
    @xfailIfPallas("Integer indexing not supported on Pallas")
    def test_atomic_add_1d_tensor(self):
        M, N = 32, 64
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        y = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_1d_tensor_kernel,
            args,
            block_sizes=[32],
        )

        expected = (x * y).sum(dim=0)
        torch.testing.assert_close(result, expected)

    def test_atomic_add_returns_prev(self):
        @helion.kernel()
        def k(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            prev = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                old = hl.atomic_add(x, [i], y[i])
                prev[i] = old
            return x, prev

        x = torch.zeros(8, device=DEVICE)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, (out, prev) = code_and_output(k, (x, y))
        torch.testing.assert_close(out, y)
        torch.testing.assert_close(prev, torch.zeros_like(x))

    @xfailIfPallas("gather indexing with different-sized tensors unsupported on Pallas")
    def test_overlapping_atomic_add(self):
        # Test with overlapping indices
        x = torch.zeros(5, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], device=DEVICE)
        args = (x, y, indices)

        code, result = code_and_output(
            atomic_add_overlap_kernel,
            args,
            block_sizes=[32],
        )

        expected = torch.ones(5, device=DEVICE) * 2
        torch.testing.assert_close(result, expected)

    def test_2d_atomic_add(self):
        """Test atomic_add with 2D tensor indexing."""
        x = torch.zeros(3, 4, device=DEVICE)
        y = torch.ones(3, 4, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_2d_kernel,
            args,
            block_sizes=[8, 8],
        )

        expected = torch.ones(3, 4, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @onlyBackends(["pallas"])
    def test_atomic_add_f32_into_bf16(self):
        """atomic_add of a float32 value into a bfloat16 output tensor."""
        x = torch.ones(64, 128, device=DEVICE, dtype=torch.bfloat16)
        y = torch.ones(64, 128, device=DEVICE, dtype=torch.bfloat16)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_f32_into_bf16_kernel,
            args,
            block_sizes=[64, 128],
        )

        expected = (x.float() + y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected)

    @onlyBackends(["pallas"])
    def test_split_k_atomic_add_vmem_preload(self):
        """Split-K matmul where output is initialised to ones (not zeros)."""
        m, k, n = 128, 1024, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        args = (x, y)

        code, result = code_and_output(
            split_k_atomic_add_kernel,
            args,
            block_sizes=[128, 128, 1024, 128],
            pallas_loop_type="fori_loop",
        )

        # expected = 1 + x @ y  (ones init + matmul via atomic_add)
        expected = torch.ones(m, n, device=DEVICE, dtype=torch.bfloat16) + (x @ y).to(
            torch.bfloat16
        )
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.05)

    def test_atomic_add_code_generation(self):
        """Test that the generated code contains atomic_add."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(atomic_add_kernel, args)
        expected = torch.ones(10, device=DEVICE)
        torch.testing.assert_close(result, expected)
        self.assertIn("atomic_add", code)

    @xfailIfPallas("int64 index dtype causes MLIR type mismatch on TPU")
    def test_atomic_add_float(self):
        """Test that atomic_add works with float constants."""
        x = torch.zeros(5, device=DEVICE, dtype=torch.float32)

        indices = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4], device=DEVICE)
        expected = torch.tensor(
            [2.0, 2.0, 4.0, 6.0, 2.0], device=DEVICE, dtype=torch.float32
        )

        args = (x, indices)
        code, result = code_and_output(
            atomic_add_float_kernel,
            args,
            block_sizes=[32],
        )

        torch.testing.assert_close(result, expected)

    def test_atomic_add_invalid_sem(self):
        """Test that atomic_add raises with an invalid sem value."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)

        @helion.kernel()
        def bad_atomic_add_kernel(x: torch.Tensor, y: torch.Tensor):
            for i in hl.tile(x.size(0)):
                hl.atomic_add(x, [i], y[i], sem="ERROR")
            return x

        with self.assertRaises(helion.exc.InternalError) as ctx:
            code_and_output(
                bad_atomic_add_kernel,
                (x, y),
                block_sizes=[32],
            )
        self.assertIn("Invalid memory semantic 'ERROR'", str(ctx.exception))

    @xfailIfPallas("block_size=2 does not meet TPU alignment requirements")
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_atomic_add_w_tile_attr(self):
        """Test atomic_add where the index is a symbolic int"""
        x = torch.randn(20, device=DEVICE)
        code, result = code_and_output(
            atomic_add_w_tile_attr,
            (x,),
            block_sizes=[2],
        )

        expected = torch.tensor([1, 0], device=DEVICE, dtype=torch.int32).repeat(10)
        torch.testing.assert_close(result, expected)

    @xfailIfPallas(
        "atomic scalar-origin reduction pattern is only validated on GPU backends"
    )
    def test_atomic_add_tile_begin_reduce_other_axis(self):
        if _get_backend() != "cute":
            self.skipTest("CuTe regression coverage")
        x = torch.ones((4, 4), device=DEVICE)
        code, result = code_and_output(
            atomic_add_tile_begin_reduce_other_axis,
            (x,),
            block_sizes=[2, 2],
        )

        expected = torch.tensor([4, 0, 4, 0], device=DEVICE, dtype=x.dtype)
        torch.testing.assert_close(result, expected)

    @xfailIfPallas("AtomicOnDeviceTensor error message differs on Pallas")
    @skipIfRefEager("Error only raises in normal mode")
    def test_atomic_add_device_tensor_error(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0), block_size=128):
                device_tensor = hl.zeros([tile], dtype=x.dtype)
                hl.atomic_add(device_tensor, [tile], x[tile])
            return x

        x = torch.ones(256, device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(
            helion.exc.AtomicOnDeviceTensor,
            r"hl\.atomic_add\(\)",
        ):
            kernel(x)

    def test_atomic_and(self):
        x0 = torch.full((8,), 0b1111, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_and_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1111 & 0b1010, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_and", code)

    def test_atomic_or(self):
        x0 = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_or_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1010, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_or", code)

    def test_atomic_xor(self):
        x0 = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1100] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_xor_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1010 ^ 0b1100, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_xor", code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_xchg(self):
        x0 = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        y = torch.arange(8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_xchg_kernel, (x0.clone(), y))
        torch.testing.assert_close(result, y)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_xchg", code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_max(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_max_kernel, (x.clone(), y))
        expected = torch.tensor([4, 5, 9, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_max", code)

    def test_atomic_max_return_value(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        out = torch.empty(4, device=DEVICE, dtype=torch.int32)
        _, result = code_and_output(
            atomic_max_return_kernel, (x.clone(), y, out), block_sizes=[4]
        )
        # Return value should be the previous values of x
        expected = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_min(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_min_kernel, (x.clone(), y))
        expected = torch.tensor([1, 2, 3, 1], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_min", code)

    def test_atomic_cas(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        expect = torch.tensor([1, 6, 3, 0], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([9, 9, 9, 9], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_cas_kernel, (x.clone(), y, expect))
        # Only positions where expect matches original x are replaced
        expected = torch.tensor([9, 5, 9, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_cas", code)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    def test_atomic_td_fallbacks(self):
        """Test that tensor_descriptor atomics fall back to pointer when needed."""

        # Return value consumed: should fall back to pointer
        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing="tensor_descriptor",
            ),
            static_shapes=True,
        )
        def atomic_add_td_prev_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i, j in hl.tile([x.size(0), x.size(1)]):
                prev = hl.atomic_add(x, [i, j], y[i, j])
                out[i, j] = prev
            return out

        M, N = 128, 64
        x = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        y = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(atomic_add_td_prev_kernel, (x, y))
        expected = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        self.assertIn("tl.atomic_add", code)
        self.assertNotIn("desc.atomic_add(", code)

        # Non-relaxed sem: should fall back to pointer
        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing="tensor_descriptor",
            ),
            static_shapes=True,
        )
        def atomic_add_td_release_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            for i, j in hl.tile([x.size(0), x.size(1)]):
                hl.atomic_add(x, [i, j], y[i, j], sem="release")
            return x

        x2 = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        y2 = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        code2, result2 = code_and_output(atomic_add_td_release_kernel, (x2, y2))
        expected2 = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(result2, expected2)
        self.assertIn("tl.atomic_add", code2)
        self.assertNotIn("desc.atomic_add(", code2)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    def test_atomic_add_per_op_indexing(self):
        """Test per-op atomic_indexing list: first op pointer, second op tensor_descriptor."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing=["pointer", "tensor_descriptor"],
            ),
            static_shapes=True,
        )
        def two_atomic_adds(
            out1: torch.Tensor, out2: torch.Tensor, val: torch.Tensor
        ) -> torch.Tensor:
            for i, j in hl.tile([out1.size(0), out1.size(1)]):
                hl.atomic_add(out1, [i, j], val[i, j])  # pointer
                hl.atomic_add(out2, [i, j], val[i, j])  # tensor_descriptor
            return out1

        M, N = 128, 64
        out1 = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        out2 = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        val = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(two_atomic_adds, (out1, out2, val))
        expected = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(out2, expected)
        # out1 uses pointer: tl.atomic_add(out1 + ...)
        self.assertIn("tl.atomic_add(out1", code)
        # out2 uses tensor_descriptor: out2_desc.atomic_add(...)
        self.assertIn("out2_desc.atomic_add(", code)
        # out1 should NOT use descriptor, out2 should NOT use pointer
        self.assertNotIn("out1_desc", code)
        self.assertNotIn("tl.atomic_add(out2", code)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    def test_atomic_ops_tensor_descriptor(self):
        """Test all TMA-supported atomic ops generate desc.atomic_{op} codegen."""
        M, N = 128, 64
        td_config = {
            "block_sizes": [64, 64],
            "indexing": "tensor_descriptor",
            "atomic_indexing": "tensor_descriptor",
        }
        # (op_name, kernel, x, y, expected)
        cases = [
            (
                "atomic_add",
                atomic_add_2d_td_kernel,
                torch.zeros(M, N, device=DEVICE, dtype=torch.float32),
                torch.ones(M, N, device=DEVICE, dtype=torch.float32),
                torch.ones(M, N, device=DEVICE, dtype=torch.float32),
            ),
            (
                "atomic_and",
                atomic_and_2d_td_kernel,
                torch.full((M, N), 0b1111, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_or",
                atomic_or_2d_td_kernel,
                torch.zeros(M, N, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_xor",
                atomic_xor_2d_td_kernel,
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1100, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b0110, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_max",
                atomic_max_2d_td_kernel,
                torch.ones(M, N, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 5, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 5, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_min",
                atomic_min_2d_td_kernel,
                torch.full((M, N), 10, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 3, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 3, device=DEVICE, dtype=torch.int32),
            ),
        ]
        for op_name, kernel, x, y, expected in cases:
            with self.subTest(op=op_name):
                code, result = code_and_output(kernel, (x, y), **td_config)
                torch.testing.assert_close(result, expected)
                self.assertIn(f"desc.{op_name}(", code)
                self.assertNotIn(f"tl.{op_name}", code)

        # xchg is NOT a TMA reduction op — should fall back to pointer
        with self.subTest(op="atomic_xchg_fallback"):
            x = torch.zeros(M, N, device=DEVICE, dtype=torch.int32)
            y = torch.ones(M, N, device=DEVICE, dtype=torch.int32)
            code, result = code_and_output(
                atomic_xchg_2d_td_kernel, (x, y), **td_config
            )
            torch.testing.assert_close(
                result, torch.ones(M, N, device=DEVICE, dtype=torch.int32)
            )
            self.assertIn("tl.atomic_xchg", code)
            self.assertNotIn("desc.atomic_xchg", code)


if __name__ == "__main__":
    unittest.main()
