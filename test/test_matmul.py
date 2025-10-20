from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import helion
from helion import Config
from helion import _compat
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
import helion.language as hl

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
examples_dir = Path(__file__).parent.parent / "examples"
examples_matmul = import_path(examples_dir / "matmul.py").matmul


@helion.kernel
def matmul_with_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel
def matmul_without_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


class TestMatmul(RefEagerTestBase, TestCase):
    def test_matmul0(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_without_addmm,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    def test_matmul1(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            examples_matmul,
            args,
            block_sizes=[16, 16, 16],
            loop_order=[1, 0],
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    def test_matmul3(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_addmm,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_matmul_block_ptr(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            examples_matmul,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
            indexing="block_ptr",
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
    @skipIfRefEager("to_triton_code is not supported in ref eager mode")
    def test_matmul_tensor_descriptor(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        config = Config(
            block_sizes=[16, 16, 16],
            l2_grouping=4,
            indexing="tensor_descriptor",
        )
        # Note TensorDescriptor doesn't run on older cards
        code = examples_matmul.bind(args).to_triton_code(config)
        self.assertExpectedJournal(code)

    def test_matmul_static_shapes0(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
            indexing="pointer",
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    def test_matmul_static_shapes1(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    def test_matmul_static_shapes2(self):
        args = (
            torch.randn([128, 127], device=DEVICE, dtype=torch.float32),
            torch.randn([127, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    def test_matmul_static_shapes3(self):
        args = (
            torch.randn([127, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 127], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[16, 16, 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    def test_matmul_packed_int4_block_size_constexpr(self):
        torch.manual_seed(0)
        M = N = K = 32

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def matmul_bf16_packed_int4(
            A: torch.Tensor, B_packed: torch.Tensor, C: torch.Tensor
        ) -> torch.Tensor:
            M0, K0 = A.shape
            _, N0 = B_packed.shape

            block_n = hl.register_block_size(N0)
            block_k = hl.register_block_size(K0)

            for tile_m in hl.tile(M0):
                for tile_n in hl.tile(N0, block_size=block_n):
                    acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)

                    for tile_k in hl.tile(K0, block_size=block_k):
                        tile_k_begin = tile_k.begin
                        b_tile = B_packed[
                            tile_k_begin // 2 : tile_k_begin // 2 + block_k // 2,
                            tile_n,
                        ]
                        shift = hl.full((1,), 4, dtype=torch.int8)
                        b_lo = (b_tile << shift) >> shift
                        b_hi = b_tile >> shift
                        stacked = torch.stack(
                            (b_lo.to(A.dtype), b_hi.to(A.dtype)), dim=2
                        )
                        stacked = stacked.permute(0, 2, 1)
                        b_block = stacked.reshape([block_k, block_n])
                        acc = hl.dot(A[tile_m, tile_k], b_block, acc=acc)

                    C[tile_m, tile_n] = acc

            return C

        A = torch.randn((M, K), dtype=torch.bfloat16, device=DEVICE)
        B_packed = torch.randint(0, 16, (K // 2, N), dtype=torch.int8, device=DEVICE)
        C = torch.zeros((M, N), dtype=torch.float32, device=DEVICE)

        matmul_bf16_packed_int4(A, B_packed, C)
        torch.accelerator.synchronize()

        self.assertTrue(torch.isfinite(C).all())
        self.assertFalse(torch.allclose(C, torch.zeros_like(C)))

    def test_matmul_split_k(self):
        @helion.kernel(dot_precision="ieee")
        def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n, outer_k in hl.tile([m, n, k]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for inner_k in hl.tile(outer_k.begin, outer_k.end):
                    acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
                hl.atomic_add(out, [tile_m, tile_n], acc)
            return out

        x = torch.randn([32, 2000], device=DEVICE, dtype=torch.float32)
        y = torch.randn([2000, 32], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            matmul_split_k,
            (x, y),
            block_sizes=[16, 16, 256, 32],
            indexing="pointer",
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)

    @skipIfRocm("ROCm triton error in TritonAMDGPUBlockPingpong")
    @skipIfRefEager("config_spec is not supported in ref eager mode")
    def test_matmul_config_reuse_with_unit_dim(self):
        torch.manual_seed(0)
        big_args = (
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        big_bound = matmul_with_addmm.bind(big_args)
        big_spec = big_bound.config_spec
        self.assertEqual(len(big_spec.block_sizes), 3)
        big_config = big_spec.default_config()

        small_args = (
            torch.randn([1, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 64], device=DEVICE, dtype=torch.float32),
        )
        small_bound = matmul_with_addmm.bind(small_args)
        small_spec = small_bound.config_spec
        self.assertEqual(len(small_spec.block_sizes), 3)

        # Previously raised when reusing configs tuned on larger shapes.
        small_bound.set_config(big_config)
        result = small_bound(*small_args)
        expected = small_args[0] @ small_args[1]
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)

    def test_matmul_packed_rhs(self):
        @helion.kernel(static_shapes=False)
        def matmul_with_packed_b(
            A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
        ) -> None:
            M, K = A.shape
            _, N = B.shape

            block_size_k = hl.register_block_size(K // 2)

            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=A.dtype)

                for tile_k in hl.tile(K // 2, block_size=block_size_k):
                    lhs = A[
                        tile_m,
                        tile_k.begin * 2 : tile_k.begin * 2 + tile_k.block_size * 2,
                    ]
                    packed = B[tile_k, tile_n]
                    rhs = torch.stack([packed, packed], dim=1).reshape(
                        tile_k.block_size * 2, tile_n.block_size
                    )
                    acc = torch.addmm(acc, lhs, rhs)

                C[tile_m, tile_n] = acc

        M, K, N = 32, 64, 32
        A = torch.randn(M, K, device=DEVICE)
        B = torch.randn(K // 2, N, device=DEVICE)
        C = torch.empty(M, N, device=DEVICE)
        code, _ = code_and_output(matmul_with_packed_b, (A, B, C))
        B_unpacked = torch.stack([B, B], dim=1).reshape(K, N)
        expected = A @ B_unpacked
        torch.testing.assert_close(C, expected, atol=5e-2, rtol=1e-3)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
