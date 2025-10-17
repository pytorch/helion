from __future__ import annotations

import re
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


class TestConstExpr(RefEagerTestBase, TestCase):
    def test_constexpr_float(self):
        @helion.kernel()
        def fn(x: torch.Tensor, v: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = torch.sigmoid(x[tile] + v)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, 5.0),
        )
        torch.testing.assert_close(result, torch.sigmoid(x + 5.0))
        self.assertExpectedJournal(code)

    def test_constexpr_float_wrapped(self):
        @helion.kernel()
        def fn(x: torch.Tensor, v: float) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = torch.sigmoid(x[tile] + v)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, hl.constexpr(5.0)),
        )
        torch.testing.assert_close(result, torch.sigmoid(x + 5.0))
        self.assertExpectedJournal(code)

    def test_constexpr_size(self):
        @helion.kernel()
        def fn(x: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
            (b,) = x.size()
            out = torch.empty([b, s], device=x.device, dtype=x.dtype)
            for tile_b, tile_s in hl.tile([b, s]):
                out[tile_b, tile_s] = x[tile_b].view(-1, 1).expand(tile_b, tile_s)
            return out

        x = torch.randn([512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, 16),
        )
        torch.testing.assert_close(result, x.view(-1, 1).expand(512, 16))
        self.assertExpectedJournal(code)

    def test_string_literal_arg(self):
        @helion.kernel()
        def fn(x: torch.Tensor, mode: str) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                if mode == "add":
                    out[tile] = x[tile] + 1.0
                elif mode == "mul":
                    out[tile] = x[tile] * 2.0
                else:
                    out[tile] = x[tile]
            return out

        x = torch.randn([512, 512], device=DEVICE)

        # Test "add" mode
        code, result = code_and_output(fn, (x, "add"))
        torch.testing.assert_close(result, x + 1.0)
        self.assertExpectedJournal(code)

        # Test "mul" mode
        code, result = code_and_output(fn, (x, "mul"))
        torch.testing.assert_close(result, x * 2.0)
        self.assertExpectedJournal(code)

        # Test default mode
        code, result = code_and_output(fn, (x, "default"))
        torch.testing.assert_close(result, x)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Triton codegen does not work in ref eager mode")
    def test_block_size_constexpr_assignment_in_host_code(self) -> None:
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1, 1, 16],
                indexing="pointer",
                l2_groupings=[8],
                loop_orders=[[0, 1]],
                num_stages=8,
                num_warps=1,
                pid_type="persistent_blocked",
                range_flattens=[True, True],
                range_multi_buffers=[None, False],
                range_num_stages=[3, 1],
                range_unroll_factors=[1, 4],
            ),
            static_shapes=True,
        )
        def matmul_int4_block_expr(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            M, K = A.shape
            _, N = B.shape

            C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)
            block_size_k_packed = hl.register_block_size(K // 2)

            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

                for tile_k_packed in hl.tile(K // 2, block_size=block_size_k_packed):
                    packed = B[tile_k_packed, tile_n]
                    lo = ((packed << 4) >> 4).to(torch.int8)
                    hi = (packed >> 4).to(torch.int8)
                    lo_bf16 = lo.to(torch.bfloat16)
                    hi_bf16 = hi.to(torch.bfloat16)
                    stacked = torch.stack([lo_bf16, hi_bf16], dim=1)
                    unpacked = stacked.reshape(
                        tile_k_packed.block_size * 2, tile_n.block_size
                    )

                    k_begin = tile_k_packed.begin * 2
                    k_len = tile_k_packed.block_size * 2
                    a_tile = A[tile_m, k_begin : (k_begin + k_len)]

                    acc = acc + hl.dot(a_tile, unpacked)

                C[tile_m, tile_n] = acc.to(torch.bfloat16)

            return C

        M, K, N = 16, 32, 16
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        B_unpacked = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=DEVICE)
        B_halves = B_unpacked.reshape(K // 2, 2, N).permute(1, 0, 2)
        B_packed = ((B_halves[0] & 0xF) | (B_halves[1] << 4)).to(torch.int8)

        bound = matmul_int4_block_expr.bind((A, B_packed))
        (config,) = matmul_int4_block_expr.configs
        code = bound.to_triton_code(config)
        # TODO(oulgen): needs mindot size mocked
        # self.assertExpectedJournal(code)

        match = re.search(r"(?m)^def matmul_int4_block_expr\(", code)
        assert match is not None
        device_code, host_code = code[: match.start()], code[match.start() :]
        self.assertIn("_BLOCK_SIZE_0 = 1", host_code)
        self.assertIn("2 * _BLOCK_SIZE_0, ", host_code)
        self.assertIn("[2 * _BLOCK_SIZE_0, ", device_code)


if __name__ == "__main__":
    unittest.main()
