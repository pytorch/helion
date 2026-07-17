from __future__ import annotations

import re
import unittest

import torch

import helion
from helion import exc
from helion._compat import use_tileir_tunables
from helion._compiler.compile_environment import FixedBlockSizeSource
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfMTIA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.settings import _get_backend


@onlyBackends(["triton", "cute"])
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

    @skipIfRefEager("Triton codegen does not work in ref eager mode")
    def test_to_triton_code_dedupes_future_import(self) -> None:
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([128], device=DEVICE)
        bound = fn.bind((x,))
        code = bound.to_triton_code(bound.config_spec.default_config())

        self.assertEqual(code.count("from __future__ import annotations"), 1)
        self.assertTrue(code.startswith("from __future__ import annotations\n\n"))

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

        # Test "mul" mode
        code, result = code_and_output(fn, (x, "mul"))
        torch.testing.assert_close(result, x * 2.0)

        # Test default mode
        code, result = code_and_output(fn, (x, "default"))
        torch.testing.assert_close(result, x)

    def test_constexpr_in_body_selects_branch(self):
        """`hl.constexpr(cond)` used inside a kernel body must decide the branch
        by the wrapped value, not unconditionally take the `if` branch."""

        @helion.kernel()
        def fn(x: torch.Tensor, flag: bool) -> torch.Tensor:
            take_if = hl.constexpr(flag)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                if take_if:
                    out[tile] = x[tile] + 1.0
                else:
                    out[tile] = x[tile] + 2.0
            return out

        x = torch.zeros([64], device=DEVICE)
        _, result_true = code_and_output(fn, (x, True), block_sizes=[32])
        torch.testing.assert_close(result_true, x + 1.0)
        _, result_false = code_and_output(fn, (x, False), block_sizes=[32])
        torch.testing.assert_close(result_false, x + 2.0)

    @skipIfRefEager("Triton codegen does not work in ref eager mode")
    @skipIfMTIA('Not supported on MTIA. Error: "Expected IntList but got GenericList"')
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
                range_flattens=[True, True] if not use_tileir_tunables() else [],
                range_multi_buffers=[None, False] if not use_tileir_tunables() else [],
                range_num_stages=[3, 1] if not use_tileir_tunables() else [],
                range_unroll_factors=[1, 4] if not use_tileir_tunables() else [],
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

        match = re.search(r"(?m)^def matmul_int4_block_expr\(", code)
        assert match is not None
        device_code, host_code = code[: match.start()], code[match.start() :]
        if _get_backend() == "cute":
            self.assertIn("_default_cute_launcher", host_code)
            self.assertIn("block=(16, 1, 1)", host_code)
            self.assertNotIn("_BLOCK_SIZE_", host_code)
        else:
            self.assertIn("_BLOCK_SIZE_0 = 1", host_code)
            self.assertRegex(host_code, r"2 \* _BLOCK_SIZE_\d+, ")
            self.assertIn("[_SHAPE_DIM, _BLOCK_SIZE_2])", device_code)

    @skipIfRefEager("metadata-only bind inspection does not exercise run_ref")
    def test_symbolic_tile_block_size_reuses_registered_block_id(self) -> None:
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            _m, n = x.shape
            shared = hl.register_block_size(n)
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.shape):
                for tile_k in hl.tile(n, block_size=shared):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + tile_k.block_size
            return out

        x = torch.randn(8, 16, device=DEVICE)
        bound = fn.bind((x,))
        symbolic_fixed_sources = [
            info
            for info in bound.env.block_sizes
            if isinstance(info.block_size_source, FixedBlockSizeSource)
            and isinstance(info.block_size_source.value, torch.SymInt)
        ]
        self.assertEqual(symbolic_fixed_sources, [])

    @skipIfRefEager("compile_config not supported in ref eager mode")
    @skipIfMTIA("Not supported on MTIA. PE failure crashes on DMA_IN")
    def test_constexpr_branch_indexing_config_reuse(self):
        """Reusing the same Config across constexpr variants must not carry
        a stale indexing list from a previous compilation (issue #1501)."""

        @helion.kernel()
        def fn(x: torch.Tensor, w: torch.Tensor, flag: hl.constexpr) -> torch.Tensor:
            (N,) = x.shape
            (W,) = w.shape
            W = hl.specialize(W)
            out = torch.empty_like(x)
            for tile in hl.tile(N):
                acc = hl.zeros([tile], dtype=torch.float32)
                for j in hl.static_range(W):
                    v = hl.load(x, [tile.index + j], extra_mask=tile.index + j < N).to(
                        torch.float32
                    )
                    if flag:
                        tmp = hl.zeros([tile], dtype=torch.float32)
                        for k in hl.static_range(W):
                            tmp += hl.load(
                                x,
                                [tile.index + k],
                                extra_mask=tile.index + k < N,
                            ).to(torch.float32)
                        v = v * tmp
                    acc += w[j].to(torch.float32) * v
                out[tile] = acc.to(out.dtype)
            return out

        N, W = 512, 4
        x = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(W, device=DEVICE, dtype=torch.bfloat16)
        config = helion.Config(block_sizes=[64], num_warps=4)

        # Compile with flag=False first (fewer loads), then flag=True (more loads)
        for flag in [False, True]:
            bound = fn.bind((x, w, hl.constexpr(flag)))
            compiled = bound.compile_config(config)
            result = compiled(x, w, hl.constexpr(flag))
            self.assertEqual(result.shape, x.shape)

    @skipIfRefEager("compile_config not supported in ref eager mode")
    @skipIfMTIA("Not supported on MTIA. PE failure crashes on DMA_IN")
    @skipIfCute("cute backend does not support ConstExpr launcher scalar arguments")
    def test_block_size_constexpr_branch_selects_per_config(self):
        """A branch whose condition depends on a block size must pick the branch
        per-config, not freeze one branch during the single frontend pass
        (issue #3044)."""

        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.shape
            bs = hl.register_block_size(n)
            flag = hl.constexpr(bs < 64)
            out = torch.empty_like(x)
            for tile in hl.tile(n, block_size=bs):
                if flag:
                    out[tile] = x[tile] + 1
                else:
                    out[tile] = x[tile] + 2
            return out

        x = torch.zeros(128, device=DEVICE)
        bound = fn.bind((x,))
        for block_size in [32, 128]:
            result = bound.compile_config(helion.Config(block_sizes=[block_size]))(x)
            expected = x + (1 if block_size < 64 else 2)
            torch.testing.assert_close(result, expected)

    @skipIfMTIA("Not supported on MTIA. PE failure crashes on DMA_IN")
    @skipIfCute("cute backend does not support mixed tcgen05 matmul collective plans")
    def test_block_size_constexpr_branch_divergent_shapes(self):
        """Branches selected by a block-size constexpr may use variables with
        different shapes -- e.g. a swap-AB GEMM that transposes its accumulator
        for small M (issue #3044).

        The branches must be fully duplicated: each branch owns its accumulator
        and store, so no variable is *merged* across the branches (a merge would
        require one consistent shape).  The block-size condition is resolved
        per-config at codegen, so only the live branch is emitted.  A shared K
        block size keeps the inner ``hl.tile(K)`` loops on one block id (an
        un-shared loop would create distinct tile ids that then conflict when the
        branch scopes merge).
        """

        @helion.kernel(static_shapes=True)
        def matmul_swap_ab(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            M, K = a.size()
            _, N = b.size()
            out = torch.empty([M, N], dtype=a.dtype, device=a.device)
            block_m = hl.register_block_size(M)
            block_n = hl.register_block_size(N)
            block_k = hl.register_block_size(K)
            swap_ab = hl.constexpr(block_m < 64 and block_n >= 64)
            for tile_m, tile_n in hl.tile([M, N], block_size=[block_m, block_n]):
                if swap_ab:
                    acc_swap = hl.zeros([tile_n, tile_m], dtype=torch.float32)
                    for tile_k in hl.tile(K, block_size=block_k):
                        a_blk = hl.load(
                            a, [tile_m.index[None, :], tile_k.index[:, None]]
                        )
                        b_blk = hl.load(
                            b, [tile_k.index[None, :], tile_n.index[:, None]]
                        )
                        acc_swap = hl.dot(
                            b_blk, a_blk, acc=acc_swap, out_dtype=torch.float32
                        )
                    out[tile_m, tile_n] = acc_swap.t().to(out.dtype)
                else:
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(K, block_size=block_k):
                        acc = hl.dot(
                            a[tile_m, tile_k],
                            b[tile_k, tile_n],
                            acc=acc,
                            out_dtype=torch.float32,
                        )
                    out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        M, K, N = 64, 512, 256
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        expected = a @ b

        if self._in_ref_eager_mode:
            # Ref-eager runs the body directly with no config/codegen (block
            # sizes are the full dims), so just check the result is correct.
            torch.testing.assert_close(
                matmul_swap_ab(a, b), expected, rtol=1e-2, atol=1e-2
            )
            return

        bound = matmul_swap_ab.bind((a, b))
        # Both swap (block_m < 64 <= block_n) and non-swap configs must compile
        # and produce correct results.  The swap branches are mathematically
        # equivalent (A @ B == (B.T @ A.T).T), so also assert on the generated
        # code that the condition folds to the correct branch per config: the
        # taken branch becomes `if True:` and the dead one `if False:`.
        for bm, bn in [(32, 128), (64, 64), (128, 32)]:
            config = helion.Config(block_sizes=[bm, bn, 64])
            code = bound.to_triton_code(config)
            swap = bm < 64 <= bn
            self.assertEqual(code.count("if True:"), 1 if swap else 0)
            self.assertEqual(code.count("if False:"), 0 if swap else 1)
            result = bound.compile_config(config)(a, b)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("compile_config not supported in ref eager mode")
    @skipIfMTIA("Not supported on MTIA. PE failure crashes on DMA_IN")
    def test_block_size_constexpr_branch_merged_divergent_shapes_rejected(self):
        """Merging a variable with different shapes across the branches is still
        rejected even when the condition is a block-size constexpr: the branches
        must be duplicated instead (see
        test_block_size_constexpr_branch_divergent_shapes)."""

        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            n, m = x.shape
            bs_n = hl.register_block_size(n)
            bs_m = hl.register_block_size(m)
            swap = hl.constexpr(bs_n < 64 and bs_m >= 64)
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile([n, m], block_size=[bs_n, bs_m]):
                # `acc` is written in both branches with different shapes and then
                # used after the `if` -> a control-flow merge with divergent shape.
                if swap:
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                else:
                    acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
                out[tile_n, tile_m] = x[tile_n, tile_m] + acc.sum()
            return out

        with self.assertRaises(exc.ControlFlowTensorMismatch):
            fn.bind((torch.zeros([64, 128], device=DEVICE),)).to_triton_code(
                helion.Config(block_sizes=[32, 128])
            )

    @skipIfRefEager("evaluate_constexpr_condition is a codegen-time helper")
    @skipIfMTIA("Not supported on MTIA. PE failure crashes on DMA_IN")
    def test_evaluate_constexpr_condition(self):
        """DeviceFunction.evaluate_constexpr_condition resolves a block-size test
        to a concrete bool for the current config, and returns None when the test
        is not a per-config compile-time constant."""
        from helion._compiler.generate_ast import GenerateAST

        @helion.kernel(static_shapes=True)
        def k(x: torch.Tensor) -> torch.Tensor:
            n, m = x.shape
            block_m = hl.register_block_size(n)
            block_n = hl.register_block_size(m)
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([n, m], block_size=[block_m, block_n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        bound = k.bind((torch.zeros([128, 128], device=DEVICE),))

        def evaluate(config, make_test):
            with bound.env, bound.host_function:
                device_function = GenerateAST(
                    bound.host_function, config
                ).device_function
                block_m = bound.env.block_sizes[0].var
                block_n = bound.env.block_sizes[1].var
                runtime = bound.env.create_unbacked_symint()
                return device_function.evaluate_constexpr_condition(
                    make_test(block_m, block_n, runtime)
                )

        swap = helion.Config(block_sizes=[32, 128])  # block_m < 64 <= block_n
        non_swap = helion.Config(block_sizes=[128, 32])

        # A block-size condition resolves to the concrete bool for the config.
        self.assertIs(evaluate(swap, lambda bm, bn, r: (bm < 64) & (bn >= 64)), True)
        self.assertIs(
            evaluate(non_swap, lambda bm, bn, r: (bm < 64) & (bn >= 64)), False
        )
        self.assertIs(evaluate(swap, lambda bm, bn, r: bm < 64), True)
        self.assertIs(evaluate(non_swap, lambda bm, bn, r: bm < 64), False)

        # Plain bool/int literals pass through.
        self.assertIs(evaluate(swap, lambda bm, bn, r: True), True)
        self.assertIs(evaluate(swap, lambda bm, bn, r: False), False)
        self.assertIs(evaluate(swap, lambda bm, bn, r: 1), True)

        # Non-constant tests return None: runtime symbols, a mix of block-size and
        # runtime symbols, and non-bool objects.
        self.assertIsNone(evaluate(swap, lambda bm, bn, r: r < 5))
        self.assertIsNone(evaluate(swap, lambda bm, bn, r: (bm < 64) & (r < 5)))
        self.assertIsNone(evaluate(swap, lambda bm, bn, r: "not-a-bool"))


if __name__ == "__main__":
    unittest.main()
