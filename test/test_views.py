from __future__ import annotations

import ast
import itertools
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compat import use_tileir_tunables
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfPallas
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessChunkLowering
from helion._testing import skipUnlessTensorDescriptor
from helion._testing import xfailIfPallas
from helion._testing import xfailIfPallasInterpret
import helion.language as hl
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from helion._compiler.aten_lowering import LoweringContext


@onlyBackends(["triton", "pallas", "cute"])
class TestViews(RefEagerTestBase, TestCase):
    def test_specialize_reshape(self):
        @helion.kernel()
        def fn(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
            batch, seqlen = x.shape
            chunk_size = hl.specialize(chunk_size)
            nchunks = (seqlen + chunk_size - 1) // chunk_size
            reshaped = x.reshape(batch, nchunks, chunk_size)
            out = torch.empty_like(reshaped)
            for tile in hl.tile(reshaped.size()):
                out[tile] = reshaped[tile] + 1
            return out.reshape(batch, seqlen)

        chunk_size = 32
        x = torch.randn(2, chunk_size * 3, device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, chunk_size),
            block_sizes=[1, 1, 32],
        )
        torch.testing.assert_close(result, x + 1)

    def test_softmax_unsqueeze(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).unsqueeze(1)
                exp = torch.exp(values - amax)
                sum_exp = torch.unsqueeze(torch.sum(exp, dim=1), -1)
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=DEVICE, dtype=HALF_DTYPE)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )

    def test_softmax_view_reshape(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).view(tile_n, 1)
                exp = torch.exp(values - amax)
                sum_exp = torch.reshape(torch.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=DEVICE, dtype=HALF_DTYPE)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_squeeze(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_m, :].squeeze(
                    1
                ).unsqueeze(0)
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1][:, 0].unsqueeze(0))

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_transpose(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_m, :].transpose(0, 1)
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1].transpose(0, 1))

    @skipIfPallas("blockwise transpose stores are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_blockwise_transpose_store(self):
        # The slot binds the transposed tile's dims to the other tile: with
        # equal block sizes each 16x16 tile is transposed in place (positional
        # tile semantics), which on cute needs an exchange between threads.
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n].T
            return out

        x = torch.randn([64, 48], device=DEVICE)
        _, result = code_and_output(fn, (x,), block_sizes=[16, 16])
        expected = torch.empty_like(x)
        for i in range(0, 64, 16):
            for j in range(0, 48, 16):
                expected[i : i + 16, j : j + 16] = x[i : i + 16, j : j + 16].T
        torch.testing.assert_close(result, expected)

    @skipIfPallas("lower-rank store values are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_lower_rank_store_value(self):
        # hl.store receives its value unexpanded, so a rank-1 b[tile_m] is
        # right-aligned to the tile_n axis: column j of each tile carries
        # b[m0 + j] on every backend, and unequal extents are a ShapeMismatch.
        @helion.kernel(static_shapes=True)
        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile_m, tile_n in hl.tile(a.size()):
                hl.store(out, [tile_m, tile_n], b[tile_m])
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64], device=DEVICE)
        _, result = code_and_output(fn, (a, b), block_sizes=[16, 16])
        expected = torch.empty_like(a)
        for m0 in range(0, 64, 16):
            for n0 in range(0, 64, 16):
                expected[m0 : m0 + 16, n0 : n0 + 16] = b[m0 : m0 + 16][None, :]
        torch.testing.assert_close(result, expected)
        with self.assertRaises(helion.exc.ShapeMismatch):
            code_and_output(fn, (a, b), block_sizes=[16, 32])

    @skipIfPallas("slice stores of transposed slices are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_slice_store_of_transposed_slice(self):
        # ``out[tile_m, :] = x[:, tile_m]`` is positional: with one 32-tile it
        # is the identity on every backend (cute exchanges the value through
        # shared memory, reading the slice at the load's reduction block).
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :] = x[:, tile_m]
            return out

        @helion.kernel(static_shapes=True)
        def fn_b(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[:, tile_m] = x[tile_m, :]
            return out

        x = torch.randn([32, 32], device=DEVICE)
        for kernel in (fn, fn_b):
            _, result = code_and_output(kernel, (x,), block_sizes=[32])
            torch.testing.assert_close(result, x)

    @skipIfPallas("slice stores beside a tile are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_two_slices_beside_a_tile(self):
        # ``out[tile_m, :, :] = x[tile_m, :, tile_n]`` with the second slice
        # as wide as tile_n is the identity on every backend.
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            M, A, B = x.shape
            out = torch.empty([M, A, B], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([M, B]):
                out[tile_m, :, :] = x[tile_m, :, tile_n]
            return out

        x = torch.randn([4, 8, 8], device=DEVICE)
        _, result = code_and_output(fn, (x,), block_sizes=[1, 8])
        torch.testing.assert_close(result, x)

    @skipIfPallas("positional load masks are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_load_extra_mask_is_positional(self):
        # Triton ANDs extra_mask positionally into the index masks: m.T masks
        # with the blockwise transpose of m; cute exchanges the mask between
        # threads through shared memory.
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile(x.size()):
                out[tm, tn] = hl.load(x, [tm, tn], extra_mask=m[tm, tn].T)
            return out

        x = torch.randn([64, 64], device=DEVICE)
        m = torch.rand([64, 64], device=DEVICE) > 0.5
        _, result = code_and_output(fn, (x, m), block_sizes=[16, 16])
        blockwise = torch.empty_like(m)
        for i in range(0, 64, 16):
            for j in range(0, 64, 16):
                blockwise[i : i + 16, j : j + 16] = m[i : i + 16, j : j + 16].T
        torch.testing.assert_close(
            result, torch.where(blockwise, x, torch.zeros_like(x))
        )

    @skipIfPallas("lower-rank where operands are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_where_lower_rank_operand(self):
        # tl.where receives its operands unexpanded, so a rank-1 row[tm] is
        # right-aligned to the tn axis: Triton reads row[m0 + j] and raises
        # ShapeMismatch for unequal block sizes; cute exchanges the operand
        # between threads through shared memory.
        @helion.kernel(static_shapes=True)
        def fn(c: torch.Tensor, x: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile(x.size()):
                out[tm, tn] = torch.where(c[tm, tn], x[tm, tn], row[tm])
            return out

        c = torch.rand([64, 64], device=DEVICE) > 0.5
        x = torch.randn([64, 64], device=DEVICE)
        row = torch.randn([64], device=DEVICE)
        _, result = code_and_output(fn, (c, x, row), block_sizes=[16, 16])
        positional = torch.empty_like(x)
        for m0 in range(0, 64, 16):
            for n0 in range(0, 64, 16):
                positional[m0 : m0 + 16, n0 : n0 + 16] = row[m0 : m0 + 16][None, :]
        torch.testing.assert_close(result, torch.where(c, x, positional))
        with self.assertRaises(helion.exc.ShapeMismatch):
            code_and_output(fn, (c, x, row), block_sizes=[16, 32])

    @skipIfPallas("reordered lower-rank operands are not verified on Pallas")
    @skipIfRefEager("ref eager runs the whole tensor as one tile")
    def test_reordered_lower_rank_operand(self):
        # The implicit broadcast only inserts None, so b[t1, t0]'s tile dims
        # meet the [t0, t1, t2] result positionally: Triton adds
        # b[t1_0 + i, t0_0 + j]; cute exchanges b between threads (it holds b
        # at its block-id coordinates).
        @helion.kernel(static_shapes=True)
        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for t0, t1, t2 in hl.tile(a.size()):
                out[t0, t1, t2] = a[t0, t1, t2] + b[t1, t0]
            return out

        a = torch.randn([16, 16, 8], device=DEVICE)
        b = torch.randn([16, 16], device=DEVICE)
        _, result = code_and_output(fn, (a, b), block_sizes=[8, 8, 8])
        expected = torch.empty_like(a)
        for t0_0 in range(0, 16, 8):
            for t1_0 in range(0, 16, 8):
                tile = b[t1_0 : t1_0 + 8, t0_0 : t0_0 + 8]
                expected[t0_0 : t0_0 + 8, t1_0 : t1_0 + 8] = (
                    a[t0_0 : t0_0 + 8, t1_0 : t1_0 + 8] + tile[:, :, None]
                )
        torch.testing.assert_close(result, expected)

    @skipIfPallas("blockwise transpose stores are not verified on Pallas")
    def test_single_tile_transpose_store(self):
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n].T
            return out

        x = torch.randn([16, 16], device=DEVICE)
        _, result = code_and_output(fn, (x,), block_sizes=[16, 16])
        torch.testing.assert_close(result, x.T)

    def test_transpose_T_unsqueeze(self):
        @helion.kernel(autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                tile3d = x[tile_n, tile_m].T.unsqueeze(0)
                out[tile_n, tile_m] = tile3d.squeeze(0).T
            return out

        args = (torch.randn([512, 384], device=DEVICE),)
        _, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0])

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_expand(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_n, :].expand(
                    tile_n, tile_m
                )
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_expand_as(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m].expand_as(a)
                out[tile_n, tile_m] = a + b
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_expand_slicing(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "pointer"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m]
                out[tile_n, tile_m] = a + b[None, :]
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_implicit(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "pointer"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m]
                out[tile_n, tile_m] = a + b
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_split_join_roundtrip(self):
        @helion.kernel(config={"block_size": 64})
        def fn(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                lo, hi = hl.split(x[tile, :])
                out[tile, :] = hl.join(hi, lo)
            return out

        x = torch.randn([256, 2], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        expected = torch.stack((x[:, 1], x[:, 0]), dim=-1)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.split", code)
            self.assertIn("tl.join", code)

    @skipUnlessChunkLowering
    def test_torch_chunk_two(self):
        @helion.kernel(autotune_effort="none")
        def fn(
            x: torch.Tensor, use_method: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n, d = x.shape
            d = hl.specialize(d)
            lo = x.new_empty((n, d // 2))
            hi = torch.empty_like(lo)
            for tile in hl.tile(n):
                # The split extent comes from a full-slice reduction block.
                values = x[tile, :]
                if use_method:
                    a, b = values.chunk(2, dim=-1)
                else:
                    a, b = torch.chunk(input=values, chunks=2, dim=1)
                lo[tile, :] = a
                hi[tile, :] = b
            return lo, hi

        for d in (64, 128):
            x = torch.arange(65 * d, device=DEVICE, dtype=torch.float32).reshape(65, d)
            expected = torch.chunk(x, 2, dim=-1)
            for use_method in (False, True):
                with self.subTest(d=d, use_method=use_method):
                    if not self._in_ref_eager_mode:
                        bound = fn.bind((x, use_method))
                        # The specialized full axis must stay persistent.
                        self.assertEqual(
                            bound.env.config_spec.reduction_loops.valid_block_ids(), []
                        )
                        with self.assertRaisesRegex(
                            helion.exc.InvalidConfig, "Too many values.*reduction_loops"
                        ):
                            bound.compile_config(
                                helion.Config(block_sizes=[32], reduction_loops=[16])
                            )
                    code, result = code_and_output(
                        fn, (x, use_method), block_sizes=[32]
                    )
                    torch.testing.assert_close(result, expected)
                    if _get_backend() == "triton":
                        self.assertIn("tl.split", code)
                    else:
                        # Both halves are re-read from the loaded tile.
                        self.assertIn("split_lo", code)
                        self.assertNotIn("split_smem", code)

    @skipUnlessChunkLowering
    def test_torch_unbind_full_slice(self):
        @helion.kernel(autotune_effort="none")
        def fn(
            x: torch.Tensor, use_method: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n, d = x.shape
            d = hl.specialize(d)
            lo = x.new_empty((n,))
            hi = torch.empty_like(lo)
            for tile in hl.tile(n):
                # Keep the full-slice block symbol until unbind specializes it.
                values = x[tile, :]
                if use_method:
                    unbind = values.unbind
                    a, b = unbind(dim=-1)
                else:
                    a, b = torch.unbind(values, dim=1)
                lo[tile] = a
                hi[tile] = b
            return lo, hi

        x = torch.arange(65 * 2, device=DEVICE, dtype=torch.float32).reshape(65, 2)
        for use_method in (False, True):
            with self.subTest(use_method=use_method):
                _code, result = code_and_output(fn, (x, use_method), block_sizes=[32])
                torch.testing.assert_close(result, torch.unbind(x, dim=1))

    @skipUnlessChunkLowering
    def test_torch_unbind_two(self):
        @helion.kernel(autotune_effort="none")
        def fn(
            x: torch.Tensor, use_method: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n, d = x.shape
            d = hl.specialize(d)
            lo = x.new_empty((n, d // 2))
            hi = torch.empty_like(lo)
            for tile in hl.tile(n):
                values = x[tile, :].reshape(tile, 2, d // 2)
                if use_method:
                    permuted = values.permute(0, 2, 1)
                    # Exercise a saved bound method as well as a direct call.
                    unbind = permuted.unbind
                    a, b = unbind(dim=-1)
                else:
                    a, b = torch.unbind(input=values, dim=1)
                lo[tile, :] = a
                hi[tile, :] = b
            return lo, hi

        for d in (64, 128):
            x = torch.arange(65 * d, device=DEVICE, dtype=torch.float32).reshape(65, d)
            expected = torch.unbind(x.reshape(65, 2, d // 2), dim=1)
            for use_method in (False, True):
                with self.subTest(d=d, use_method=use_method):
                    if not self._in_ref_eager_mode:
                        bound = fn.bind((x, use_method))
                        # Reshaping the full slice must not permit partial loads.
                        self.assertEqual(
                            bound.env.config_spec.reduction_loops.valid_block_ids(), []
                        )
                        with self.assertRaisesRegex(
                            helion.exc.InvalidConfig, "Too many values.*reduction_loops"
                        ):
                            bound.compile_config(
                                helion.Config(block_sizes=[32], reduction_loops=[16])
                            )
                    _code, result = code_and_output(
                        fn, (x, use_method), block_sizes=[32]
                    )
                    torch.testing.assert_close(result, expected)

    @skipUnlessChunkLowering
    @skipIfCute(
        "CuTe refuses this (BackendUnsupported): the halves of the computed "
        "64-wide accumulator are held by different synthetic lane iterations "
        "of one thread, out of reach of the hl.split exchange, and the "
        "torch.stack(...).reshape back has a stacked dim no lane owns"
    )
    def test_torch_chunk_unbind_accumulator(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def fn(x: torch.Tensor, use_unbind: hl.constexpr) -> torch.Tensor:
            n, d = x.shape
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                acc = hl.zeros([tile, d])
                for _step in hl.tile(2, block_size=1):
                    acc = acc + x[tile, :]
                    if use_unbind:
                        grouped = acc.reshape(tile, 2, d // 2).permute(0, 2, 1)
                        a, b = grouped.unbind(dim=-1)
                    else:
                        a, b = torch.chunk(acc, 2, dim=-1)
                    acc = torch.stack((a * 2, b * 3), dim=-2).reshape(tile, d)
                out[tile, :] = acc
            return out

        x = torch.arange(65 * 64, device=DEVICE, dtype=torch.float32).reshape(65, 64)
        left, right = torch.chunk(x, 2, dim=-1)
        expected = torch.cat((left * 6, right * 12), dim=-1)
        for use_unbind in (False, True):
            with self.subTest(use_unbind=use_unbind):
                _code, result = code_and_output(fn, (x, use_unbind), block_sizes=[32])
                torch.testing.assert_close(result, expected)

    @skipUnlessChunkLowering
    def test_torch_chunk_unbind_axes(self):
        @helion.kernel(autotune_effort="none")
        def fn(
            x: torch.Tensor, leading_axis: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n, d, k = x.shape
            d, k = hl.specialize((d, k))
            lo = x.new_empty((n, d // 2, k))
            hi = torch.empty_like(lo)
            for tile in hl.tile(n):
                values = x[tile, :, :].reshape(tile, d, k)
                if leading_axis:
                    # Default dim=0 and the unbound Tensor method form.
                    transposed = values.permute(1, 0, 2)
                    left, right = torch.Tensor.chunk(transposed, 2)
                    a = left.permute(1, 0, 2)
                    b = right.permute(1, 0, 2)
                else:
                    a, b = values.chunk(2, dim=-2)
                grouped = torch.stack((a, b), dim=0)
                c, e = grouped.unbind()
                lo[tile, :, :] = c
                hi[tile, :, :] = e
            return lo, hi

        x = torch.arange(65 * 8 * 4, device=DEVICE, dtype=torch.float32).reshape(
            65, 8, 4
        )
        expected = torch.chunk(x, 2, dim=1)
        for leading_axis in (False, True):
            with self.subTest(leading_axis=leading_axis):
                _code, result = code_and_output(fn, (x, leading_axis), block_sizes=[32])
                torch.testing.assert_close(result, expected)

    @skipUnlessChunkLowering
    def test_torch_unbind_stack_flattened_tiles(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n, _ = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                left, right = torch.unbind(x[tile_m, tile_n, :], dim=-1)
                out[tile_m, tile_n, :] = torch.stack((right, left), dim=-1)
            return out

        x = torch.arange(4 * 8 * 2, device=DEVICE, dtype=torch.float32).reshape(4, 8, 2)
        _code, result = code_and_output(
            fn, (x,), block_sizes=[2, 8], flatten_loops=[True]
        )
        torch.testing.assert_close(result, x.flip(-1))

    @onlyBackends(["triton", "cute"])
    def test_torch_stack_flattened_tiles_dim_zero(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                values = x[tile_m, tile_n]
                out[:, tile_m, tile_n] = torch.stack((values, values + 1), dim=0)
            return out

        x = torch.arange(4 * 8, device=DEVICE, dtype=torch.float32).reshape(4, 8)
        _code, result = code_and_output(
            fn, (x,), block_sizes=[2, 8], flatten_loops=[True]
        )
        torch.testing.assert_close(result, torch.stack((x, x + 1), dim=0))

    @onlyBackends(["cute"])
    def test_stack_and_reshape_flattened_tiles(self):
        """Stacks of flattened multi-block tiles that do not align with rows.

        Triton offers no flatten_loops for these kernels; on CuTe each block's
        tile-local coordinate is its digit of the flat position in the tile.
        """

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def stack_dim1(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, 2, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                values = x[tile_m, tile_n]
                out[tile_m, :, tile_n] = torch.stack((values, values + 1), dim=1)
            return out

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def stack_reshape(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((2, m, n), device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                values = x[tile_m, tile_n]
                pair = torch.stack((values, values * 3), dim=0)
                out[:, tile_m, tile_n] = pair.reshape(2, -1).reshape(2, tile_m, tile_n)
            return out

        x = torch.arange(6 * 12, device=DEVICE, dtype=torch.float32).reshape(6, 12)
        cases = [
            (stack_dim1, torch.stack((x, x + 1), dim=1)),
            (stack_reshape, torch.stack((x, x * 3), dim=0)),
        ]
        for fn, expected in cases:
            for block_sizes in ([2, 8], [4, 4], [1, 16]):
                with self.subTest(fn=fn.fn.__name__, block_sizes=block_sizes):
                    _code, result = code_and_output(
                        fn, (x,), block_sizes=block_sizes, flatten_loops=[True]
                    )
                    torch.testing.assert_close(result, expected)

    @skipUnlessChunkLowering
    @skipIfCute(
        "CuTe refuses this (BackendUnsupported): the halves of the computed "
        "64-wide accumulator are held by different synthetic lane iterations "
        "of one thread, out of reach of the hl.split exchange, and the "
        "torch.stack(...).reshape back has a stacked dim no lane owns"
    )
    def test_torch_chunk_unbind_dot_accumulator(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def fn(
            x: torch.Tensor, weight: torch.Tensor, use_unbind: hl.constexpr
        ) -> torch.Tensor:
            n, k = x.shape
            d = weight.size(1)
            out = x.new_empty((n, d), dtype=torch.float32)
            for tile in hl.tile(n):
                acc = hl.zeros([tile, d])
                for tile_k in hl.tile(k, block_size=16):
                    acc = hl.dot(x[tile, tile_k], weight[tile_k, :], acc=acc)
                    if use_unbind:
                        grouped = acc.reshape(tile, 2, d // 2).permute(0, 2, 1)
                        a, b = grouped.unbind(dim=-1)
                    else:
                        a, b = torch.chunk(acc, 2, dim=-1)
                    acc = torch.stack((a * 2, b * 3), dim=-2).reshape(tile, d)
                out[tile, :] = acc
            return out

        x = torch.randn((65, 32), device=DEVICE, dtype=torch.float16)
        weight = torch.randn((32, 64), device=DEVICE, dtype=torch.float16)
        scale = torch.tensor([2.0] * 32 + [3.0] * 32, device=DEVICE)
        first = x[:, :16].float() @ weight[:16, :].float()
        second = x[:, 16:].float() @ weight[16:, :].float()
        expected = (first * scale + second) * scale
        for use_unbind in (False, True):
            with self.subTest(use_unbind=use_unbind):
                _code, result = code_and_output(
                    fn, (x, weight, use_unbind), block_sizes=[32]
                )
                torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @skipIfPallas("hl.split/hl.join over permuted pair views is not verified on Pallas")
    def test_split_join_halves_permute(self):
        # The rope pattern: a full-slice dim viewed as (2, half), transposed so
        # the pair dim is last, split, recombined and transposed back.
        @helion.kernel(config={"block_size": 64}, static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            n, d = x.size()
            half = d // 2
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                pair = (
                    x[tile, :]
                    .to(torch.float32)
                    .reshape([tile, 2, half])
                    .permute(0, 2, 1)
                )
                lo, hi = hl.split(pair)
                out[tile, :] = (
                    hl.join(hi * 2.0, lo * 3.0)
                    .permute(0, 2, 1)
                    .reshape([tile, d])
                    .to(x.dtype)
                )
            return out

        x = torch.randn([256, 64], device=DEVICE)
        _code, result = code_and_output(fn, (x,))
        expected = torch.cat((x[:, 32:] * 2.0, x[:, :32] * 3.0), dim=-1)
        torch.testing.assert_close(result, expected)

    @skipIfPallas("hl.split/hl.join over permuted pair views is not verified on Pallas")
    def test_split_full_slice_store(self):
        # Split halves stored through full slices of their own extent.
        @helion.kernel(config={"block_size": 64}, static_shapes=True)
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n, d = x.size()
            half = d // 2
            lo_out = torch.empty([n, half], dtype=x.dtype, device=x.device)
            hi_out = torch.empty([n, half], dtype=x.dtype, device=x.device)
            for tile in hl.tile(n):
                pair = x[tile, :].reshape([tile, 2, half]).permute(0, 2, 1)
                lo, hi = hl.split(pair)
                lo_out[tile, :] = lo
                hi_out[tile, :] = hi + 1.0
            return lo_out, hi_out

        x = torch.randn([256, 64], device=DEVICE)
        _code, (lo, hi) = code_and_output(fn, (x,))
        torch.testing.assert_close(lo, x[:, :32])
        torch.testing.assert_close(hi, x[:, 32:] + 1.0)

    @skipIfPallas("hl.split/hl.join over permuted pair views is not verified on Pallas")
    @skipIfRefEager("block-local halves depend on the tile block size")
    def test_split_join_tiled_halves(self):
        # Same pattern on a real tile of d (block-local halves).
        @helion.kernel(config={"block_sizes": [32, 16]}, static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            n, d = x.size()
            out = torch.empty_like(x)
            for tile_n, tile_d in hl.tile([n, d]):
                pair = (
                    x[tile_n, tile_d]
                    .reshape([tile_n, 2, tile_d.block_size // 2])
                    .permute(0, 2, 1)
                )
                lo, hi = hl.split(pair)
                out[tile_n, tile_d] = (
                    hl.join(lo * 2.0, hi * 3.0)
                    .permute(0, 2, 1)
                    .reshape([tile_n, tile_d])
                )
            return out

        x = torch.randn([128, 64], device=DEVICE)
        _code, result = code_and_output(fn, (x,))
        blocks = x.view(128, 4, 2, 8)
        expected = torch.cat(
            (blocks[:, :, :1] * 2.0, blocks[:, :, 1:] * 3.0), dim=2
        ).view(128, 64)
        torch.testing.assert_close(result, expected)

    @skipIfPallas("hl.split/hl.join over pair views is not verified on Pallas")
    @skipIfRefEager("checks the backend lowering of hl.split over a masked load")
    def test_split_join_masked_load(self):
        # The pair tile comes from a load with ``extra_mask``: positions where
        # the mask is false are zero-filled and hl.split must see those zeros
        # for both pair elements, not the raw memory behind them.
        @helion.kernel(config={"block_size": 64}, static_shapes=True)
        def fn(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            n, d = x.size()
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                v = hl.load(x, [tile, slice(None)], extra_mask=mask[tile, :])
                lo, hi = hl.split(v.reshape([tile, d // 2, 2]))
                out[tile, :] = hl.join(hi * 2.0, lo * 3.0).reshape([tile, d])
            return out

        x = torch.randn([256, 64], device=DEVICE)
        mask = torch.rand([256, 64], device=DEVICE) < 0.5
        code, result = code_and_output(fn, (x, mask))
        masked = torch.where(mask, x, torch.zeros_like(x)).view(256, 32, 2)
        expected = torch.stack((masked[..., 1] * 2.0, masked[..., 0] * 3.0), -1)
        torch.testing.assert_close(result, expected.view(256, 64))
        if _get_backend() == "cute":
            # The masked load cannot be re-read element-wise; the tile goes
            # through the verified shared-memory exchange instead.
            self.assertIn("split_smem", code)

    @skipIfPallas("hl.split/hl.join over pair views is not verified on Pallas")
    @skipIfRefEager(
        "an eager view aliases the loaded tile; checks the device load copy"
    )
    def test_split_after_store_through_host_view(self):
        # ``y`` is a host-side view of ``x``; the store through it must not be
        # observed by hl.split of the tile loaded from ``x`` before it.
        @helion.kernel(config={"block_size": 64}, static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            n, d = x.size()
            y = x.view(n, d)
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                v = x[tile, :]
                y[tile, :] = torch.zeros_like(v)
                lo, hi = hl.split(v.reshape([tile, d // 2, 2]))
                out[tile, :] = hl.join(hi, lo).reshape([tile, d])
            return out

        x = torch.randn([256, 64], device=DEVICE)
        pairs = x.view(256, 32, 2)
        expected = torch.stack((pairs[..., 1], pairs[..., 0]), dim=-1).view(256, 64)
        code, result = code_and_output(fn, (x.clone(),))
        torch.testing.assert_close(result, expected)
        if _get_backend() == "cute":
            self.assertIn("split_smem", code)

    @skipIfPallas("hl.split/hl.join over pair views is not verified on Pallas")
    @skipIfRefEager(
        "an eager view aliases the loaded tile; checks the device load copy"
    )
    def test_split_after_store_to_aliased_arg(self):
        # Two kernel arguments may be the same tensor at call time.
        @helion.kernel(config={"block_size": 64}, static_shapes=True)
        def fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            n, d = x.size()
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                v = x[tile, :]
                w[tile, :] = torch.zeros_like(v)
                lo, hi = hl.split(v.reshape([tile, d // 2, 2]))
                out[tile, :] = hl.join(hi, lo).reshape([tile, d])
            return out

        x = torch.randn([256, 64], device=DEVICE)
        pairs = x.view(256, 32, 2)
        expected = torch.stack((pairs[..., 1], pairs[..., 0]), dim=-1).view(256, 64)
        x = x.clone()
        code, result = code_and_output(fn, (x, x))
        torch.testing.assert_close(result, expected)
        if _get_backend() == "cute":
            self.assertIn("split_smem", code)

    def test_join_broadcast_scalar(self):
        @helion.kernel(config={"block_size": 64})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
            for tile in hl.tile(n):
                scalar = hl.load(y, [0])
                out[tile, :] = hl.join(x[tile], scalar)
            return out

        x = torch.randn([128], device=DEVICE)
        y = torch.randn([1], device=DEVICE)
        code, result = code_and_output(fn, (x, y))
        broadcast_y = torch.broadcast_to(y, x.shape)
        expected = torch.stack((x, broadcast_y), dim=-1)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.join", code)

    def test_scalar_broadcast_2d(self):
        """Test that scalars broadcast correctly with 2D tensors."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[2, 64],
                flatten_loops=[True],
                indexing=["pointer", "pointer", "tensor_descriptor"],
            )
        )
        def scalar_multiply(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_idx in hl.tile(out.shape):
                scale_val = hl.load(scale, [0])
                out[tile_idx] = x[tile_idx] * scale_val
            return out

        input_tensor = torch.randn([4, 128], device=DEVICE)
        scale_tensor = torch.tensor([2.0], device=DEVICE)
        result = scalar_multiply(input_tensor, scale_tensor)
        expected = input_tensor * scale_tensor[0]
        torch.testing.assert_close(result, expected)

    @xfailIfPallasInterpret("jax interpret-mode discharge bug on pipeline buffers")
    def test_reshape_input_types(self):
        @helion.kernel(static_shapes=True)
        def reshape_reduction_dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"

            out = torch.zeros(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )

            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

                # Test different reshape input types
                reshaped_acc = acc.reshape(-1, tile_m.block_size * tile_n.block_size)
                reshaped_acc = reshaped_acc.reshape(
                    tile_m.block_size, tile_n.block_size
                )
                reshaped_acc = reshaped_acc.flatten(0)
                reshaped_acc = reshaped_acc.reshape(tile_m, tile_n)
                reshaped_acc = reshaped_acc.reshape(
                    tile_m.block_size * 2 // 2, tile_n.block_size + 1 - 1
                )
                out[tile_m, tile_n] = reshaped_acc

            return out

        x = torch.randn(8, 16, device=DEVICE)
        y = torch.randn(16, 32, device=DEVICE)
        _code, result = code_and_output(reshape_reduction_dim, (x, y))
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_reshape_sum(self):
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0], dtype=x.dtype)
                for tile1, tile2 in hl.tile([x.size(1), x.size(2)]):
                    acc += x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)
                out[tile0] = acc
            return out

        x = torch.randn(3, 4, 5, device=DEVICE)
        code, result = code_and_output(fn, (x,))
        expected = x.sum(dim=(1, 2))
        torch.testing.assert_close(result, expected)

    @skipIfPallas("repeat_interleave reshapes are not verified on Pallas")
    def test_repeat_interleave_reshape_times_tile(self):
        """An expand + merge reshape (repeat_interleave) times a tile of the
        merged layout: every output element reads a source element another
        CuTe thread holds.  The computed-fragment lowering stages the values
        in shared memory; without it the backend refuses rather than
        miscompute."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[32]))
        def fn(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                sf = s[tile_m, :]
                se = sf[:, :, None].expand(tile_m, sf.size(1), 32).reshape(tile_m, k)
                out[tile_m, :] = x[tile_m, :] * se
            return out

        s = torch.randn(64, 4, device=DEVICE)
        x = torch.randn(64, 128, device=DEVICE)
        if _get_backend() == "cute":
            with (
                patch(
                    "helion._compiler.cute.computed_fragment.codegen_computed_fragment_root",
                    return_value=False,
                ),
                self.assertRaisesRegex(
                    helion.exc.BackendUnsupported, "moves elements between threads"
                ),
            ):
                code_and_output(fn, (s, x))
            fn.reset()
        _, result = code_and_output(fn, (s, x))
        torch.testing.assert_close(result, x * s.repeat_interleave(32, dim=1))

    @skipIfPallas("repeat_interleave reshapes are not verified on Pallas")
    def test_repeat_interleave_tile(self):
        """``torch.repeat_interleave`` lowers to unsqueeze / expand / clone /
        reshape, which on CuTe leaves each thread the element at its source
        position.  The store re-reads the element its slot names from the
        load; scaling it first would compute with the wrong element, which
        CuTe refuses."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def scaled(a: torch.Tensor) -> torch.Tensor:
            out = torch.empty([a.size(0), a.size(1) * 2], device=a.device)
            for t in hl.tile(a.size(0)):
                out[t, :] = torch.repeat_interleave(a[t, :], 2, dim=1) * 2
            return out

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def bare(a: torch.Tensor) -> torch.Tensor:
            out = torch.empty([a.size(0), a.size(1) * 2], device=a.device)
            for t in hl.tile(a.size(0)):
                out[t, :] = torch.repeat_interleave(a[t, :], 2, dim=1)
            return out

        a = torch.randn(64, 16, device=DEVICE)
        expected = torch.repeat_interleave(a, 2, dim=1)
        for fn, scale in ((scaled, 2), (bare, 1)):
            with self.subTest(fn=fn.fn.__name__):
                if _get_backend() == "cute" and fn is scaled:
                    with self.assertRaisesRegex(
                        helion.exc.BackendUnsupported, "moves elements between threads"
                    ):
                        code_and_output(fn, (a,))
                    continue
                _, result = code_and_output(fn, (a,))
                torch.testing.assert_close(result, expected * scale)

    @skipIfPallas("consumers of merging reshapes are not verified on Pallas")
    def test_thread_moving_reshape_consumers(self):
        """Consumers of a merging reshape whose elements sit on other CuTe
        threads: a cast before the store, a sum over a static merged dim
        beside a slice of that size and a virtual view mixed with a tile,
        which CuTe refuses, and a deinterleaving store, whose elements the
        store re-reads from the load.  Elsewhere all match eager."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def cast_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0), 128], device=x.device, dtype=torch.float16)
            for t in hl.tile(x.size(0)):
                out[t, :] = x[t, :, :].reshape(t, 128).to(torch.float16)
            return out

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def merged_sum(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(z)
            for t in hl.tile(x.size(0)):
                r = x[t, :, :].reshape(t, 128)
                out[t, :] = z[t, :] * r.sum(-1, keepdim=True)
            return out

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[16]))
        def stacked_view(
            a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            m, n = a.size()
            out = torch.empty_like(z)
            for t in hl.tile(m):
                s = torch.stack((a[t, :], b[t, :]), dim=2)
                r = s.reshape(t, 2 * n).unsqueeze(0)
                out[t, :] = (r * z[t, :]).squeeze(0)
            return out

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def deinterleave(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for t in hl.tile(x.size(0)):
                out[t, :] = x[t, :].reshape(t, 64, 2).permute(0, 2, 1).reshape(t, 128)
            return out

        x3 = torch.randn(64, 4, 32, device=DEVICE)
        x2 = torch.randn(64, 128, device=DEVICE)
        z = torch.randn(64, 128, device=DEVICE)
        a = torch.randn(64, 16, device=DEVICE)
        b = torch.randn(64, 16, device=DEVICE)
        z32 = torch.randn(64, 32, device=DEVICE)
        cases = [
            (cast_store, (x3,), x3.reshape(64, 128).half()),
            (merged_sum, (x3, z), z * x3.reshape(64, 128).sum(-1, keepdim=True)),
            (stacked_view, (a, b, z32), torch.stack((a, b), 2).reshape(64, 32) * z32),
            (
                deinterleave,
                (x2,),
                x2.reshape(64, 64, 2).permute(0, 2, 1).reshape(64, 128),
            ),
        ]
        for fn, args, expected in cases:
            with self.subTest(fn=fn.fn.__name__):
                if _get_backend() == "cute" and fn is not deinterleave:
                    with self.assertRaisesRegex(
                        helion.exc.BackendUnsupported, "moves elements between threads"
                    ):
                        code_and_output(fn, args)
                    continue
                _, result = code_and_output(fn, args)
                torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @skipIfPallas("reshape round trips are not verified on Pallas")
    def test_reshape_round_trip_consumers(self):
        """A merge that a later reshape splits back: the restoring reshape is
        proven to put every element back on its own CuTe thread, so any
        consumer may read it (``x.reshape(-1).reshape(t, n) + y``, an
        epilogue after a round trip of the accumulator)."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[4, 4]))
        def round_trip_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tm, tn in hl.tile([m, n]):
                flat = x[tm, tn].reshape(tm.block_size * tn.block_size)
                out[tm, tn] = flat.reshape(tm, tn) + y[tm, tn]
            return out

        @helion.kernel(
            static_shapes=True, config=helion.Config(block_sizes=[8, 16, 16])
        )
        def round_trip_epilogue(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                r = acc.reshape(-1, tile_m.block_size * tile_n.block_size)
                out[tile_m, tile_n] = r.reshape(tile_m, tile_n) * 2.0 + 1.0
            return out

        x = torch.randn(64, 32, device=DEVICE)
        y = torch.randn(64, 32, device=DEVICE)
        a = torch.randn(32, 16, device=DEVICE)
        b = torch.randn(16, 32, device=DEVICE)
        for fn, args, expected, tol in (
            (round_trip_add, (x, y), x + y, 1e-3),
            # tf32 dots on Triton
            (round_trip_epilogue, (a, b), (a @ b) * 2 + 1, 1e-1),
        ):
            with self.subTest(fn=fn.fn.__name__):
                _, result = code_and_output(fn, args)
                torch.testing.assert_close(result, expected, rtol=tol, atol=tol)

    @xfailIfPallas("torch.stack not supported on pallas")
    def test_stack_power_of_2(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def test_stack_power_of_2_kernel(
            a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            M, N = a.shape
            result = torch.zeros(M * 2, N, dtype=a.dtype, device=a.device)

            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N):
                    a_tile = a[tile_m, tile_n]
                    b_tile = b[tile_m, tile_n]

                    # Stack tensors along dim=1 (creates [BLOCK_M, 2, BLOCK_N])
                    stacked = torch.stack([a_tile, b_tile], dim=1)

                    # Reshape to [BLOCK_M * 2, BLOCK_N]
                    reshaped = stacked.reshape(tile_m.block_size * 2, tile_n.block_size)

                    result[
                        (tile_m.begin * 2) : (tile_m.begin * 2 + tile_m.block_size * 2),
                        tile_n,
                    ] = reshaped

            return result

        M, N = 64, 128
        device = DEVICE

        a = torch.randn(M, N, dtype=torch.float32, device=device)
        b = torch.randn(M, N, dtype=torch.float32, device=device)

        result = test_stack_power_of_2_kernel(a, b)
        expected = torch.zeros(M * 2, N, dtype=torch.float32, device=device)
        expected[0::2] = a  # Every 2nd row starting from 0
        expected[1::2] = b  # Every 2nd row starting from 1
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @xfailIfPallas("torch.stack not supported on pallas")
    def test_stack_non_power_of_2(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def test_stack_non_power_of_2_kernel(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> torch.Tensor:
            M, N = a.shape
            result = torch.zeros(M, 3, N, dtype=a.dtype, device=a.device)

            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N):
                    a_tile = a[tile_m, tile_n]
                    b_tile = b[tile_m, tile_n]
                    c_tile = c[tile_m, tile_n]

                    # Stack tensors along dim=1 (creates [BLOCK_M, 3, BLOCK_N])
                    stacked = torch.stack([a_tile, b_tile, c_tile], dim=1)

                    result[tile_m, :, tile_n] = stacked

            return result

        M, N = 65, 129
        device = DEVICE

        a = torch.randn(M, N, dtype=torch.float32, device=device)
        b = torch.randn(M, N, dtype=torch.float32, device=device)
        c = torch.randn(M, N, dtype=torch.float32, device=device)

        code, result = code_and_output(test_stack_non_power_of_2_kernel, (a, b, c))
        expected = torch.stack([a, b, c], dim=1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @skipIfRefEager("ref eager does not support lifted variable")
    def test_view_blocksize_constexpr(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def foo(x: torch.Tensor) -> torch.Tensor:
            N = x.shape[0]
            N = hl.specialize(N)
            out = x.new_empty(N // 2)
            for (n_tile,) in hl.tile([N]):
                val = x[n_tile]
                val = val.view(n_tile.block_size // 2, 2)
                val_a, val_b = hl.split(val)
                out[n_tile.begin + hl.arange(0, n_tile.block_size // 2)] = val_a + val_b
            return out

        x = torch.randn(1024, dtype=torch.bfloat16, device=DEVICE)
        code, result = code_and_output(foo, (x,))
        self.assertEqual(result.numel(), x.numel() // 2)
        if _get_backend() == "triton":
            self.assertIn("tl.reshape", code)

    @skipIfRefEager("ref eager does not support lifted variable")
    def test_view_blocksize_constexpr_pairsum(self):
        # The split-over-view + compacted-store machinery exercised by
        # ``test_view_blocksize_constexpr`` (which only checks codegen shape)
        # must produce genuine ``x.view(N // 2, 2).sum(-1)`` values.  This
        # variant writes the compacted result to the matching output offset
        # (``n_tile.begin // 2``) so the result is numerically meaningful.
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def foo(x: torch.Tensor) -> torch.Tensor:
            N = x.shape[0]
            N = hl.specialize(N)
            out = x.new_empty(N // 2)
            for (n_tile,) in hl.tile([N]):
                val = x[n_tile]
                val = val.view(n_tile.block_size // 2, 2)
                val_a, val_b = hl.split(val)
                out[n_tile.begin // 2 + hl.arange(0, n_tile.block_size // 2)] = (
                    val_a + val_b
                )
            return out

        x = torch.randn(1024, dtype=torch.float32, device=DEVICE)
        _code, result = code_and_output(foo, (x,))
        expected = x.view(x.numel() // 2, 2).sum(-1)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @xfailIfPallas("torch.stack not supported on pallas")
    def test_stack_dim0(self):
        with torch._inductor.config.patch(
            {"use_static_cuda_launcher": False} if use_tileir_tunables() else {}
        ):

            @helion.kernel(autotune_effort="none", static_shapes=True)
            def test_stack_dim0_kernel(
                a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
            ) -> torch.Tensor:
                M, N = a.shape
                result = torch.zeros(3, M, N, dtype=a.dtype, device=a.device)

                for tile_m in hl.tile(M):
                    for tile_n in hl.tile(N):
                        a_tile = a[tile_m, tile_n]
                        b_tile = b[tile_m, tile_n]
                        c_tile = c[tile_m, tile_n]

                        # Stack 3 tensors along dim=0
                        # This creates [3, BLOCK_M, BLOCK_N]
                        stacked = torch.stack([a_tile, b_tile, c_tile], dim=0)

                        result[:, tile_m, tile_n] = stacked

                return result

            M, N = 65, 129
            device = DEVICE

            a = torch.randn(M, N, dtype=torch.float32, device=device)
            b = torch.randn(M, N, dtype=torch.float32, device=device)
            c = torch.randn(M, N, dtype=torch.float32, device=device)

            code, result = code_and_output(test_stack_dim0_kernel, (a, b, c))
            expected = torch.stack([a, b, c], dim=0)
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

            # Verify torch.compile still decomposes aten.stack to aten.cat
            from torch._inductor import config as inductor_config

            def capture_graph(graph):
                self._graph = str(graph)
                return graph

            with inductor_config.patch(post_grad_custom_pre_pass=capture_graph):
                torch.compile(
                    lambda x, y, z: torch.stack([x, y, z], dim=0),
                    backend="inductor",
                )(
                    torch.randn(4, 4, device=device),
                    torch.randn(4, 4, device=device),
                    torch.randn(4, 4, device=device),
                )
            assert "aten.cat" in self._graph and "aten.stack" not in self._graph

    @skipIfRefEager("ref eager does not support view dtype")
    @xfailIfPallas("view dtype reinterpret not supported on pallas")
    def test_view_dtype_reinterpret(self):
        """Test viewing a tensor with a different dtype (bitcast/reinterpret)."""

        @helion.kernel(static_shapes=True)
        def view_dtype_kernel(x: torch.Tensor) -> torch.Tensor:
            # x is bfloat16, view as int16 to access raw bits
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                val = x[tile]
                # View bf16 as int16, add 1 to raw bits, view back as bf16
                val_as_int = val.view(dtype=torch.int16)
                val_as_int = val_as_int + 1
                val_back = val_as_int.view(dtype=torch.bfloat16)
                out[tile] = val_back
            return out

        x = torch.randn(1024, dtype=torch.bfloat16, device=DEVICE)
        code, result = code_and_output(view_dtype_kernel, (x,))
        # Verify that the operation is a bitcast (add 1 to raw bits)
        expected = (x.view(dtype=torch.int16) + 1).view(dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertTrue(
                ".to(tl.int16)" in code or "tl.cast(" in code,
                "Expected bitcast to int16 via .to() or tl.cast()",
            )

    @onlyBackends(["cute"])
    @skipIfRefEager("checks generated numeric bitcasts")
    def test_numeric_view_dtype_preserves_bits(self):
        @helion.kernel(static_shapes=True)
        def reinterpret(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            out = torch.empty(x.shape, device=x.device, dtype=dtype)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile].view(dtype)
            return out

        for source, target in [
            (torch.float16, torch.int16),
            (torch.bfloat16, torch.int16),
            (torch.float32, torch.int32),
            (torch.float64, torch.int64),
            (torch.int8, torch.uint8),
            (torch.int32, torch.uint32),
            (torch.int64, torch.uint64),
        ]:
            with self.subTest(source=source, target=target):
                x = torch.arange(137, device=DEVICE).to(source)
                if source.is_floating_point:
                    x[:4] = torch.tensor(
                        [float("inf"), -float("inf"), float("nan"), -0.0],
                        device=DEVICE,
                        dtype=source,
                    )
                code, actual = code_and_output(
                    reinterpret, (x, target), block_sizes=[32]
                )
                self.assertNotIn("view_dtype_smem", code)
                self.assertIn(".bitcast(", code)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), x.view(torch.uint8))
                )

    @onlyBackends(["cute"])
    @skipIfRefEager("checks declared dtype boundary")
    def test_view_dtype_after_low_precision_arithmetic(self):
        @helion.kernel(static_shapes=True)
        def reinterpret(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape, device=x.device, dtype=torch.int16)
            for tile in hl.tile(x.numel()):
                values = x[tile] + 0.0078125
                out[tile] = values.view(torch.int16)
            return out

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.linspace(0.5, 2, 137, device=DEVICE, dtype=dtype)
                _, actual = code_and_output(reinterpret, (x,), block_sizes=[32])
                self.assertTrue(torch.equal(actual, (x + 0.0078125).view(torch.int16)))

    @onlyBackends(["cute"])
    @skipIfRefEager("requires replicated physical thread axes")
    def test_view_dtype_reduction_with_independent_output_axis(self):
        @helion.kernel(static_shapes=True)
        def count_bits(x: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape, dtype=torch.int32, device=x.device)
            for row in hl.tile(x.size(0)):
                count = (
                    (x[row, :].view(torch.int32) >= cutoff[row, None])
                    .to(torch.int32)
                    .sum(-1)
                )
                for col in hl.tile(x.size(1)):
                    bits = x[row, col].view(torch.int32)
                    out[row, col] = count[:, None] + (bits & 15)
            return out

        x = torch.rand((17, 64), device=DEVICE)
        cutoff = x[:, 0].contiguous().view(torch.int32)
        code, actual = code_and_output(
            count_bits,
            (x, cutoff),
            block_sizes=[8, 32],
            reduction_loops=[4],
        )
        bits = x.view(torch.int32)
        expected = (bits >= cutoff[:, None]).sum(-1)[:, None] + (bits & 15)
        self.assertNotIn("view_dtype_smem", code)
        torch.testing.assert_close(actual, expected.to(torch.int32), rtol=0, atol=0)


def _execute_triton_stack_lowering(values, dim):
    from helion._compiler.triton.aten_lowering import codegen_stack

    graph = torch.fx.Graph()
    inputs = [graph.placeholder(f"arg{i}") for i in range(len(values))]
    for node, value in zip(inputs, values, strict=True):
        node.meta["val"] = value
    node = graph.call_function(torch.ops.aten.stack.default, (inputs, dim))
    node.meta["val"] = torch.stack(values, dim=dim)
    statements = []
    sequence = itertools.count()
    context = SimpleNamespace(
        env={n: ast.Name(id=f"arg{i}", ctx=ast.Load()) for i, n in enumerate(inputs)},
        cg=SimpleNamespace(
            add_statement=statements.append,
            device_function=SimpleNamespace(
                new_var=lambda name: f"{name}_{next(sequence)}",
                tile_strategy=SimpleNamespace(
                    compact_shape=lambda shape: [
                        SimpleNamespace(user_indices=[axis])
                        for axis in range(len(shape))
                    ]
                ),
            ),
        ),
    )
    expression = codegen_stack(cast("LoweringContext", context), node)
    statements.append(
        ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())],
            value=cast("ast.expr", expression),
        )
    )
    namespace = {
        "tl": SimpleNamespace(
            arange=torch.arange,
            expand_dims=torch.unsqueeze,
            zeros_like=torch.zeros_like,
            where=torch.where,
        ),
        **{f"arg{i}": value for i, value in enumerate(values)},
    }
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(module, "<triton-stack-lowering>", "exec"), namespace)
    return namespace["result"]


class TestTritonStackAxisCPU(unittest.TestCase):
    def test_stack_axes_follow_output_rank(self):
        # Execute the actual emitted selector/expand/where operations. Distinct
        # inputs expose reordered stacking; non-power-of-two lists check padding.
        for shape in ((), (5,), (2, 4), (2, 3, 4)):
            base = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            layouts = (base, base.transpose(-1, -2)) if len(shape) >= 2 else (base,)
            for value in layouts:
                for count in (1, 2, 3, 4):
                    values = [value + 100 * i for i in range(count)]
                    rank = value.ndim + 1
                    for dim in range(-rank, rank):
                        with self.subTest(shape=value.shape, count=count, dim=dim):
                            actual = _execute_triton_stack_lowering(values, dim)
                            axis = dim % rank
                            torch.testing.assert_close(
                                actual.narrow(axis, 0, count),
                                torch.stack(values, dim=dim),
                                rtol=0,
                                atol=0,
                            )
                            padded = 1 << (count - 1).bit_length()
                            self.assertEqual(actual.ndim, rank)
                            self.assertEqual(actual.size(axis), padded)
                            if padded != count:
                                self.assertEqual(
                                    torch.count_nonzero(
                                        actual.narrow(axis, count, padded - count)
                                    ),
                                    0,
                                )


if __name__ == "__main__":
    unittest.main()
