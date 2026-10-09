from __future__ import annotations

import ast
import math
from typing import cast
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion import _compat
from helion import exc
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_block_ptr
from helion._compat import supports_tensor_descriptor
from helion._compat import use_tileir_tunables
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.backend import validate_thread_axis_accesses
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfLowVRAM
from helion._testing import skipIfNormalMode
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessBlockPtr
from helion._testing import skipUnlessTensorDescriptor
from helion._testing import xfailIfCute
from helion._testing import xfailIfPallas
import helion.language as hl

_LARGE_BF16_SHAPE = (51200, 51200)
_LARGE_BF16_REQUIRED_BYTES = (
    8
    * math.prod(_LARGE_BF16_SHAPE)
    * torch.tensor([], dtype=torch.bfloat16).element_size()
)
_LARGE_TENSOR_B = 2**15
_LARGE_TENSOR_D = 2**17
_LARGE_TENSOR_REQUIRED_BYTES = (
    4
    * _LARGE_TENSOR_B
    * _LARGE_TENSOR_D
    * torch.tensor([], dtype=torch.float16).element_size()
)


@helion.kernel
def broadcast_add_3d(
    x: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor
) -> torch.Tensor:
    d0, d1, d2 = x.size()
    out = torch.empty_like(x)
    for tile_l, tile_m, tile_n in hl.tile([d0, d1, d2]):
        # bias1 has shape [1, d1, d2], bias2 has shape [d0, 1, d2]
        out[tile_l, tile_m, tile_n] = (
            x[tile_l, tile_m, tile_n]
            + bias1[tile_l, tile_m, tile_n]
            + bias2[tile_l, tile_m, tile_n]
        )
    return out


@helion.kernel
def reduction_sum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m], device=x.device, dtype=x.dtype)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)

    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _computed_tile_coordinates(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = torch.empty_like(x)
    reused = torch.empty_like(x)
    for row, col in hl.tile(x.shape, block_size=[2, 32]):
        values = (row.index[:, None] * x.size(1) + col.index[None, :]).to(torch.float32)
        prefix[row, col] = hl.cumsum(values, dim=-1)
        reused[row, col] = values + 1
    return prefix, reused


@onlyBackends(["triton", "cute"])
class TestIndexing(RefEagerTestBase, TestCase):
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_count_top_level(self):
        @helion.kernel
        def fn(n: int, device: torch.device) -> torch.Tensor:
            out = torch.zeros([n], dtype=torch.int32, device=device)
            for tile in hl.tile(n, block_size=64):
                out[tile] = tile.count
            return out

        n = 100
        code, result = code_and_output(fn, (n, DEVICE))
        expected = torch.full([n], (n + 64 - 1) // 64, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_count_with_begin_end(self):
        @helion.kernel
        def fn(begin: int, end: int, device: torch.device) -> torch.Tensor:
            out = torch.zeros([1], dtype=torch.int32, device=device)
            for tile in hl.tile(begin, end, block_size=32):
                out[0] = tile.count
            return out

        begin, end = 10, 97
        code, result = code_and_output(fn, (begin, end, DEVICE))
        expected = torch.tensor(
            [(end - begin + 32 - 1) // 32], dtype=torch.int32, device=DEVICE
        )
        torch.testing.assert_close(result, expected)

    def test_arange(self):
        @helion.kernel
        def arange(length: int, device: torch.device) -> torch.Tensor:
            out = torch.empty([length], dtype=torch.int32, device=device)
            for tile in hl.tile(length):
                out[tile] = tile.index
            return out

        code, result = code_and_output(
            arange,
            (100, DEVICE),
            block_size=32,
        )
        torch.testing.assert_close(
            result, torch.arange(0, 100, device=DEVICE, dtype=torch.int32)
        )

    @onlyBackends(["triton"])
    @skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
    @skipIfRefEager("asserts on generated Triton code")
    def test_contiguity_hint_fires_on_swizzle_gather(self):
        # Allowlist: swizzled 1-D gathers with four-element contiguous runs.
        @helion.kernel(static_shapes=True)
        def swizzle_gather(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                idx = (tg.index // 4) * 512 + tg.index % 4
                out[tg] = scale[idx]
            return out

        @helion.kernel(static_shapes=True)
        def bitwise_swizzle_gather(
            scale: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                idx = (tg.index >> 2) * 512 + (tg.index & 3)
                out[tg] = scale[idx]
            return out

        n = 64
        scale = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        out = torch.empty(n, device=DEVICE, dtype=torch.float32)
        for fn in (swizzle_gather, bitwise_swizzle_gather):
            code, result = code_and_output(
                fn, (scale, out), indexing="pointer", block_size=[16]
            )
            self.assertIn("tl.max_contiguous(", code)
            self.assertIn(", [4])", code)
            i = torch.arange(n, device=DEVICE)
            torch.testing.assert_close(result, scale[(i // 4) * 512 + i % 4])

    @onlyBackends(["triton"])
    @skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
    @skipIfRefEager("asserts on generated Triton code")
    def test_contiguity_hint_ignores_outer_axis_modulus(self):
        # Allowlist: outer-axis modulo is constant along the inner run axis.
        @helion.kernel(static_shapes=True)
        def swizzle_2d(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            rows, cols = out.shape
            for tr, tc in hl.tile([rows, cols]):
                idx = (
                    (tc.index[None, :] // 4) * 512
                    + (tr.index[:, None] % 8) * 16
                    + tc.index[None, :] % 4
                )
                out[tr, tc] = scale[idx]
            return out

        rows, cols = 2, 32
        scale = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        out = torch.empty(rows, cols, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            swizzle_2d, (scale, out), indexing="pointer", block_size=[2, 32]
        )
        self.assertIn("tl.max_contiguous(", code)
        self.assertIn(", [1, 4])", code)
        r = torch.arange(rows, device=DEVICE)[:, None]
        c = torch.arange(cols, device=DEVICE)[None, :]
        torch.testing.assert_close(result, scale[(c // 4) * 512 + (r % 8) * 16 + c % 4])

    @onlyBackends(["triton"])
    @skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
    @skipIfRefEager("asserts on generated Triton code")
    def test_contiguity_hint_allows_scalar_outer_terms(self):
        # Allowlist: scalar row swizzle terms are uniform across the inner run.
        @helion.kernel(static_shapes=True)
        def swizzle_scalar_row(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            rows, cols = out.shape
            for tr in hl.tile(rows, block_size=1):
                row = tr.begin
                for tc in hl.tile(cols, block_size=16):
                    idx = (
                        ((row >> 7) * 1 + (tc.index >> 2)) * 512
                        + (row & 31) * 16
                        + ((row >> 5) & 3) * 4
                        + (tc.index & 3)
                    )
                    out[row, tc] = scale[idx]
            return out

        rows, cols = 2, 32
        scale = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        out = torch.empty(rows, cols, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            swizzle_scalar_row, (scale, out), indexing="pointer"
        )
        self.assertIn("tl.max_contiguous(", code)
        self.assertIn(", [4])", code)
        r = torch.arange(rows, device=DEVICE)[:, None]
        c = torch.arange(cols, device=DEVICE)[None, :]
        expected = scale[
            ((r >> 7) * 1 + (c >> 2)) * 512
            + (r & 31) * 16
            + ((r >> 5) & 3) * 4
            + (c & 3)
        ]
        torch.testing.assert_close(result, expected)

    @onlyBackends(["triton"])
    @skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
    @skipIfRefEager("asserts on generated Triton code")
    def test_contiguity_hint_does_not_fire_outside_swizzle(self):
        # Blocklist: plain loads, data gathers, permutations, and non-vectorizable widths.
        @helion.kernel(static_shapes=True)
        def affine_load(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                out[tg] = scale[tg]  # plain affine tile load (no gather)
            return out

        @helion.kernel(static_shapes=True)
        def clean_gather(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                out[tg] = scale[tg.index]  # contiguous gather: run == block (no win)
            return out

        @helion.kernel(static_shapes=True)
        def data_gather(
            scale: torch.Tensor, idxbuf: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                out[tg] = scale[idxbuf[tg]]  # data-dependent index: purity bail
            return out

        @helion.kernel(static_shapes=True)
        def permute_gather(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                idx = (tg.index % 4) * 512 + tg.index // 4  # permutation: run == 1
                out[tg] = scale[idx]
            return out

        @helion.kernel(static_shapes=True)
        def swizzle_gather_wide(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                idx = (tg.index // 4) * 512 + tg.index % 4  # swizzle, but 8-byte elt
                out[tg] = scale[idx]  # k(4) * 8 bytes == 32, not a vectorizable width
            return out

        @helion.kernel(static_shapes=True)
        def unsupported_bitwise_mask(
            scale: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(n):
                idx = (tg.index & 5) * 512 + (tg.index & 3)
                out[tg] = scale[idx]
            return out

        @helion.kernel(static_shapes=True)
        def scalar_shifted_mask(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            rows, cols = out.shape
            for tr in hl.tile(rows, block_size=1):
                row = tr.begin
                for tc in hl.tile(cols, block_size=16):
                    idx = ((tc.index + row) >> 2) * 512 + ((tc.index + row) & 3)
                    out[row, tc] = scale[idx]
            return out

        n = 64
        f32 = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        small = f32[:n].contiguous()
        out = torch.empty(n, device=DEVICE, dtype=torch.float32)
        idxbuf = torch.randint(0, 8192, (n,), device=DEVICE, dtype=torch.int64)
        i64 = torch.randint(0, 1000, (8192,), device=DEVICE, dtype=torch.int64)
        out64 = torch.empty(n, device=DEVICE, dtype=torch.int64)

        cases = [
            (affine_load, (small, out)),
            (clean_gather, (small, out)),
            (data_gather, (f32, idxbuf, out)),
            (permute_gather, (f32, out)),
            (swizzle_gather_wide, (i64, out64)),
            (unsupported_bitwise_mask, (f32, out)),
        ]
        for fn, args in cases:
            code, _ = code_and_output(fn, args, indexing="pointer", block_size=[16])
            self.assertNotIn(
                "tl.max_contiguous", code, f"{fn.fn.__name__} should get no hint"
            )

        out2d = torch.empty(2, 32, device=DEVICE)
        code, _ = code_and_output(scalar_shifted_mask, (f32, out2d), indexing="pointer")
        self.assertNotIn("tl.max_contiguous", code)

    @onlyBackends(["triton"])
    @skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
    @skipIfRefEager("asserts on generated Triton code")
    def test_contiguity_hint_does_not_fire_on_shifted_tile(self):
        # Blocklist: begin=1 breaks the four-element swizzle run.
        @helion.kernel(static_shapes=True)
        def shifted_swizzle(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(1, n):
                idx = (tg.index >> 2) * 512 + (tg.index & 3)
                out[tg] = scale[idx]
            return out

        n = 64
        scale = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(n, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            shifted_swizzle, (scale, out), indexing="pointer", block_size=[16]
        )
        self.assertNotIn("tl.max_contiguous", code)
        i = torch.arange(1, n, device=DEVICE)
        torch.testing.assert_close(result[1:], scale[(i // 4) * 512 + i % 4])

    @onlyBackends(["triton"])
    @skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
    @skipIfRefEager("asserts on generated Triton code")
    def test_contiguity_hint_fires_on_aligned_begin(self):
        # Allowlist: begin=4 keeps the four-element swizzle run aligned.
        @helion.kernel(static_shapes=True)
        def aligned_swizzle(scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            (n,) = out.shape
            for tg in hl.tile(4, n):
                idx = (tg.index >> 2) * 512 + (tg.index & 3)
                out[tg] = scale[idx]
            return out

        n = 64
        scale = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(n, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            aligned_swizzle, (scale, out), indexing="pointer", block_size=[16]
        )
        self.assertIn("tl.max_contiguous(", code)
        self.assertIn(", [4])", code)
        i = torch.arange(4, n, device=DEVICE)
        torch.testing.assert_close(result[4:], scale[(i // 4) * 512 + i % 4])

    @pytest.mark.xfail(
        _get_backend() == "cute",
        reason="CuTe matmul fallback with non-power-of-two static dimensions can generate invalid shared-memory indexing",
        run=False,
    )
    def test_hl_arange_non_power_of_2(self):
        @helion.kernel
        def _matmul_layernorm_bwd_dxdy(
            grad_out: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
            mean: torch.Tensor,
            rstd: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = z.shape
            k = x.shape[1]
            n = hl.specialize(n)
            k = hl.specialize(k)

            grad_x = torch.empty_like(x)
            grad_y = torch.zeros_like(y)

            for tile_m in hl.tile(m):
                z_tile = z[tile_m, :].to(torch.float32)
                dy_tile = grad_out[tile_m, :].to(torch.float32)
                w = weight[:].to(torch.float32)
                mean_tile = mean[tile_m]
                rstd_tile = rstd[tile_m]

                z_hat = (z_tile - mean_tile[:, None]) * rstd_tile[:, None]
                wdy = w * dy_tile
                c1 = torch.sum(z_hat * wdy, dim=-1, keepdim=True) / float(n)
                c2 = torch.sum(wdy, dim=-1, keepdim=True) / float(n)
                dz = (wdy - (z_hat * c1 + c2)) * rstd_tile[:, None]

                grad_x[tile_m, :] = (dz @ y[:, :].t().to(torch.float32)).to(x.dtype)
                grad_y_update = (x[tile_m, :].t().to(torch.float32) @ dz).to(y.dtype)

                hl.atomic_add(
                    grad_y,
                    [
                        hl.arange(0, k),
                        hl.arange(0, n),
                    ],
                    grad_y_update,
                )

            return grad_x, grad_y

        m, k, n = 5, 3, 7
        eps = 1e-5

        x = torch.randn((m, k), device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn((k, n), device=DEVICE, dtype=HALF_DTYPE)
        weight = torch.randn((n,), device=DEVICE, dtype=HALF_DTYPE)
        grad_out = torch.randn((m, n), device=DEVICE, dtype=HALF_DTYPE)

        z = (x @ y).to(torch.float32)
        var, mean = torch.var_mean(z, dim=-1, keepdim=True, correction=0)
        rstd = torch.rsqrt(var + eps)

        code, (grad_x, grad_y) = code_and_output(
            _matmul_layernorm_bwd_dxdy,
            (
                grad_out,
                x,
                y,
                z.to(x.dtype),
                mean.squeeze(-1),
                rstd.squeeze(-1),
                weight,
            ),
            block_size=[16],
            indexing="pointer",
        )

        # PyTorch reference gradients
        z_hat = (z - mean) * rstd
        wdy = weight.to(torch.float32) * grad_out.to(torch.float32)
        c1 = torch.sum(z_hat * wdy, dim=-1, keepdim=True) / float(n)
        c2 = torch.sum(wdy, dim=-1, keepdim=True) / float(n)
        dz = (wdy - (z_hat * c1 + c2)) * rstd
        ref_grad_x = (dz @ y.to(torch.float32).t()).to(grad_x.dtype)
        ref_grad_y = (x.to(torch.float32).t() @ dz).to(grad_y.dtype)

        torch.testing.assert_close(grad_x, ref_grad_x, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(grad_y, ref_grad_y, rtol=1e-2, atol=1e-2)
        # TODO(oulgen): needs mindot size mocked

    def test_pairwise_add(self):
        @helion.kernel()
        def pairwise_add(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0) - 1])
            for tile in hl.tile(out.size(0)):
                out[tile] = x[tile] + x[tile.index + 1]
            return out

        x = torch.randn([500], device=DEVICE)
        code, result = code_and_output(
            pairwise_add,
            (x,),
            block_size=32,
        )
        torch.testing.assert_close(result, x[:-1] + x[1:])

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_pairwise_add_commuted_and_multi_offset(self):
        @helion.kernel()
        def pairwise_add_variants(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0) - 3])
            for tile in hl.tile(out.size(0)):
                left = x[1 + tile.index]
                right = x[tile.index + 1 + 2]
                out[tile] = left + right
            return out

        x = torch.randn([256], device=DEVICE)
        code, result = code_and_output(
            pairwise_add_variants,
            (x,),
            block_size=32,
            indexing="tensor_descriptor",
        )
        expected = x[1:-2] + x[3:]
        torch.testing.assert_close(result, expected)

    def test_mask_store(self):
        @helion.kernel
        def masked_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                hl.store(out, [tile], x[tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            masked_store,
            (x,),
            block_size=16,
        )
        torch.testing.assert_close(
            result, torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0)
        )

    def test_mask_store_cartesian(self):
        @helion.kernel(autotune_effort="none")
        def cartesian_masked_store_kernel(
            A_packed: torch.Tensor,
            B: torch.Tensor,
            group_offsets: torch.Tensor,
        ) -> torch.Tensor:
            block_m = 8
            block_n = 8

            total_m, _ = A_packed.shape
            _, n = B.shape

            out = torch.zeros(total_m, n, device=A_packed.device, dtype=A_packed.dtype)

            groups = group_offsets.size(0) - 1

            for g in hl.grid(groups):
                start = group_offsets[g]
                end = group_offsets[g + 1]

                # Deliberately request a larger tile than the group so some rows go out of bounds.
                row_idx = start + hl.arange(block_m)
                col_idx = hl.arange(block_n)
                rows_valid = row_idx < end
                cols_valid = col_idx < n

                payload = torch.zeros(
                    block_m, block_n, device=out.device, dtype=out.dtype
                )

                # Mask keeps the logical writes in-bounds.
                mask_2d = rows_valid[:, None] & cols_valid[None, :]
                hl.store(
                    out,
                    [row_idx, col_idx],
                    payload.to(out.dtype),
                    extra_mask=mask_2d,
                )

            return out

        def _pack_inputs(
            group_a: list[torch.Tensor], group_b: list[torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert group_a, "group list must be non-empty"
            device = group_a[0].device
            dtype = group_a[0].dtype

            offsets = [0]
            for tensor in group_a:
                offsets.append(offsets[-1] + int(tensor.size(0)))

            group_offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
            packed = (
                torch.cat(group_a, dim=0).to(device=device, dtype=dtype).contiguous()
            )
            return packed, group_b[0], group_offsets

        dtype = HALF_DTYPE
        group_a = [
            torch.randn(m, 32, device=DEVICE, dtype=dtype).contiguous()
            for m in (8, 12, 4)
        ]
        group_b = [torch.randn(32, 4, device=DEVICE, dtype=dtype).contiguous()] * len(
            group_a
        )
        packed, shared_b, offsets = _pack_inputs(group_a, group_b)
        expected = torch.zeros(
            packed.size(0), shared_b.size(1), device=DEVICE, dtype=dtype
        )
        result = cartesian_masked_store_kernel(packed, shared_b, offsets)
        torch.testing.assert_close(result, expected)

    def test_mask_store_cartesian_3d(self):
        @helion.kernel(autotune_effort="none")
        def cartesian_masked_store_kernel_3d(
            group_offsets: torch.Tensor, total_m: int, n: int, p: int
        ) -> torch.Tensor:
            block_m = 4
            block_n = 5
            block_p = 6

            groups = group_offsets.size(0) - 1

            out = torch.zeros(
                total_m, n, p, device=group_offsets.device, dtype=HALF_DTYPE
            )

            for g in hl.grid(groups):
                start = group_offsets[g]
                end = group_offsets[g + 1]

                row_idx = start + hl.arange(block_m)
                col_idx = hl.arange(block_n)
                depth_idx = hl.arange(block_p)

                rows_valid = row_idx < end
                cols_valid = col_idx < n
                depth_valid = depth_idx < p

                mask_3d = (
                    rows_valid[:, None, None]
                    & cols_valid[None, :, None]
                    & depth_valid[None, None, :]
                )

                payload = torch.ones(
                    block_m, block_n, block_p, device=out.device, dtype=out.dtype
                )

                hl.store(
                    out, [row_idx, col_idx, depth_idx], payload, extra_mask=mask_3d
                )

            return out

        dtype = HALF_DTYPE
        group_offsets = torch.tensor([0, 2, 5, 6], device=DEVICE, dtype=torch.int32)
        n, p = 4, 3
        total_m = int(group_offsets[-1])
        expected = torch.zeros((total_m, n, p), device=DEVICE, dtype=dtype)
        expected[:2] = 1
        expected[2:5] = 1
        expected[5:6] = 1
        result = cartesian_masked_store_kernel_3d(group_offsets, total_m, n, p)
        torch.testing.assert_close(result, expected)

    def test_mask_load(self):
        @helion.kernel
        def masked_load(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                out[tile] = hl.load(x, [tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            masked_load,
            (x,),
            block_size=16,
        )
        torch.testing.assert_close(
            result, torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0)
        )

    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_extra_mask_load(self):
        """Verify extra_mask loads produce correct results with block_ptr
        and tensor_descriptor backends.
        """

        @helion.kernel
        def masked_load_3d(
            x: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            m, n, k = x.size()
            out = torch.zeros_like(x)
            for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
                out[tile_m, tile_n, tile_k] = hl.load(
                    x,
                    [tile_m, tile_n, tile_k],
                    extra_mask=mask[tile_m, tile_n, tile_k],
                )
            return out

        x = torch.randn(8, 4, 16, device=DEVICE)
        block_size = [4, 4, 8]

        backends = []
        if supports_block_ptr():
            backends.append("block_ptr")
        if supports_tensor_descriptor():
            backends.append("tensor_descriptor")
        if not backends:
            self.skipTest("needs block_ptr or tensor_descriptor indexing")

        mask_shapes = [
            (8, 4, 16),  # full size, no broadcast
            (1, 4, 1),  # broadcast along M and K
        ]

        for indexing in backends:
            for shape in mask_shapes:
                with self.subTest(indexing=indexing, mask_shape=shape):
                    mask = torch.randint(0, 2, shape, device=DEVICE, dtype=torch.bool)
                    args = (x, mask)

                    _, result_pointer = code_and_output(
                        masked_load_3d,
                        args,
                        block_size=block_size,
                        indexing="pointer",
                    )
                    code_test, result_test = code_and_output(
                        masked_load_3d,
                        args,
                        block_size=block_size,
                        indexing=indexing,
                    )
                    if _get_backend() == "triton":
                        self.assertIn(indexing, code_test)
                        self.assertIn("tl.where", code_test)
                    torch.testing.assert_close(result_test, result_pointer)

    def test_extra_mask_load_size_one_dim(self):
        """An extra_mask load whose only block index hits a size-1 dimension."""

        @helion.kernel(static_shapes=True)
        def gather_rows(
            x: torch.Tensor,
            base: torch.Tensor,
            valid: torch.Tensor,
            out: torch.Tensor,
        ) -> None:
            for tile_j, tile_r in hl.tile(
                [out.size(0), out.size(1)], block_size=[1, None]
            ):
                j = tile_j.begin
                v = tile_r.index < valid[j]
                xt = hl.load(x, [base[j] + tile_r.index, 0], extra_mask=v)
                out[tile_j.begin, tile_r] = torch.where(v, xt, 0)

        # x.size(0) == 1 makes the row index fold away, leaving a scalar offset.
        x = torch.randn(1, 4, device=DEVICE, dtype=HALF_DTYPE)
        base = torch.zeros(1, device=DEVICE, dtype=torch.int32)
        valid = torch.ones(1, device=DEVICE, dtype=torch.int32)
        out = x.new_empty(1, 8)
        code_and_output(gather_rows, (x, base, valid, out), block_size=[8])
        expected = torch.zeros_like(out)
        expected[0, 0] = x[0, 0]
        torch.testing.assert_close(out, expected)

    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @skipIfRefEager("test checks generated Triton code")
    def test_mask_store_falls_back_to_pointer(self):
        @helion.kernel
        def masked_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                hl.store(out, [tile], x[tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)

        backends = []
        if supports_block_ptr():
            backends.append("block_ptr")
        if supports_tensor_descriptor():
            backends.append("tensor_descriptor")
        if not backends:
            self.skipTest("needs block_ptr or tensor_descriptor indexing")

        for indexing in backends:
            with self.subTest(indexing=indexing):
                code, result = code_and_output(
                    masked_store,
                    (x,),
                    block_size=16,
                    indexing=indexing,
                )
                if _get_backend() == "triton":
                    # The masked store should fall back to pointer
                    store_lines = [
                        line for line in code.splitlines() if "tl.store(" in line
                    ]
                    self.assertTrue(store_lines)
                    for line in store_lines:
                        self.assertNotIn("block_ptr", line)
                        self.assertNotIn("tensor_descriptor", line)
                torch.testing.assert_close(
                    result,
                    torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0),
                )

    def test_tile_begin_end(self):
        @helion.kernel
        def tile_range_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(x.size(0)):
                for inner_tile in hl.tile(tile.begin, tile.end):
                    out[inner_tile] = x[inner_tile]
            return out

        x = torch.randn([100], device=DEVICE)
        code, result = code_and_output(
            tile_range_copy,
            (x,),
            block_size=[32, 16],
        )
        torch.testing.assert_close(result, x)
        code, result = code_and_output(
            tile_range_copy,
            (x,),
            block_size=[1, 1],
        )
        torch.testing.assert_close(result, x)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_block_size(self):
        @helion.kernel
        def test_block_size_access(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile in hl.tile(x.size(0)):
                out[tile] = tile.block_size
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            test_block_size_access,
            (x,),
            block_size=16,
        )
        expected = torch.full_like(x, 16, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_block_size_access,
            (x,),
            block_size=1,
        )
        expected = torch.full_like(x, 1, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "IndexOffsetOutOfRangeForInt32 error is not raised in ref eager mode"
    )
    @skipIfLowVRAM(
        "Test requires high VRAM",
        required_bytes=_LARGE_BF16_REQUIRED_BYTES,
    )
    @skipIfXPU("worker crash on XPU")
    def test_int32_offset_out_of_range_error(self):
        repro_config = helion.Config(
            block_sizes=[32, 32],
            flatten_loops=[False],
            indexing="pointer",
            l2_groupings=[1],
            loop_orders=[[0, 1]],
            num_stages=3,
            num_warps=4,
            pid_type="flat",
            range_flattens=[None] if not use_tileir_tunables() else [],
            range_multi_buffers=[None] if not use_tileir_tunables() else [],
            range_num_stages=[],
            range_unroll_factors=[0] if not use_tileir_tunables() else [],
            range_warp_specializes=[],
        )

        def make_kernel(*, index_dtype: torch.dtype | None = None):
            kwargs = {"config": repro_config, "static_shapes": True}
            if index_dtype is not None:
                kwargs["index_dtype"] = index_dtype
            decorator = helion.kernel(**kwargs)

            @decorator
            def repro_bf16_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x, y = torch.broadcast_tensors(x, y)
                out = torch.empty(
                    x.shape,
                    dtype=torch.promote_types(x.dtype, y.dtype),
                    device=x.device,
                )
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            return repro_bf16_add

        def run_case(
            shape,
            *,
            index_dtype: torch.dtype | None,
            expect_int64_in_code: bool = False,
            expect_error: type[Exception] | None = None,
        ) -> None:
            kernel = make_kernel(index_dtype=index_dtype)
            x = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
            y = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
            torch.accelerator.synchronize()
            if expect_error is not None:
                with self.assertRaisesRegex(
                    expect_error,
                    f"index_dtype is {index_dtype}",
                ):
                    code_and_output(kernel, (x, y))
                del x, y
                torch.cuda.empty_cache()
                torch.accelerator.synchronize()
                return

            code, out = code_and_output(kernel, (x, y))
            torch.accelerator.synchronize()
            checker = self.assertIn if expect_int64_in_code else self.assertNotIn
            int64_token = "cutlass.Int64" if _get_backend() == "cute" else "tl.int64"
            checker(int64_token, code)
            torch.accelerator.synchronize()
            ref_out = torch.add(x, y)
            del x, y
            torch.cuda.empty_cache()
            torch.accelerator.synchronize()
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)

        small_shape = (128, 128)
        large_shape = _LARGE_BF16_SHAPE

        run_case(
            small_shape,
            index_dtype=torch.int32,
            expect_int64_in_code=False,
            expect_error=None,
        )
        run_case(
            large_shape,
            index_dtype=torch.int32,
            expect_int64_in_code=False,
            expect_error=helion.exc.InputTensorNumelExceedsIndexType,
        )
        # Add margin for reference + comparison buffers (isclose/temporary).
        run_case(
            large_shape,
            index_dtype=torch.int64,
            expect_int64_in_code=True,
            expect_error=None,
        )
        run_case(
            large_shape,
            index_dtype=None,
            expect_int64_in_code=True,
            expect_error=None,
        )

    @skipIfRefEager("specialization_key is not used in ref eager mode")
    def test_dynamic_shape_specialization_key_tracks_large_tensors(self) -> None:
        @helion.kernel(static_shapes=False)
        def passthrough(x: torch.Tensor) -> torch.Tensor:
            return x

        @helion.kernel(static_shapes=False, index_dtype=torch.int64)
        def passthrough_int64(x: torch.Tensor) -> torch.Tensor:
            return x

        meta = "meta"
        small = torch.empty((4, 4), device=meta)
        large = torch.empty((51200, 51200), device=meta)

        self.assertNotEqual(
            passthrough.specialization_key((small,)),
            passthrough.specialization_key((large,)),
        )
        self.assertEqual(
            passthrough_int64.specialization_key((small,)),
            passthrough_int64.specialization_key((large,)),
        )

    @skipIfRefEager("specialization_key is not used in ref eager mode")
    def test_dynamic_shape_specialization_key_does_not_bucket_zero_one(self) -> None:
        @helion.kernel(static_shapes=False)
        def passthrough(x: torch.Tensor) -> torch.Tensor:
            return x

        keys = {
            passthrough.specialization_key((torch.empty((n, 4), device=DEVICE),))
            for n in (0, 1, 2, 9)
        }
        self.assertEqual(len(keys), 1)

    @skipIfRefEager("specialization_key is not used in ref eager mode")
    def test_symint_specialization_key_disambiguates_shape_envs(self) -> None:
        @helion.kernel(static_shapes=True)
        def passthrough(x: torch.Tensor) -> torch.Tensor:
            return x

        se1 = ShapeEnv()
        se2 = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=se1)
        mode2 = FakeTensorMode(shape_env=se2)

        si1 = se1.create_unbacked_symint()
        si2 = se2.create_unbacked_symint()
        # Both fresh ShapeEnvs produce the same symbol name
        self.assertEqual(str(si1.node.expr), str(si2.node.expr))

        meta = "meta"
        with mode1:
            ft1 = torch.empty(si1, 4, device=meta)
        with mode2:
            ft2 = torch.empty(si2, 4, device=meta)

        self.assertNotEqual(
            passthrough.specialization_key((ft1,)),
            passthrough.specialization_key((ft2,)),
        )

    @skipIfRefEager("Test checks generated code")
    def test_program_id_cast_to_int64(self):
        """Test that tl.program_id() is cast to int64 when index_dtype is int64."""

        @helion.kernel(index_dtype=torch.int64)
        def add_kernel_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel(index_dtype=torch.int32)
        def add_kernel_int32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)

        # Test int64 case: program_id should be cast to int64
        code_int64, result_int64 = code_and_output(add_kernel_int64, (x, y))
        if _get_backend() == "cute":
            self.assertIn("cutlass.Int64(cute.arch.block_idx()[0])", code_int64)
        else:
            self.assertIn("tl.program_id(0).to(tl.int64)", code_int64)

        # Test int32 case: program_id should NOT be cast
        code_int32, result_int32 = code_and_output(add_kernel_int32, (x, y))
        if _get_backend() == "cute":
            self.assertNotIn("cutlass.Int64(cute.arch.block_idx()[0])", code_int32)
            self.assertIn("cutlass.Int32(cute.arch.block_idx()[0])", code_int32)
        else:
            self.assertNotIn(".to(tl.int64)", code_int32)
            self.assertIn("tl.program_id(0)", code_int32)

        # Both should produce correct results
        expected = x + y
        torch.testing.assert_close(result_int64, expected)
        torch.testing.assert_close(result_int32, expected)

    @skipIfRefEager("Test checks for no IMA")
    @skipIfLowVRAM(
        "Test requires large memory",
        required_bytes=_LARGE_TENSOR_REQUIRED_BYTES,
    )
    @skipIfXPU("Timeout on XPU")
    def test_large_tensor(self):
        @helion.kernel(autotune_effort="none")
        def f(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty(x.shape)
            for (b,) in hl.grid([x.shape[0]]):
                for (x_tile,) in hl.tile([x.shape[1]]):
                    out[b, x_tile] = x[b, x_tile]
            return out

        inp = torch.randn(
            _LARGE_TENSOR_B,
            _LARGE_TENSOR_D,
            device=DEVICE,
            dtype=HALF_DTYPE,
        )
        out = f(inp)
        assert (out == inp).all()

    def test_assign_int(self):
        @helion.kernel
        def fn(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                x[tile] = 1
            return x

        x = torch.zeros([200], device=DEVICE)
        expected = torch.ones_like(x)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_id(self):
        @helion.kernel
        def test_tile_id_access(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile in hl.tile(x.size(0)):
                out[tile] = tile.id
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            test_tile_id_access,
            (x,),
            block_size=16,
        )
        expected = torch.arange(4, device=DEVICE, dtype=torch.int32).repeat_interleave(
            repeats=16
        )
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_tile_id_access,
            (x,),
            block_size=1,
        )
        expected = torch.arange(64, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_id_1d_indexing(self):
        @helion.kernel
        def test_tile_id_atomic_add(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m in hl.tile(x.size(0)):
                hl.atomic_add(out, [tile_m.id], 1)
            return out

        x = torch.randn(64, device=DEVICE)
        code, result = code_and_output(
            test_tile_id_atomic_add,
            (x,),
            block_size=[
                16,
            ],
        )

        expected = torch.zeros(64, device=DEVICE, dtype=torch.int32)
        expected[:4] = 1
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_tile_id_atomic_add,
            (x,),
            block_size=[
                1,
            ],
        )
        expected = torch.ones(64, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_id_2d_indexing(self):
        @helion.kernel
        def test_tile_id_index_st(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m.id, tile_n.id] = 1
            return out

        x = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            test_tile_id_index_st,
            (x,),
            block_size=[16, 16],
        )

        expected = torch.zeros(64, 64, device=DEVICE, dtype=torch.int32)
        expected[:4, :4] = 1
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_tile_id_index_st,
            (x,),
            block_size=[1, 1],
        )
        expected = torch.ones(64, 64, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_atomic_add_symint(self):
        @helion.kernel(config={"block_size": 32})
        def fn(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_add(x, [tile], tile.block_size + 1)
            return x

        x = torch.zeros([200], device=DEVICE)
        expected = x + 33
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, expected)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_arange_tile_block_size(self):
        @helion.kernel(autotune_effort="none")
        def arange_from_block_size(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0)], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0)):
                # Test the exact pattern requested: torch.arange(tile.block_size, device=x.device)
                out[tile] = torch.arange(tile.block_size, device=x.device)
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_from_block_size,
            (x,),
            block_size=16,
        )
        expected = torch.arange(16, dtype=torch.int32, device=DEVICE).repeat(4)
        torch.testing.assert_close(result, expected)

    def test_arange_two_args(self):
        @helion.kernel(autotune_effort="none")
        def arange_two_args(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0)], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0)):
                # Test the exact pattern requested: torch.arange(tile.begin, tile.begin+tile.block_size, device=x.device)
                out[tile] = torch.arange(
                    tile.begin, tile.begin + tile.block_size, device=x.device
                )
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_two_args,
            (x,),
            block_size=16,
        )
        expected = torch.arange(64, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_arange_three_args_step(self):
        @helion.kernel(config={"block_size": 8})
        def arange_three_args_step(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) // 2], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0) // 2):
                # Test the exact pattern requested: torch.arange(start, end, step=2, device=x.device)
                start_idx = tile.begin * 2
                end_idx = (tile.begin + tile.block_size) * 2
                out[tile] = torch.arange(start_idx, end_idx, step=2, device=x.device)
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_three_args_step,
            (x,),
        )
        expected = torch.arange(0, 64, step=2, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_arange_hl_alias(self):
        @helion.kernel(config={"block_size": 8})
        def arange_three_args_step(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) // 2], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0) // 2):
                start_idx = tile.begin * 2
                end_idx = (tile.begin + tile.block_size) * 2
                out[tile] = hl.arange(start_idx, end_idx, step=2)
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_three_args_step,
            (x,),
        )
        expected = torch.arange(0, 64, step=2, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_arange_block_size_multiple(self):
        """Test that tile.block_size * constant works in hl.arange"""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def arange_block_size_mul(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) * 2], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0)):
                indices = hl.arange(
                    tile.begin * 2, tile.begin * 2 + tile.block_size * 2
                )
                out[indices] = indices
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(arange_block_size_mul, (x,))

        expected = torch.arange(128, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_slice_block_size_multiple(self):
        """Test that tile.block_size * constant works as slice bounds"""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def arange_block_size_mul(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) * 2], dtype=torch.int32, device=x.device)
            ones = torch.ones_like(out)
            for tile in hl.tile(x.size(0)):
                indices_start = tile.begin * 2
                indices_end = indices_start + tile.block_size * 2
                out[indices_start:indices_end] = ones[indices_start:indices_end]
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(arange_block_size_mul, (x,))

        expected = torch.ones(128, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_broadcasting_pointer_indexing(self):
        x = torch.randn([16, 24, 32], device=DEVICE)
        bias1 = torch.randn([1, 24, 32], device=DEVICE)
        bias2 = torch.randn([16, 1, 32], device=DEVICE)
        code, result = code_and_output(
            broadcast_add_3d,
            (x, bias1, bias2),
            indexing="pointer",
            block_size=[8, 8, 8],
        )
        expected = x + bias1 + bias2
        torch.testing.assert_close(result, expected)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_broadcasting_block_ptr_indexing(self):
        x = torch.randn([16, 24, 32], device=DEVICE)
        bias1 = torch.randn([1, 24, 32], device=DEVICE)
        bias2 = torch.randn([16, 1, 32], device=DEVICE)
        code, result = code_and_output(
            broadcast_add_3d,
            (x, bias1, bias2),
            indexing="block_ptr",
            block_size=[8, 8, 8],
        )
        expected = x + bias1 + bias2
        torch.testing.assert_close(result, expected)

    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() == "tl._experimental_make_tensor_descriptor",
        "LLVM ERROR: Illegal shared layout",
    )
    def test_broadcasting_tensor_descriptor_indexing(self):
        x = torch.randn([16, 24, 32], device=DEVICE)
        bias1 = torch.randn([1, 24, 32], device=DEVICE)
        bias2 = torch.randn([16, 1, 32], device=DEVICE)
        code, result = code_and_output(
            broadcast_add_3d,
            (x, bias1, bias2),
            indexing="tensor_descriptor",
            block_size=[8, 8, 8],
        )
        expected = x + bias1 + bias2
        torch.testing.assert_close(result, expected)

    def test_size1_dimension_tile_reshape(self):
        """Test that tile indexing on size-1 dimensions works with reshape.

        This tests a fix where loading from a tensor with a size-1 dimension
        and then reshaping to tile sizes would fail because shape inference
        returned [1, block_size] instead of [block_size_0, block_size_1].
        """

        @helion.kernel(autotune_effort="none")
        def size1_reshape_kernel(
            x: torch.Tensor,
            out: torch.Tensor,
        ):
            for tile_1, tile_2 in hl.tile([x.size(0), x.size(1)]):
                block = x[tile_1, tile_2]
                # This reshape would fail before the fix when x.size(0) == 1
                block_reshape = block.reshape([tile_1, tile_2])
                out[tile_1, tile_2] = block_reshape

        # Test with size-1 first dimension (this was the failing case)
        x = torch.randn(1, 16, dtype=torch.bfloat16, device=DEVICE)
        out = torch.empty_like(x)
        code, _ = code_and_output(size1_reshape_kernel, (x, out))
        torch.testing.assert_close(out, x)

        # Test with non-size-1 first dimension (should also work)
        x2 = torch.randn(4, 16, dtype=torch.bfloat16, device=DEVICE)
        out2 = torch.empty_like(x2)
        size1_reshape_kernel(x2, out2)
        torch.testing.assert_close(out2, x2)

    def test_size1_dimension_variable_tile_range(self):
        """Test tile indexing on size-1 dimensions with variable tile ranges.

        This tests the case where a tile loop uses runtime-determined start/end
        values (from tensor lookups) and indexes into a size-1 dimension.
        """

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def variable_tile_range_kernel(
            query: torch.Tensor,
            query_start_lens: torch.Tensor,
            num_seqs: int,
            output: torch.Tensor,
        ) -> None:
            q_size_1 = hl.specialize(query.size(1))

            for seq_tile in hl.tile(num_seqs, block_size=1):
                seq_idx = seq_tile.begin
                query_start = query_start_lens[seq_idx]
                query_end = query_start_lens[seq_idx + 1]

                for tile_q in hl.tile(query_start, query_end):
                    q = query[tile_q, :]
                    q = q.reshape([tile_q.block_size, q_size_1])
                    output[tile_q, :] = q

        query = torch.randn(1, 16, dtype=torch.bfloat16, device=DEVICE)
        query_start_lens = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        num_seqs = 1
        out = torch.empty_like(query)

        code, _ = code_and_output(
            variable_tile_range_kernel, (query, query_start_lens, num_seqs, out)
        )
        torch.testing.assert_close(out, query)

    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() != "tl._experimental_make_tensor_descriptor",
        "Not using experimental tensor descriptor",
    )
    def test_reduction_tensor_descriptor_indexing_block_size(self):
        x = torch.randn([64, 64], dtype=torch.float32, device=DEVICE)

        # Given block_size 4, tensor_descriptor should not actually be used
        # Convert to default pointer indexing
        code, result = code_and_output(
            reduction_sum,
            (x,),
            indexing="tensor_descriptor",
            block_size=[4],
        )

        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)

    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() != "tl._experimental_make_tensor_descriptor",
        "Not using experimental tensor descriptor",
    )
    def test_reduction_tensor_descriptor_indexing_reduction_loop(self):
        x = torch.randn([64, 256], dtype=HALF_DTYPE, device=DEVICE)

        # Given reduction_loop 2, # of columns not compatible with tensor_descriptor
        # Convert to default pointer indexing
        code, result = code_and_output(
            reduction_sum,
            (x,),
            indexing="tensor_descriptor",
            block_size=[8],
            reduction_loops=[8],
        )

        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)

    @onlyBackends(["triton"])
    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @skipIfTileIR("block-size overshoot is gated to the triton backend")
    def test_tensor_descriptor_rejects_overshoot_dynamic_dim(self):
        # Matmul block-size overshoot lets the autotuner pick an M/N block
        # larger than a small dimension. When that dimension is dynamic (here M
        # is left unspecialized) the static block_size > dim_size guard cannot
        # fire, so without the symbolic-dim hint check the overshooting block
        # would ride onto the TMA path and build a descriptor with
        # boxDim > tensorDim -- an invalid TMA descriptor that crashes at
        # runtime with a misaligned-address error. The overshooting dimension
        # must instead fall back to pointer indexing.
        @helion.kernel(static_shapes=False)
        def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.shape
            k2, n = b.shape
            hl.specialize(k)
            hl.specialize(n)
            out = torch.empty([m, n], dtype=torch.float32, device=a.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc
            return out

        # M=16 is dynamic; block_m=64 overshoots it.
        a = torch.randn([16, 64], dtype=HALF_DTYPE, device=DEVICE)
        b = torch.randn([64, 64], dtype=HALF_DTYPE, device=DEVICE)
        code, result = code_and_output(
            matmul,
            (a, b),
            block_sizes=[64, 64, 32],
            indexing="tensor_descriptor",
        )
        torch.testing.assert_close(result, (a @ b).float(), atol=1e-1, rtol=1e-1)

        # _BLOCK_SIZE_0 is the (overshooting) M tile; it must never appear inside
        # a tensor-descriptor box. Non-overshooting tensors (e.g. b) may still
        # use a descriptor.
        descriptor_lines = [
            line for line in code.splitlines() if "make_tensor_descriptor(" in line
        ]
        for line in descriptor_lines:
            self.assertNotIn("_BLOCK_SIZE_0", line)

    def test_2d_slice_index(self):
        """Test both setter from scalar and getter for [:,i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[1]
            for i in hl.grid(N):
                dst[:, i] = 1.0  # Test setter with scalar
                src[:, i] = dst[:, i]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([1, N], device=DEVICE)
        dst = torch.zeros([1, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([1, N], device=DEVICE)
        expected_dst = torch.ones([1, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_2d_full_slice(self):
        """Test both setter from scalar and getter for [:,:]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[1]
            for _ in hl.grid(N):
                dst[:, :] = 1.0  # Test setter with scalar
                src[:, :] = dst[:, :]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([1, N], device=DEVICE)
        dst = torch.zeros([1, N], device=DEVICE)

        code, (src_result, dst_result) = code_and_output(kernel, (src, dst))

        # Both should be ones after the kernel
        expected_src = torch.ones([1, N], device=DEVICE)
        expected_dst = torch.ones([1, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

        if _get_backend() == "cute":
            # Regression: the scalar store `dst[:, :] = 1.0` must vary across
            # the slice's second dim. Either a lane loop variable
            # (`rindex_*`) or a per-thread index derived from
            # `cute.arch.thread_idx` (`indices_*`) is correct. What is NOT
            # correct is binding the slice to a constant or grid-only PID,
            # which would only write one element per block and race with the
            # subsequent load.
            store_line = next(
                (
                    line
                    for line in code.split("\n")
                    if ".store(cutlass.Float32(1.0))" in line
                ),
                None,
            )
            self.assertIsNotNone(store_line)
            assert "rindex_" in store_line or "indices_" in store_line, store_line
            # Guard against the original race condition where the second-dim
            # index resolved to a constant.
            self.assertNotIn(
                "cutlass.Int32(0) * cutlass.Int32(dst.layout.stride[1])",
                store_line,
            )

    def test_1d_index(self):
        """Test both setter from scalar and getter for [i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i] = 1.0  # Test setter with scalar
                src[i] = dst[i]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([N], device=DEVICE)
        expected_dst = torch.ones([N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_1d_full_slice(self):
        """Test both setter from scalar and getter for [:] with multiple scalar types"""

        @helion.kernel(config={"block_size": 128})
        def kernel(
            src_float: torch.Tensor,
            dst_float: torch.Tensor,
            src_int: torch.Tensor,
            dst_int: torch.Tensor,
            src_symint: torch.Tensor,
            dst_symint: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            N = src_float.shape[0]
            for tile in hl.tile(N):
                # Test float scalar
                dst_float[:] = 1.0
                src_float[:] = dst_float[:]

                # Test int scalar
                dst_int[:] = 99
                src_int[:] = dst_int[:]

                # Test SymInt scalar
                dst_symint[:] = tile.block_size
                src_symint[:] = dst_symint[:]

            return (
                src_float,
                dst_float,
                src_int,
                dst_int,
                src_symint,
                dst_symint,
            )

        N = 128
        src_float = torch.zeros([N], device=DEVICE)
        dst_float = torch.zeros([N], device=DEVICE)
        src_int = torch.zeros([N], device=DEVICE)
        dst_int = torch.zeros([N], device=DEVICE)
        src_symint = torch.zeros([N], device=DEVICE)
        dst_symint = torch.zeros([N], device=DEVICE)

        results = kernel(
            src_float,
            dst_float,
            src_int,
            dst_int,
            src_symint,
            dst_symint,
        )

        # Check float results
        expected_float = torch.ones([N], device=DEVICE)
        torch.testing.assert_close(results[0], expected_float)
        torch.testing.assert_close(results[1], expected_float)

        # Check int results
        expected_int = torch.full([N], 99.0, device=DEVICE)
        torch.testing.assert_close(results[2], expected_int)
        torch.testing.assert_close(results[3], expected_int)

        # Check SymInt results
        expected_symint = torch.full([N], 128.0, device=DEVICE)
        torch.testing.assert_close(results[4], expected_symint)
        torch.testing.assert_close(results[5], expected_symint)

    def test_1d_slice_from_indexed_value(self):
        """buf[:] = zeros[i] - Assign slice from indexed value"""

        @helion.kernel(autotune_effort="none")
        def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
            N = buf.shape[0]
            for i in hl.grid(N):
                buf[:] = zeros[i]
            return buf

        N = 128
        buf = torch.ones([N], device=DEVICE)
        zeros = torch.zeros([N], device=DEVICE)

        result = kernel(buf.clone(), zeros)
        expected = torch.zeros([N], device=DEVICE)
        torch.testing.assert_close(result, expected)

    @unittest.skip("takes 5+ minutes to run")
    def test_1d_indexed_value_from_slice(self):
        """buf2[i] = buf[:] - Assign slice to indexed value"""

        @helion.kernel
        def getter_kernel(buf: torch.Tensor, buf2: torch.Tensor) -> torch.Tensor:
            N = buf2.shape[0]
            for i in hl.grid(N):
                buf2[i, :] = buf[:]
            return buf2

        N = 128
        buf = torch.rand([N], device=DEVICE)
        buf2 = torch.zeros(
            [N, N], device=DEVICE
        )  # Note: Different shape to accommodate slice assignment

        result = getter_kernel(buf.clone(), buf2.clone())
        expected = buf.expand(N, N).clone()
        torch.testing.assert_close(result, expected)

    def test_1d_index_from_index(self):
        """buf[i] = zeros[i] - Index to index assignment"""

        @helion.kernel(autotune_effort="none")
        def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
            N = buf.shape[0]
            for i in hl.grid(N):
                buf[i] = zeros[i]
            return buf

        N = 128
        buf = torch.ones([N], device=DEVICE)
        zeros = torch.zeros([N], device=DEVICE)

        result = kernel(buf.clone(), zeros)
        expected = torch.zeros([N], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_mixed_slice_index(self):
        """Test both setter from scalar and getter for [i,:]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i, :] = 1.0  # Test setter with scalar
                src[i, :] = dst[i, :]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([N, N], device=DEVICE)
        dst = torch.zeros([N, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([N, N], device=DEVICE)
        expected_dst = torch.ones([N, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_strided_slice(self):
        """Test both setter from scalar and getter for strided slices [::2] and [1::3]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src1: torch.Tensor,
            dst1: torch.Tensor,
            src2: torch.Tensor,
            dst2: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            for _ in hl.grid(1):
                # Test [::2] - every other element starting from 0
                dst1[::2] = 1.0  # Test setter with scalar
                src1[::2] = dst1[::2]  # Test getter from dst and setter to src

                # Test [1::3] - every 3rd element starting from 1
                dst2[1::3] = 2.0  # Test setter with scalar
                src2[1::3] = dst2[1::3]  # Test getter from dst and setter to src
            return src1, dst1, src2, dst2

        N = 128
        src1 = torch.zeros([N], device=DEVICE)
        dst1 = torch.zeros([N], device=DEVICE)
        src2 = torch.zeros([N], device=DEVICE)
        dst2 = torch.zeros([N], device=DEVICE)

        src1_result, dst1_result, src2_result, dst2_result = kernel(
            src1, dst1, src2, dst2
        )

        # Only even indices should be ones for [::2]
        expected_src1 = torch.zeros([N], device=DEVICE)
        expected_src1[::2] = 1.0
        expected_dst1 = expected_src1.clone()
        torch.testing.assert_close(src1_result, expected_src1)
        torch.testing.assert_close(dst1_result, expected_dst1)

        # Elements at indices 1, 4, 7, ... should be twos for [1::3]
        expected_src2 = torch.zeros([N], device=DEVICE)
        expected_src2[1::3] = 2.0
        expected_dst2 = expected_src2.clone()
        torch.testing.assert_close(src2_result, expected_src2)
        torch.testing.assert_close(dst2_result, expected_dst2)

    @skipIfCute("CuTe negative indexes can poison the CUDA context")
    def test_negative_indexing(self):
        """Test both setter from scalar and getter for [-1]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for _ in hl.grid(1):
                dst[-1] = 1.0  # Test setter with scalar
                src[-1] = dst[-1]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Only last element should be one
        expected_src = torch.zeros([N], device=DEVICE)
        expected_src[-1] = 1.0
        expected_dst = expected_src.clone()
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfCute("CuTe negative indexes can poison the CUDA context")
    def test_negative_indexing_multidim(self):
        """Test negative indexing on multiple dimensions: x[-1, -1]"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            for _ in hl.grid(1):
                x[-1, -1] = 42.0
            return x

        M, N = 64, 128
        x = torch.zeros([M, N], device=DEVICE)
        result = kernel(x)

        expected = torch.zeros([M, N], device=DEVICE)
        expected[-1, -1] = 42.0
        torch.testing.assert_close(result, expected)

    @skipIfCute("CuTe negative indexes can poison the CUDA context")
    def test_negative_indexing_with_tile(self):
        """Test mixed tile and negative index: x[tile, -1]"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            (rows,) = x.shape[:1]
            for tile_r in hl.tile(rows):
                x[tile_r, -1] = 1.0
            return x

        M, N = 64, 128
        x = torch.zeros([M, N], device=DEVICE)
        result = kernel(x)

        expected = torch.zeros([M, N], device=DEVICE)
        expected[:, -1] = 1.0
        torch.testing.assert_close(result, expected)

    def test_ellipsis_indexing(self):
        """Test both setter from scalar and getter for [..., i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[-1]
            for i in hl.grid(N):
                dst[..., i] = 1.0  # Test setter with scalar
                src[..., i] = dst[..., i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([2, 3, N], device=DEVICE)
        dst = torch.zeros([2, 3, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements should be ones after the kernel
        expected_src = torch.ones([2, 3, N], device=DEVICE)
        expected_dst = torch.ones([2, 3, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_ellipsis_trailing(self):
        """Test trailing ellipsis: x[i, ...]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i, ...] = 1.0
                src[i, ...] = dst[i, ...]
            return src, dst

        N = 8
        src = torch.zeros([N, 4, 16], device=DEVICE)
        dst = torch.zeros([N, 4, 16], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        expected = torch.ones([N, 4, 16], device=DEVICE)
        torch.testing.assert_close(src_result, expected)
        torch.testing.assert_close(dst_result, expected)

    def test_ellipsis_middle(self):
        """Test middle ellipsis: x[i, ..., j]"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            M, N = x.shape[0], x.shape[-1]
            for i in hl.grid(M):
                for j in hl.grid(N):
                    x[i, ..., j] = 42.0
            return x

        x = torch.zeros([4, 8, 16], device=DEVICE)
        result = kernel(x)

        expected = torch.full([4, 8, 16], 42.0, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_ellipsis_bare(self):
        """Test bare ellipsis: x[...]"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            for _ in hl.grid(1):
                x[...] = 7.0
            return x

        x = torch.zeros([4, 8], device=DEVICE)
        result = kernel(x)

        expected = torch.full([4, 8], 7.0, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_ellipsis_with_none(self):
        """Test ellipsis with None (newaxis): x[None, ..., i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            N = x.shape[-1]
            for i in hl.grid(N):
                x[None, ..., i] = 1.0
            return x

        x = torch.zeros([4, 8], device=DEVICE)
        result = kernel(x)

        expected = torch.ones([4, 8], device=DEVICE)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager("Type inference errors are not raised in ref eager mode")
    def test_ellipsis_multiple_error(self):
        """Multiple ellipses should raise an error"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            for _ in hl.grid(1):
                x[..., ...] = 1.0
            return x

        x = torch.zeros([4, 8], device=DEVICE)
        with self.assertRaisesRegex(
            exc.TypeInferenceError,
            r"an index can only have a single ellipsis",
        ):
            code_and_output(kernel, (x,))

    @skipIfRefEager("Type inference errors are not raised in ref eager mode")
    def test_ellipsis_over_indexing_error(self):
        """Too many indices with ellipsis should raise an error"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            M, N = x.shape
            for i in hl.grid(M):
                for j in hl.grid(N):
                    for k in hl.grid(1):
                        x[i, j, k, ...] = 1.0
            return x

        x = torch.zeros([4, 8], device=DEVICE)
        with self.assertRaisesRegex(
            exc.TypeInferenceError,
            r"too many indices for tensor of dimension 2",
        ):
            code_and_output(kernel, (x,))

    @skipIfNormalMode(
        "RankMismatch: Cannot assign a tensor of rank 2 to a buffer of rank 3"
    )
    def test_multi_dim_slice(self):
        """Test both setter from scalar and getter for [:, :, i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[-1]
            for i in hl.grid(N):
                dst[:, :, i] = 1.0  # Test setter with scalar
                src[:, :, i] = dst[:, :, i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([2, 3, N], device=DEVICE)
        dst = torch.zeros([2, 3, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements should be ones after the kernel
        expected_src = torch.ones([2, 3, N], device=DEVICE)
        expected_dst = torch.ones([2, 3, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_tensor_value(self):
        """Test both setter from tensor value and getter for [i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor, val: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i] = val  # Test setter with tensor value
                src[i] = dst[i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([N, 4], device=DEVICE)
        dst = torch.zeros([N, 4], device=DEVICE)
        val = torch.ones([4], device=DEVICE)

        src_result, dst_result = kernel(src, dst, val)

        # All rows should be equal to val
        expected_src = val.expand(N, -1)
        expected_dst = val.expand(N, -1)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_tensor_value_3d(self):

        @helion.kernel(autotune_effort="none")
        def kernel(dst: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
            N = dst.shape[0]
            for i in hl.grid(N):
                dst[i] = val
            return dst

        N = 8
        dst = torch.zeros([N, 3, 4], device=DEVICE)
        val = torch.ones([3, 4], device=DEVICE)

        result = kernel(dst, val)

        expected = val.expand(N, -1, -1)
        torch.testing.assert_close(result, expected)

    def test_slice_to_slice(self):
        """buf[:] = zeros[:] - Full slice to slice assignment"""

        @helion.kernel(autotune_effort="none")
        def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
            N = buf.shape[0]
            for _ in hl.grid(N):
                buf[:] = zeros[:]
            return buf

        N = 128
        buf = torch.ones([N], device=DEVICE)
        zeros = torch.zeros([N], device=DEVICE)

        result = kernel(buf.clone(), zeros)
        expected = torch.zeros([N], device=DEVICE)
        torch.testing.assert_close(result, expected)

    @xfailIfPallas("slice-based stores not yet supported")
    def test_partial_slice(self):
        """Test both setter and getter for partial slices [:n] and [n:]"""

        @helion.kernel(autotune_effort="none")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, :16] = src[tile, :16]
                dst[tile, 16:] = src[tile, 16:]
            return dst

        N = 64
        src = torch.randn([N, 32], device=DEVICE)
        dst = torch.zeros([N, 32], device=DEVICE)
        result = kernel(src, dst)
        torch.testing.assert_close(result, src)

    @xfailIfPallas("slice-based stores not yet supported")
    def test_partial_slice_dim0(self):
        """Test partial slices on dim 0 (the tiled dimension)"""

        @helion.kernel(autotune_effort="none")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(1)):
                dst[:32, tile] = src[:32, tile]
                dst[32:, tile] = src[32:, tile]
            return dst

        src = torch.randn([64, 32], device=DEVICE)
        dst = torch.zeros([64, 32], device=DEVICE)
        result = kernel(src, dst)
        torch.testing.assert_close(result, src)

    @xfailIfPallas("slice-based stores not yet supported")
    def test_partial_slice_unaligned(self):
        """Test non-power-of-2 slice boundary for load and store"""

        @helion.kernel(autotune_effort="none")
        def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(src.size(0)):
                dst[tile, :13] = src[tile, :13]
            return dst

        src = torch.randn([16, 16], device=DEVICE)
        dst = torch.zeros([16, 13], device=DEVICE)
        result = kernel(src, dst)
        torch.testing.assert_close(result, src[:, :13])

    @xfailIfPallas("slice-based stores not yet supported")
    def test_partial_slice_unaligned_multi(self):
        """Test multiple non-power-of-2 slices in one kernel"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor, out: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for tile in hl.tile(src.size(0)):
                dst[tile, :13] = 1.0
                dst[tile, 13:] = 2.0
                out[tile, :] = src[tile, :13]
            return dst, out

        src = torch.randn([16, 16], device=DEVICE)
        dst = torch.zeros([16, 16], device=DEVICE)
        out = torch.zeros([16, 13], device=DEVICE)
        dst_result, out_result = kernel(src, dst, out)
        expected_dst = torch.zeros([16, 16], device=DEVICE)
        expected_dst[:, :13] = 1.0
        expected_dst[:, 13:] = 2.0
        torch.testing.assert_close(dst_result, expected_dst)
        torch.testing.assert_close(out_result, src[:, :13])

    @xfailIfPallas("slice-based stores not yet supported")
    def test_partial_slice_concat(self):
        """Test concat via full-slice load + partial-slice store"""

        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [x.size(0), x.size(1) + y.size(1)],
                device=x.device,
                dtype=x.dtype,
            )
            n1 = x.size(1)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :n1] = x[tile_m, :]
                out[tile_m, n1:] = y[tile_m, :]
            return out

        x = torch.randn([32, 16], device=DEVICE)
        y = torch.randn([32, 24], device=DEVICE)
        result = kernel(x, y)
        expected = torch.cat([x, y], dim=1)
        torch.testing.assert_close(result, expected)

    def test_broadcast(self):
        """Test both setter from scalar and getter for [:, i]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[1]
            for i in hl.grid(N):
                dst[:, i] = 1.0  # Test setter with scalar (broadcast)
                src[:, i] = dst[:, i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([N, N], device=DEVICE)
        dst = torch.zeros([N, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements should be ones after the kernel
        expected_src = torch.ones([N, N], device=DEVICE)
        expected_dst = torch.ones([N, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfNormalMode("InternalError: Unexpected type <class 'slice'>")
    def test_range_slice(self):
        """Test both setter from scalar and getter for [10:20]"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for _ in hl.grid(1):
                dst[10:20] = 1.0  # Test setter with scalar
                src[10:20] = dst[10:20]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Only indices 10:20 should be ones
        expected_src = torch.zeros([N], device=DEVICE)
        expected_src[10:20] = 1.0
        expected_dst = expected_src.clone()
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @xfailIfCute("incorrect results on cute backend")
    def test_range_slice_dynamic(self):
        """Test both [i:i+1] = scalar and [i] = [i:i+1] patterns"""

        @helion.kernel(autotune_effort="none")
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N - 1):
                dst[i : i + 1] = 1.0  # Test setter with scalar to slice
                src[i] = dst[i : i + 1]  # Test getter from slice to index
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements except last should be ones
        expected_src = torch.ones([N], device=DEVICE)
        expected_src[-1] = 0.0  # Last element not modified since loop goes to N-1
        expected_dst = expected_src.clone()

        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_tile_with_offset_pointer(self):
        """Test Tile+offset with pointer indexing"""

        @helion.kernel()
        def tile_offset_kernel(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty(x.size(0) - 10)
            for tile in hl.tile(out.size(0)):
                # Use tile + offset pattern
                tile_offset = tile + 10
                out[tile] = x[tile_offset]
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            tile_offset_kernel,
            (x,),
            indexing="pointer",
            block_size=32,
        )
        torch.testing.assert_close(result, x[10:])

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_tile_with_offset_block_ptr(self):
        """Test Tile+offset with block_ptr indexing"""

        @helion.kernel()
        def tile_offset_kernel(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty(x.size(0) - 10)
            for tile in hl.tile(out.size(0)):
                # Use tile + offset pattern
                tile_offset = tile + 10
                out[tile] = x[tile_offset]
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            tile_offset_kernel,
            (x,),
            indexing="block_ptr",
            block_size=32,
        )
        torch.testing.assert_close(result, x[10:])

    @onlyBackends(["triton"])
    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() == "tl._experimental_make_tensor_descriptor",
        "Experimental descriptors keep a per-dim block minimum",
    )
    def test_tensor_descriptor_batch_block_under_16_bytes(self):
        # Leading batch dims of the box need not span 16 bytes.
        @helion.kernel(static_shapes=True)
        def copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_a, tile_b, tile_c in hl.tile(x.size()):
                out[tile_a, tile_b, tile_c] = x[tile_a, tile_b, tile_c]
            return out

        x = torch.randint(0, 255, [4, 16, 64], dtype=torch.uint8, device=DEVICE)
        code, result = code_and_output(
            copy, (x,), block_sizes=[2, 16, 64], indexing="tensor_descriptor"
        )
        torch.testing.assert_close(result, x)
        self.assertIn(get_tensor_descriptor_fn_name(), code)

    @skipUnlessTensorDescriptor("TensorDescriptor not supported")
    @skipIfTileIR(
        "TileIR does not support descriptor with index not multiple of tile size"
    )
    def test_tile_with_offset_tensor_descriptor(self):
        """Test Tile+offset with tensor_descriptor indexing for 2D tensors"""

        @helion.kernel()
        def tile_offset_2d_kernel(x: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = x.new_empty(M - 10, N)
            for tile_m in hl.tile(out.size(0)):
                # Use tile + offset pattern
                tile_offset = tile_m + 10
                out[tile_m, :] = x[tile_offset, :]
            return out

        x = torch.randn([128, 64], device=DEVICE)
        code, result = code_and_output(
            tile_offset_2d_kernel,
            (x,),
            indexing="tensor_descriptor",
            block_size=32,
        )
        torch.testing.assert_close(result, x[10:, :])

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    @pytest.mark.xfail(
        _get_backend() == "cute",
        reason="CuTe attention dot lowering with tile-offset K/V loads is incorrect",
        run=False,
    )
    def test_tile_with_offset_from_expr(self):
        @helion.kernel(
            autotune_effort="none",
            static_shapes=True,
        )
        def attention(
            q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B, H, M, D = q_in.shape
            Bk, Hk, N, Dk = k_in.shape
            Bv, Hv, Nv, Dv = v_in.shape
            D = hl.specialize(D)
            Dv = hl.specialize(Dv)
            q = q_in.reshape(-1, D)
            k = k_in.reshape(-1, D)
            v = v_in.reshape(-1, Dv)
            MM = q.shape[0]
            o = q.new_empty(MM, Dv)
            lse = q.new_empty(MM, dtype=torch.float32)
            block_m = hl.register_block_size(M)
            block_n = hl.register_block_size(N)
            sm_scale = 1.0 / math.sqrt(D)
            qk_scale = sm_scale * 1.44269504  # 1/log(2)
            for tile_m in hl.tile(MM, block_size=block_m):
                m_i = hl.zeros([tile_m]) - float("inf")
                l_i = hl.zeros([tile_m]) + 1.0
                acc = hl.zeros([tile_m, Dv])
                q_i = q[tile_m, :]

                start_N = tile_m.begin // M * N
                for tile_n in hl.tile(0, N, block_size=block_n):
                    k_j = k[tile_n + start_N, :]
                    v_j = v[tile_n + start_N, :]
                    qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, None]
                    p = torch.exp2(qk)
                    alpha = torch.exp2(m_i - m_ij)
                    l_ij = torch.sum(p, -1)
                    acc = acc * alpha[:, None]
                    p = p.to(v.dtype)
                    acc = hl.dot(p, v_j, acc=acc)
                    l_i = l_i * alpha + l_ij
                    m_i = m_ij

                m_i += torch.log2(l_i)
                acc = acc / l_i[:, None]
                lse[tile_m] = m_i
                o[tile_m, :] = acc

            return o.reshape(B, H, M, Dv), lse.reshape(B, H, M)

        z, h, n_ctx, head_dim = 4, 32, 64, 64
        dtype = torch.bfloat16
        q, k, v = [
            torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=DEVICE)
            for _ in range(3)
        ]
        code, (o, lse) = code_and_output(attention, (q, k, v))
        torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(o, torch_out, atol=1e-2, rtol=1e-2)

    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @skipUnlessBlockPtr("asserts tl.make_block_ptr in the generated code")
    def test_per_load_indexing(self):
        @helion.kernel
        def multi_load_kernel(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> torch.Tensor:
            m, n = a.shape
            out = torch.empty_like(a)
            for tile_m, tile_n in hl.tile([m, n]):
                val_a = a[tile_m, tile_n]
                val_b = b[tile_m, tile_n]
                val_c = c[tile_m, tile_n]
                out[tile_m, tile_n] = val_a + val_b + val_c
            return out

        m, n = 64, 64
        a = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        c = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # 3 loads + 1 store = 4 operations
        code, result = code_and_output(
            multi_load_kernel,
            (a, b, c),
            indexing=["pointer", "pointer", "block_ptr", "pointer"],
            block_size=[16, 16],
        )
        expected = a + b + c
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        if _get_backend() == "triton":
            self.assertIn("tl.load", code)
            self.assertIn("tl.make_block_ptr", code)

    def test_per_load_indexing_backward_compat(self):
        @helion.kernel
        def many_loads_kernel(a: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            out = torch.empty_like(a)
            for tile_m, tile_n in hl.tile([m, n]):
                v1 = a[tile_m, tile_n]
                v2 = a[tile_m, tile_n]
                v3 = a[tile_m, tile_n]
                out[tile_m, tile_n] = v1 + v2 + v3
            return out

        m, n = 64, 64
        a = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        expected = a + a + a

        # When indexing is not specified (empty list), all loads and stores default to pointer
        code1, result = code_and_output(
            many_loads_kernel,
            (a,),
            block_size=[16, 16],
        )
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        # Single string: backward compatible mode, all loads and stores use the same strategy
        code2, result = code_and_output(
            many_loads_kernel,
            (a,),
            indexing="pointer",
            block_size=[16, 16],
        )
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        # List: per-operation mode, must provide strategy for all loads and stores (3 loads + 1 store)
        code3, result = code_and_output(
            many_loads_kernel,
            (a,),
            indexing=["pointer", "pointer", "pointer", "pointer"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        self.assertEqual(code1, code2)
        self.assertEqual(code2, code3)

    @skipIfRefEager("needs debugging")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @skipUnlessBlockPtr("asserts tl.make_block_ptr in the generated code")
    def test_per_load_and_store_indexing(self):
        """Test that both loads and stores can have independent indexing strategies."""

        @helion.kernel
        def load_store_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            out = torch.empty_like(a)
            for tile_m, tile_n in hl.tile([m, n]):
                # 2 loads
                val_a = a[tile_m, tile_n]
                val_b = b[tile_m, tile_n]
                # 1 store
                out[tile_m, tile_n] = val_a + val_b
            return out

        m, n = 64, 64
        a = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        expected = a + b

        # Test 1: Mixed strategies - pointer loads, block_ptr store
        # (2 loads + 1 store = 3 operations)
        code1, result1 = code_and_output(
            load_store_kernel,
            (a, b),
            indexing=["pointer", "pointer", "block_ptr"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result1, expected, rtol=1e-3, atol=1e-3)
        if _get_backend() == "triton":
            # Verify we have both pointer loads and block_ptr store
            self.assertIn("tl.load", code1)
            self.assertIn("tl.make_block_ptr", code1)
            # Count occurrences: should have block_ptr for store
            self.assertEqual(code1.count("tl.make_block_ptr"), 1)

        # Test 2: Different mix - block_ptr loads, pointer store
        code2, result2 = code_and_output(
            load_store_kernel,
            (a, b),
            indexing=["block_ptr", "block_ptr", "pointer"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result2, expected, rtol=1e-3, atol=1e-3)
        if _get_backend() == "triton":
            # Should have 2 block_ptrs for loads, regular store
            self.assertEqual(code2.count("tl.make_block_ptr"), 2)

        # Test 3: All block_ptr
        code3, result3 = code_and_output(
            load_store_kernel,
            (a, b),
            indexing=["block_ptr", "block_ptr", "block_ptr"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result3, expected, rtol=1e-3, atol=1e-3)
        if _get_backend() == "triton":
            # Should have 3 block_ptrs total (2 loads + 1 store)
            self.assertEqual(code3.count("tl.make_block_ptr"), 3)

        # Test 4: Verify single string applies to all loads and stores
        code4, result4 = code_and_output(
            load_store_kernel,
            (a, b),
            indexing="block_ptr",
            block_size=[16, 16],
        )
        torch.testing.assert_close(result4, expected, rtol=1e-3, atol=1e-3)
        # Should match the all-block_ptr version
        self.assertEqual(code3, code4)

    def test_indirect_indexing_2d_direct_gather(self):
        @helion.kernel()
        def test(
            col: torch.Tensor,  # [M, K] int64
            val: torch.Tensor,  # [M, K] fp32
            B: torch.Tensor,  # [K, N] fp32
        ) -> torch.Tensor:  # [M, N] fp32
            M, K = col.shape
            _, N = B.shape
            out_dtype = torch.promote_types(val.dtype, B.dtype)
            C = torch.empty((M, N), dtype=out_dtype, device=B.device)

            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

                for tile_k in hl.tile(K):
                    cols_2d = col[tile_m, tile_k]
                    B_slice = B[cols_2d[:, :, None], tile_n.index[None, None, :]]
                    vals_2d = val[tile_m, tile_k]
                    contrib = vals_2d[:, :, None] * B_slice
                    contrib = contrib.sum(dim=1)
                    acc = acc + contrib

                C[tile_m, tile_n] = acc.to(out_dtype)

            return C

        M, K, N = 32, 16, 24
        col = torch.randint(0, K, (M, K), device=DEVICE, dtype=torch.int64)
        val = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            test,
            (col, val, B),
            block_size=[8, 8, 4],
        )

        expected = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    expected[i, j] += val[i, k] * B[col[i, k], j]

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_indirect_indexing_2d_flat_load(self):
        @helion.kernel()
        def test(
            col: torch.Tensor,  # [M, K] int64
            val: torch.Tensor,  # [M, K] fp32
            B: torch.Tensor,  # [K, N] fp32
        ) -> torch.Tensor:  # [M, N] fp32
            M, K = col.shape
            _, N = B.shape
            out_dtype = torch.promote_types(val.dtype, B.dtype)
            C = torch.empty((M, N), dtype=out_dtype, device=B.device)
            B_flat = B.reshape(-1)  # [K*N]

            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

                for tile_k in hl.tile(K):
                    cols_2d = col[tile_m, tile_k]
                    B_indices = (cols_2d * N)[:, :, None] + tile_n.index[None, None, :]
                    B_slice = hl.load(B_flat, [B_indices])
                    vals_2d = val[tile_m, tile_k]
                    contrib = vals_2d[:, :, None] * B_slice
                    contrib = contrib.sum(dim=1)
                    acc = acc + contrib

                C[tile_m, tile_n] = acc.to(out_dtype)

            return C

        M, K, N = 32, 16, 24
        col = torch.randint(0, K, (M, K), device=DEVICE, dtype=torch.int64)
        val = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            test,
            (col, val, B),
            block_size=[8, 8, 4],
        )

        expected = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    expected[i, j] += val[i, k] * B[col[i, k], j]

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_indirect_indexing_3d_direct_gather(self):
        @helion.kernel()
        def test(
            col: torch.Tensor,  # [M, N, K] int64 - indices for first dimension of B
            val: torch.Tensor,  # [M, N, K] fp32 - values to multiply
            B: torch.Tensor,  # [K, P, Q] fp32 - tensor to index into
        ) -> torch.Tensor:  # [M, N, P, Q] fp32
            M, N, K = col.shape
            _, P, Q = B.shape
            out_dtype = torch.promote_types(val.dtype, B.dtype)
            C = torch.empty((M, N, P, Q), dtype=out_dtype, device=B.device)

            for tile_m, tile_n, tile_p, tile_q in hl.tile([M, N, P, Q]):
                acc = hl.zeros([tile_m, tile_n, tile_p, tile_q], dtype=torch.float32)

                for tile_k in hl.tile(K):
                    cols_3d = col[tile_m, tile_n, tile_k]
                    B_slice = B[
                        cols_3d[:, :, :, None, None],
                        tile_p.index[None, None, :, None],
                        tile_q.index[None, None, None, :],
                    ]

                    vals_3d = val[tile_m, tile_n, tile_k]
                    contrib = vals_3d[:, :, :, None, None] * B_slice
                    contrib = contrib.sum(dim=2)
                    acc = acc + contrib

                C[tile_m, tile_n, tile_p, tile_q] = acc.to(out_dtype)
            return C

        M, N, K, P, Q = 16, 12, 8, 10, 14
        col = torch.randint(0, K, (M, N, K), device=DEVICE, dtype=torch.int64)
        val = torch.rand((M, N, K), device=DEVICE, dtype=torch.float32)
        B = torch.rand((K, P, Q), device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            test,
            (col, val, B),
            block_size=[4, 4, 4, 4, 4],  # 5D tiling for M, N, P, Q, K
        )

        expected = (val[..., None, None] * B[col]).sum(dim=2)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_indirect_indexing_3d_flat_load(self):
        @helion.kernel()
        def test(
            col: torch.Tensor,  # [M, N, K] int64
            val: torch.Tensor,  # [M, N, K] fp32
            B: torch.Tensor,  # [K, P, Q] fp32
        ) -> torch.Tensor:  # [M, N, P, Q] fp32
            M, N, K = col.shape
            _, P, Q = B.shape
            out_dtype = torch.promote_types(val.dtype, B.dtype)
            C = torch.empty((M, N, P, Q), dtype=out_dtype, device=B.device)
            B_flat = B.reshape(-1)  # [K*P*Q]

            for tile_m, tile_n, tile_p, tile_q in hl.tile([M, N, P, Q]):
                acc = hl.zeros([tile_m, tile_n, tile_p, tile_q], dtype=torch.float32)

                for tile_k in hl.tile(K):
                    cols_3d = col[tile_m, tile_n, tile_k]
                    B_indices = (
                        cols_3d[:, :, :, None, None] * (P * Q)
                        + tile_p.index[None, None, :, None] * Q
                        + tile_q.index[None, None, None, :]
                    )
                    B_slice = hl.load(B_flat, [B_indices])
                    vals_3d = val[tile_m, tile_n, tile_k]
                    contrib = vals_3d[:, :, :, None, None] * B_slice
                    contrib = contrib.sum(dim=2)
                    acc = acc + contrib

                C[tile_m, tile_n, tile_p, tile_q] = acc.to(out_dtype)
            return C

        M, N, K, P, Q = 16, 12, 8, 10, 14
        col = torch.randint(0, K, (M, N, K), device=DEVICE, dtype=torch.int64)
        val = torch.rand((M, N, K), device=DEVICE, dtype=torch.float32)
        B = torch.rand((K, P, Q), device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            test,
            (col, val, B),
            block_size=[4, 4, 4, 4, 4],
        )

        expected = (val[..., None, None] * B[col]).sum(dim=2)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_tile_index_floor_div(self):
        """Test tile.index // divisor pattern used in MXFP8 dequantization.

        This tests the case where tile.index is divided to index into a scale
        tensor that has fewer elements than the data tensor.
        """
        BLOCK_SIZE = 32

        @helion.kernel
        def dequant_with_scale(
            x_data: torch.Tensor,
            x_scale: torch.Tensor,
            block_size: hl.constexpr,
        ) -> torch.Tensor:
            m, n = x_data.shape
            out = torch.empty_like(x_data)

            for m_tile, n_tile in hl.tile([m, n]):
                data = x_data[m_tile, n_tile]
                # Use floor division to index into scale
                scale = x_scale[m_tile, n_tile.index // block_size]
                out[m_tile, n_tile] = data * scale

            return out

        # Test case: n_data = 256, n_scale = 8 (256 / 32)
        m, n_data = 128, 256
        n_scale = n_data // BLOCK_SIZE

        x_data = torch.randn((m, n_data), device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn((m, n_scale), device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            dequant_with_scale,
            (x_data, x_scale, BLOCK_SIZE),
            block_size=[8, 64],
        )

        # Expected: each scale value applies to BLOCK_SIZE consecutive elements
        expanded_scale = x_scale.repeat_interleave(BLOCK_SIZE, dim=-1)
        expected = x_data * expanded_scale

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_tile_index_floor_div_block_larger_than_dim(self):
        """Test tile.index // divisor when block_size > actual dimension.

        This tests the edge case where the configured block_size is larger
        than the actual tensor dimension, with the scale tensor having only
        1 column.
        """
        BLOCK_SIZE = 32

        failing_config = helion.Config(
            block_sizes=[8, 256],  # block_size[1]=256 > n=32
            indexing=["pointer", "pointer", "pointer"],
            l2_groupings=[1],
            loop_orders=[[1, 0]],
            num_stages=2,
            num_warps=2,
            pid_type="flat",
        )

        @helion.kernel(config=failing_config)
        def dequant_with_scale_large_block(
            x_data: torch.Tensor,
            x_scale: torch.Tensor,
            block_size: hl.constexpr,
        ) -> torch.Tensor:
            m, n = x_data.shape
            out = torch.empty_like(x_data)

            for m_tile, n_tile in hl.tile([m, n]):
                data = x_data[m_tile, n_tile]
                # Use floor division to index into scale
                scale = x_scale[m_tile, n_tile.index // block_size]
                out[m_tile, n_tile] = data * scale

            return out

        # Test case: n_data = 32, n_scale = 1 (32 / 32)
        # block_size[1] = 256 is larger than n_data = 32
        m, n_data = 128, 32
        n_scale = n_data // BLOCK_SIZE

        x_data = torch.randn((m, n_data), device=DEVICE, dtype=torch.float32)
        x_scale = torch.randn((m, n_scale), device=DEVICE, dtype=torch.float32)

        result = dequant_with_scale_large_block(x_data, x_scale, BLOCK_SIZE)

        # Expected: each scale value applies to BLOCK_SIZE consecutive elements
        expanded_scale = x_scale.repeat_interleave(BLOCK_SIZE, dim=-1)
        expected = x_data * expanded_scale

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @skipIfRefEager("Test requires dynamic shapes masking")
    def test_indexed_store_mask_propagation(self):
        """Test that indexed stores with broadcast tensor subscripts propagate masks correctly.

        This tests the fix for a bug where stores like:
            dx[tile_m.index[:, None], indices[tile_m, :]] = dy[tile_m, :]
        would have None as the mask instead of propagating the tile's mask.

        The issue was that when block_id is 0, the condition
        `(bid := env.get_block_id(...))` would evaluate to False because
        0 is falsy in Python. The fix is to check `is not None` explicitly.
        """

        @helion.kernel(static_shapes=False)
        def scatter_kernel(
            dy: torch.Tensor,
            indices: torch.Tensor,
            input_shape: list[int],
            k: int,
        ) -> torch.Tensor:
            dx = dy.new_zeros(*input_shape)
            k = hl.specialize(k)
            dx = dx.reshape(-1, dx.shape[-1])
            dy = dy.reshape(-1, k)
            indices = indices.reshape(-1, k)
            for tile_m in hl.tile(dy.shape[0]):
                # This pattern uses tile_m.index[:, None] as a 2D tensor subscript
                # which should propagate the tile's mask to the store
                dx[tile_m.index[:, None], indices[tile_m, :]] = dy[tile_m, :]
            return dx.view(input_shape)

        # Test with unique indices to avoid race conditions
        dy = torch.randn(5, 8, device=DEVICE)
        idx = torch.arange(8, device=DEVICE).unsqueeze(0).expand(5, 8).contiguous()

        code, result = code_and_output(
            scatter_kernel,
            (dy, idx, (5, 20), 8),
            block_size=[2],
        )

        if _get_backend() == "triton":
            # Verify the mask is present in the store (not None)
            self.assertIn("tl.store", code)
            # The mask should be something like mask_0[:, None], not None
            self.assertNotIn(
                "tl.store(dx + (load_1 * dx_stride_0 + load_2 * dx_stride_1), load, None)",
                code,
            )

        # Compute expected result
        expected = torch.zeros(5, 20, device=DEVICE)
        for i in range(5):
            for j in range(8):
                expected[i, idx[i, j]] = dy[i, j]

        torch.testing.assert_close(result, expected)

    def test_non_consecutive_tensor_indexers_no_broadcast(self):
        """Test that non-consecutive tensor indexers don't get incorrectly broadcast.

        The issue was that when tensor indexers are not consecutive (separated by
        other index types like tile.index or SymInt), they were still being
        broadcast together, causing incorrect dimension ordering.
        """

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def store_with_mixed_indices(
            tensor_idx: torch.Tensor,
            data: torch.Tensor,
            k: int,
        ) -> torch.Tensor:
            m, n = data.size()
            k = hl.specialize(k)
            out = torch.zeros([m, m, k], device=data.device, dtype=data.dtype)

            # Use explicit block_size to ensure consistent behavior in both modes
            for tile_m in hl.tile(m, block_size=4):
                # Store 3D data into out[tensor_idx[tile_m], tile_m.index, :]
                val = hl.load(data, [tile_m, hl.arange(k, dtype=torch.int32)])
                val_3d = val[:, None, :].expand(val.size(0), val.size(0), k)
                hl.store(
                    out,
                    [tensor_idx[tile_m], tile_m.index, hl.arange(k, dtype=torch.int32)],
                    val_3d,
                )

            return out

        M = 8
        K = 16
        block_size = 4
        tensor_idx = torch.arange(M, device=DEVICE, dtype=torch.int32)
        data = torch.randn(M, K, device=DEVICE)

        code, result = code_and_output(
            store_with_mixed_indices,
            (tensor_idx, data, K),
        )

        # Verify the result is correct
        # The kernel stores at out[tensor_idx[tile_m], tile_m.index, :] = val_3d
        # With explicit block_size=4, tile_m iterates in chunks: [0:4], [4:8]
        # tile_m.index returns global indices, so stores happen in diagonal blocks
        expected = torch.zeros([M, M, K], device=DEVICE)
        for tile_start in range(0, M, block_size):
            tile_end = tile_start + block_size
            expected[tile_start:tile_end, tile_start:tile_end, :] = (
                data[tile_start:tile_end, :]
                .unsqueeze(1)
                .expand(block_size, block_size, K)
            )
        torch.testing.assert_close(result, expected)

    def test_mixed_scalar_block_store_size1_dim(self):
        """Test store with mixed scalar/block indexing when block dimension has size 1.

        This tests a bug fix where storing a block value with:
        - One index being a tile/block (e.g., m_tile) over a size-1 dimension
        - Another index being a scalar (e.g., computed from tile.begin)
        would generate invalid Triton code because the pointer became scalar
        but the value was still a block.
        """

        @helion.kernel(autotune_effort="none")
        def kernel_with_mixed_store(
            x_data: torch.Tensor, BLOCK_SIZE: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x_data.shape
            n = hl.specialize(n)
            n_scale_cols = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            scales = x_data.new_empty((m, n_scale_cols), dtype=torch.uint8)
            out = x_data.new_empty(x_data.shape, dtype=torch.float32)

            n_block = hl.register_block_size(BLOCK_SIZE, n)

            for m_tile, n_tile in hl.tile([m, n], block_size=[None, n_block]):
                for n_tile_local in hl.tile(
                    n_tile.begin, n_tile.end, block_size=BLOCK_SIZE
                ):
                    x_block = x_data[m_tile, n_tile_local]

                    # Compute one value per row in m_tile
                    row_max = x_block.abs().amax(dim=1)
                    row_value = row_max.to(torch.uint8)

                    out[m_tile, n_tile_local] = x_block * 2.0

                    # Mixed indexing: block row index + scalar column index
                    scale_col_idx = n_tile_local.begin // BLOCK_SIZE  # scalar
                    scales[m_tile, scale_col_idx] = row_value  # row_value is block

            return out, scales

        # Test with m=1 (single row - this was the failing case before the fix)
        # The fix ensures tl.reshape is applied to squeeze the value to scalar
        # when the pointer is scalar due to size-1 dimensions being dropped.
        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        code, (out1, scales1) = code_and_output(kernel_with_mixed_store, (x1, 32))
        expected_out1 = x1 * 2.0
        torch.testing.assert_close(out1, expected_out1)
        self.assertEqual(scales1.shape, (1, 2))

    @skipIfTileIR("TileIR does not support gather operation")
    def test_gather_2d_dim1(self):
        @helion.kernel()
        def test_gather(
            input_tensor: torch.Tensor,  # [N, M]
            index_tensor: torch.Tensor,  # [N, K]
        ) -> torch.Tensor:  # [N, K]
            N = input_tensor.size(0)
            K = index_tensor.size(1)
            out = torch.empty(
                [N, K], dtype=input_tensor.dtype, device=input_tensor.device
            )
            for tile_n, tile_k in hl.tile([N, K]):
                # Input sliced on non-gather dim to match index's first dim
                out[tile_n, tile_k] = torch.gather(
                    input_tensor[tile_n, :], 1, index_tensor[tile_n, tile_k]
                )
            return out

        N, M, K = 16, 32, 8
        input_tensor = torch.randn(N, M, device=DEVICE, dtype=torch.float32)
        index_tensor = torch.randint(0, M, (N, K), device=DEVICE, dtype=torch.int64)

        code, result = code_and_output(
            test_gather, (input_tensor, index_tensor), block_size=[4, 4]
        )
        expected = torch.gather(input_tensor, 1, index_tensor)

        torch.testing.assert_close(result, expected)

    @skipIfTileIR("TileIR does not support gather operation")
    def test_gather_2d_dim0(self):
        @helion.kernel()
        def test_gather(
            input_tensor: torch.Tensor,  # [N, M]
            index_tensor: torch.Tensor,  # [K, M]
        ) -> torch.Tensor:  # [K, M]
            K = index_tensor.size(0)
            M = input_tensor.size(1)
            out = torch.empty(
                [K, M], dtype=input_tensor.dtype, device=input_tensor.device
            )
            for tile_k, tile_m in hl.tile([K, M]):
                # Input sliced on non-gather dim to match index's second dim
                out[tile_k, tile_m] = torch.gather(
                    input_tensor[:, tile_m], 0, index_tensor[tile_k, tile_m]
                )
            return out

        N, M, K = 16, 32, 8
        input_tensor = torch.randn(N, M, device=DEVICE, dtype=torch.float32)
        index_tensor = torch.randint(0, N, (K, M), device=DEVICE, dtype=torch.int64)

        code, result = code_and_output(
            test_gather, (input_tensor, index_tensor), block_size=[4, 8]
        )
        expected = torch.gather(input_tensor, 0, index_tensor)

        torch.testing.assert_close(result, expected)

    def test_tile_index_with_none_dimension(self):
        """Test that tile.index[None, :] followed by slices produces correct shape.

        When using tile.index[None, :] as an indexer, the result should have
        a leading dimension of size 1, matching PyTorch's indexing behavior:
        - c.shape = [M, N]
        - idx = tile.index[None, :]  # shape [1, tile_size]
        - c[idx, :] should produce shape [1, tile_size, N]
        """

        @helion.kernel()
        def test_none_index_2d(
            c: torch.Tensor,  # [M, N]
        ) -> torch.Tensor:
            M, N = c.shape
            out = torch.empty([1, M, N], dtype=c.dtype, device=c.device)
            for tile_m in hl.tile(M):
                # idx has shape [1, tile_m_size]
                idx = tile_m.index[None, :]
                # c[idx, :] should have shape [1, tile_m_size, N] per PyTorch
                val = c[idx, :]
                # Store to output with same shape [1, tile_m_size, N]
                out[:, tile_m, :] = val
            return out

        c = torch.randn(32, 16, device=DEVICE)

        code, result = code_and_output(test_none_index_2d, (c,), block_size=8)
        expected = c.unsqueeze(0)  # [1, M, N]
        torch.testing.assert_close(result, expected)

    def test_tile_index_with_none_dimension_3d(self):
        """Test 3D version of tile.index[None, :] indexing."""

        @helion.kernel()
        def test_none_index_3d(
            c: torch.Tensor,  # [M, N, K]
        ) -> torch.Tensor:
            M, N, K = c.shape
            out = torch.empty([1, M, N, K], dtype=c.dtype, device=c.device)
            for tile_m in hl.tile(M):
                # idx has shape [1, tile_m_size]
                idx = tile_m.index[None, :]
                # c[idx, :, :] should have shape [1, tile_m_size, N, K]
                val = c[idx, :, :]
                out[:, tile_m, :, :] = val
            return out

        c = torch.randn(32, 16, 8, device=DEVICE)

        code, result = code_and_output(test_none_index_3d, (c,), block_size=8)
        expected = c.unsqueeze(0)  # [1, M, N, K]
        torch.testing.assert_close(result, expected)

    def test_loaded_tensor_as_index_with_slices(self):
        """Test that loaded 2D tensor indices with trailing slices produce correct shape.

        When loading indices from a tensor (2D result) and using them to index
        another tensor with trailing slices, the output should be 4D:
        - index_source.shape = [M, N]
        - data.shape = [X, Y, Z]
        - indices = index_source[t0, t1]  # shape [tile_t0, tile_t1]
        - data[indices, :, :] should produce shape [tile_t0, tile_t1, Y, Z]
        """

        @helion.kernel()
        def test_tensor_indices_with_slices(
            index_source: torch.Tensor,  # [M, N] tensor containing indices
            data: torch.Tensor,  # [X, Y, Z] tensor to index into
        ) -> torch.Tensor:
            m, n = index_source.shape
            x, y, z = data.shape
            out = torch.empty([m, n, y, z], dtype=data.dtype, device=data.device)
            for t0, t1 in hl.tile([m, n]):
                # Load indices from tensor - this gives a 2D result [tile_t0, tile_t1]
                indices = index_source[t0, t1]
                # Use those indices with trailing slices - should give 4D result
                result = data[indices, :, :]
                out[t0, t1, :, :] = result
            return out

        M, N = 4, 8
        X, Y, Z = 10, 20, 30

        # Create index source with valid indices into data's first dimension
        index_source = torch.randint(0, X, (M, N), device=DEVICE)
        data = torch.randn(X, Y, Z, device=DEVICE)

        code, result = code_and_output(
            test_tensor_indices_with_slices, (index_source, data), block_size=[4, 8]
        )
        expected = data[index_source, :, :]
        torch.testing.assert_close(result, expected)

    def test_full_slice_in_reduction_loop(self):
        """Full slice between two tiled dims: q[tile_n, :, tile_d]

        With static_shapes and equal dimensions (N=C=D=16), the C
        dimension appears as a plain int in tensor shapes.
        has_matmul_with_rdim must still detect the matmul uses C so
        the roller does not incorrectly roll the reduction.
        """

        @helion.kernel(static_shapes=True)
        def kernel(q: torch.Tensor) -> torch.Tensor:
            N = q.size(0)
            C = q.size(1)
            D = q.size(2)
            out = torch.empty([N, C], dtype=q.dtype, device=q.device)
            for (tile_n,) in hl.tile([N]):
                attn = hl.zeros([tile_n, C, C], dtype=torch.float32)
                for tile_d in hl.tile(D):
                    qt = q[tile_n, :, tile_d]
                    attn = torch.baddbmm(attn, qt, qt.transpose(-2, -1))
                out[tile_n, :] = attn.sum(-1).to(out.dtype)
            return out

        q = torch.randn(16, 16, 16, device=DEVICE)
        code, result = code_and_output(kernel, (q,), block_sizes=[16, 16])
        expected = torch.baddbmm(
            torch.zeros(16, 16, 16, device=DEVICE), q, q.transpose(-2, -1)
        ).sum(-1)
        torch.testing.assert_close(result, expected, atol=0.2, rtol=0.01)
        if _get_backend() == "triton":
            self.assertIn("tl.dot", code)

    def test_symbolic_index_in_host_block(self):
        """Regression test for https://github.com/pytorch/helion/issues/1339.

        Using out_offsets[n] (where n = size(0) - 1) in the host block should
        not specialize n to a concrete value, causing incorrect grid sizes and
        missing masking in the generated code.
        """

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def jagged_iota(out_offsets):
            n = out_offsets.size(0) - 1
            out = torch.zeros(out_offsets[n].item(), device=out_offsets.device)
            for tile_n in hl.tile(n):
                s = out_offsets[tile_n]
                e = out_offsets[tile_n + 1]
                lens = e - s
                max_len = lens.amax()

                for tile_l in hl.tile(max_len):
                    idx = tile_l.index[None, :] + s[:, None]
                    mask = tile_l.index[None, :] < lens[:, None]
                    hl.store(out, [idx], idx, extra_mask=mask)
            return out

        offsets = torch.tensor([0, 2, 3, 5, 7], device=DEVICE)

        # n=0: offsets[:1] has shape (1,). The host index remains symbolic;
        # CuTe additionally keys its structural ownership proof by metadata.
        result = jagged_iota(offsets[:1].clone())
        torch.testing.assert_close(
            result, torch.arange(0, dtype=torch.float32, device=DEVICE)
        )
        self.assertEqual(len(jagged_iota._bound_kernels), 1)

        for case_index, n in enumerate([1, 3, len(offsets) - 1], start=2):
            arg = offsets[: n + 1].clone()
            result = jagged_iota(arg)
            total = offsets[n].item()
            expected = torch.arange(total, dtype=torch.float32, device=DEVICE)
            torch.testing.assert_close(result, expected)
            expected_bindings = case_index if _get_backend() == "cute" else 1
            self.assertEqual(len(jagged_iota._bound_kernels), expected_bindings)

            # Changing only runtime offsets must reuse the bound program while
            # recomputing both the host allocation and device iteration bounds.
            bound = jagged_iota.bind((arg,))
            changed = arg * 3
            self.assertIs(jagged_iota.bind((changed,)), bound)
            result = jagged_iota(changed)
            torch.testing.assert_close(
                result, torch.arange(total * 3, dtype=torch.float32, device=DEVICE)
            )
            self.assertEqual(len(jagged_iota._bound_kernels), expected_bindings)

    @onlyBackends(["triton"])
    @skipIfRefEager("Test checks generated Triton code")
    def test_triton_do_not_specialize_emits_do_not_specialize(self):
        @helion.kernel(
            autotune_effort="none",
            static_shapes=False,
            triton_do_not_specialize=True,
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([1], device=DEVICE)
        code, result = code_and_output(add_one, (x,), block_size=16)
        torch.testing.assert_close(result, x + 1)
        self.assertIn("@triton.jit(", code)
        self.assertIn("'x_size_0'", code)
        self.assertIn("do_not_specialize=", code)
        self.assertIn("do_not_specialize_on_alignment=", code)
        y = torch.randn([2], device=DEVICE)
        torch.testing.assert_close(add_one(y), y + 1)
        self.assertEqual(len(add_one._bound_kernels), 1)

    @onlyBackends(["triton"])
    @skipIfRocm("ROCm exposes an unrelated cross-loop dependency in this codegen test")
    @skipIfTileIR("TileIR does not support cross-loop persistent synchronization")
    @skipIfXPU("XPU exposes an unrelated cross-loop dependency in this codegen test")
    @skipIfRefEager("Test checks generated Triton code")
    def test_dynamic_internal_strides_remain_literal(self):
        @helion.kernel(
            autotune_effort="none",
            static_shapes=False,
            triton_do_not_specialize=True,
        )
        def two_stage(x: torch.Tensor) -> torch.Tensor:
            rows = x.size(0)
            tmp = torch.empty((rows, 32), dtype=x.dtype, device=x.device)
            out = torch.empty_like(tmp)
            for tile_m, tile_n in hl.tile([rows, 32], block_size=[1, 32]):
                tmp[tile_m, tile_n] = x[tile_m, tile_n]
            for tile_m, tile_n in hl.tile([rows, 32], block_size=[1, 32]):
                out[tile_m, tile_n] = tmp[tile_m, tile_n] + 1
            return out

        x = torch.randn([2, 32], device=DEVICE)
        code, result = code_and_output(two_stage, (x,))
        torch.testing.assert_close(result, x + 1)
        # User-input layout remains generic, while compiler-owned contiguous
        # layouts must not pollute Triton's do-not-specialize set.
        self.assertIn("'x_stride_0'", code)
        self.assertNotIn("'tmp_stride_", code)
        self.assertNotIn("'out_stride_", code)

    @onlyBackends(["triton"])
    @skipIfRefEager("Test checks generated Triton code")
    def test_symbolic_internal_stride_remains_runtime(self):
        @helion.kernel(
            autotune_effort="none",
            static_shapes=False,
            triton_do_not_specialize=True,
        )
        def transpose_copy(x: torch.Tensor) -> torch.Tensor:
            rows = x.size(0)
            out = torch.empty((32, rows), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([rows, 32], block_size=[1, 32]):
                out[tile_n, tile_m] = x[tile_m, tile_n].T
            return out

        x = torch.randn([2, 32], device=DEVICE)
        code, result = code_and_output(transpose_copy, (x,))
        torch.testing.assert_close(result, x.T)
        self.assertIn("'out_stride_0'", code)

    @onlyBackends(["triton"])
    @skipIfRefEager("Test checks generated Triton code")
    def test_dynamic_size_args_match_triton_default(self):
        """Without triton_do_not_specialize, Helion follows Triton's own default
        and emits a plain @triton.jit so value/alignment specialization is
        preserved (which is what enables vectorized loads)."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([1024], device=DEVICE)
        code, result = code_and_output(add_one, (x,), block_size=16)
        torch.testing.assert_close(result, x + 1)
        self.assertNotIn("do_not_specialize=", code)
        self.assertNotIn("do_not_specialize_on_alignment=", code)
        # Helion still buckets sizes so a single BoundKernel handles every shape.
        y = torch.randn([2048], device=DEVICE)
        torch.testing.assert_close(add_one(y), y + 1)
        self.assertEqual(len(add_one._bound_kernels), 1)

    @onlyBackends(["cute", "triton"])
    def test_computed_tile_coordinates_with_singleton_views(self):
        x = torch.zeros(5, 65, device=DEVICE)
        _, (prefix, reused) = code_and_output(_computed_tile_coordinates, (x,))
        values = torch.arange(5 * 65, device=DEVICE, dtype=x.dtype).reshape(5, 65)
        expected = torch.cat([part.cumsum(-1) for part in values.split(32, -1)], -1)
        torch.testing.assert_close(prefix, expected, rtol=0, atol=0)
        torch.testing.assert_close(reused, values + 1, rtol=0, atol=0)

    @onlyBackends(["cute", "triton"])
    def test_indexed_store_preserves_earlier_aliased_writes(self):
        @helion.kernel(static_shapes=True)
        def initialize_and_select(
            slots: torch.Tensor, scores: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty(
                (slots.numel(), 2), device=slots.device, dtype=torch.int32
            )
            flat = out.reshape(-1)
            flat_scores = scores.reshape(-1)
            for row in hl.tile(slots.numel()):
                out[row, 0] = -1
                out[row, 1] = -1
                original = slots[row]
                gathered = flat_scores[row.index * scores.size(1) + original]
                chosen = torch.where(gathered >= 0, original, 1 - original)
                best = hl.full([row], -float("inf"), dtype=scores.dtype)
                for col in hl.tile(scores.size(1)):
                    offsets = (
                        row.index[:, None].to(torch.int64) * scores.size(1)
                        + col.index[None, :]
                    )
                    uniform = hl.rand([], seed=7524, offsets=offsets)
                    values = torch.where(uniform >= 0, scores[row, col], -float("inf"))
                    best = torch.maximum(best, values.amax(-1))
                flat[row.index * 2 + chosen] = best.to(torch.int32)
            return out

        # Exact tile multiples omit the ordinary row mask. The indexed store
        # must not acquire extra iterations from the tracing-time tile hint.
        slots = torch.arange(256, device=DEVICE, dtype=torch.int32) % 2
        expected = torch.full((256, 2), -1, device=DEVICE, dtype=torch.int32)
        scores = torch.arange(256 * 4, device=DEVICE, dtype=torch.float32).reshape(
            256, 4
        )
        expected.scatter_(
            1, slots.long()[:, None], scores.amax(-1).to(torch.int32)[:, None]
        )
        for rows_per_block in (16, 128):
            with self.subTest(rows_per_block=rows_per_block):
                _, output = code_and_output(
                    initialize_and_select,
                    (slots, scores),
                    block_sizes=[rows_per_block, 4],
                )
                torch.testing.assert_close(output, expected)

    def test_scalar_tensor_index_with_grid(self):
        """Index a tensor with a 0-dim scalar tensor from a grid load."""

        @helion.kernel(
            static_shapes=False,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def gather_kernel(
            data: torch.Tensor,  # [E, N]
            ids: torch.Tensor,  # [M]
        ) -> torch.Tensor:
            M = ids.shape[0]
            _E, N = data.shape
            N = hl.specialize(N)
            out = torch.empty(M, N, dtype=data.dtype, device=data.device)

            for grid_m in hl.grid(M):
                idx = ids[grid_m]  # 0-dim scalar tensor
                for tile_n in hl.tile(N):
                    out[grid_m, tile_n] = data[idx, tile_n]

            return out

        E, N, M = 8, 64, 16
        data = torch.randn(E, N, device=DEVICE, dtype=torch.float32)
        ids = (torch.arange(M, device=DEVICE) % E).to(torch.int32)

        code, result = code_and_output(gather_kernel, (data, ids), block_sizes=[64])
        expected = data[ids.long()]
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("mode", ["serial", "cooperative"])
def test_computed_tile_grid_ownership_codegen(mode):
    kernel = helion.kernel(
        _computed_tile_coordinates.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(kernel, (torch.zeros(5, 65),))
        config = bound.config_spec.default_config()
        config.config["cute_fragment_scan"] = mode
        code = bound.to_code(config)
    validate_thread_axis_accesses(ast.parse(code).body)
    assert "fragment_thread" in code
    assert "block=(128, 1, 1)" in code
    assert "thread_idx()[3]" not in code


@pytest.mark.parametrize(
    "source,valid",
    [
        ("value = cute.arch.thread_idx()[0]", True),
        ("value = cute.arch.thread_idx()[1]", True),
        ("value = cute.arch.thread_idx()[2]", True),
        ("x, y, z = cute.arch.thread_idx()", True),
        ("indices = cute.arch.thread_idx(); alias = indices; value = alias[2]", True),
        ("value = cute.arch.thread_idx()[3]", False),
        ("value = cute.arch.thread_idx()[-1]", False),
        ("value = cute.arch.thread_idx()[axis]", False),
        ("indices = cute.arch.thread_idx(); alias = indices; value = alias[4]", False),
        ("if predicate:\n    value = cute.arch.thread_idx()[3]", False),
    ],
)
def test_cute_final_thread_axis_validation(source, valid):
    if valid:
        validate_thread_axis_accesses(ast.parse(source).body)
    else:
        with pytest.raises(exc.BackendUnsupported, match="thread axis"):
            validate_thread_axis_accesses(ast.parse(source).body)


def test_cute_live_invalid_grid_axis_rejected_after_codegen():
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def add_one(x: torch.Tensor):
        out = torch.empty_like(x)
        for row, col in hl.tile(x.shape, block_size=[2, 32]):
            out[row, col] = x[row, col] + 1
        return out

    original_grid_index = CuteBackend.grid_index_expr

    def invalid_grid_index(self, offset_var, block_size_var, dtype, *, axis):
        return original_grid_index(self, offset_var, block_size_var, dtype, axis=3)

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch.object(CuteBackend, "grid_index_expr", invalid_grid_index),
    ):
        bound = _cpu_bind(add_one, (torch.zeros(5, 65),))
        with pytest.raises(exc.BackendUnsupported, match="thread axis 3"):
            bound.to_code(bound.config_spec.default_config())


@helion.kernel(static_shapes=True, autotune_effort="none")
def _partition_allocations(x: torch.Tensor, stepped: hl.constexpr):
    rows, width = x.shape
    block = hl.register_block_size(width)
    parts = (width + block - 1) // block
    first = torch.empty((rows, parts), device=x.device, dtype=x.dtype)
    second = torch.empty((rows, parts), device=x.device, dtype=x.dtype)
    out = torch.empty((rows,), device=x.device, dtype=x.dtype)
    for row, col in hl.tile([rows, width], block_size=[1, block]):
        values = x[row, col].sum(-1)
        first[row, col.id] = values
        second[row, col.id] = values + 1
    hl.barrier()
    for row in hl.tile(rows):
        if stepped:
            out[row] = (first[row, ::2] + second[row, ::2]).sum(-1)
        else:
            out[row] = (first[row, :] + second[row, :]).sum(-1)
    return first, second, out


@skipIfRefEager("requires compiler IR and explicit configurations")
@pytest.mark.parametrize("backend", ["cute", "triton"])
@pytest.mark.parametrize("stepped", [False, True])
def test_partition_allocation_slices_preserve_configured_extent(backend, stepped):
    kernel = helion.kernel(
        _partition_allocations.fn,
        backend=backend,
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, (torch.ones(3, 65), stepped))
        loads = [
            node
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is hl.load
            and any(
                isinstance(k, slice) and (k.step == 2) == stepped for k in node.args[1]
            )
        ]
        assert {node.args[0].args[0] for node in loads} == {"first", "second"}
        # local_types describes the latest root's scope. The consumer's host
        # tensor inputs retain the allocation metadata for earlier roots.
        allocations = {node.args[0].args[0]: node.args[0].meta["val"] for node in loads}
        first, second = allocations["first"], allocations["second"]
        assert first is not second
        assert first.untyped_storage() != second.untyped_storage()
        with bound.env, bound.host_function:
            assert not bound.env.known_equal(first.size(-1), 1)
            assert not bound.env.known_equal(second.size(-1), 1)
        for block in (16, 32, 128):
            config = bound.config_spec.default_config()
            config.config["pid_type"] = (
                "flat" if kernel.settings.backend == "cute" else "persistent_blocked"
            )
            cast("list[int]", config["block_sizes"])[:] = [block, 4]
            config.config["reduction_loops"] = [None] * len(
                cast("list[object]", config["reduction_loops"])
            )
            count = (65 + block - 1) // block
            expected = (count + 1) // 2 if stepped else count
            with bound.env, bound.host_function:
                for node in loads:
                    block_id = bound.env.get_block_id(node.meta["val"].size(-1))
                    assert block_id is not None
                    logical = bound.env.block_sizes[block_id].size
                    expression = logical._sympy_()
                    assert (
                        int(expression.subs(bound.env.block_sizes[0].symbol(), block))
                        == expected
                    )
            if backend == "cute" and stepped:
                with pytest.raises(exc.BackendUnsupported, match="strided slices"):
                    bound.to_code(config)
                continue
            code = bound.to_code(config)
            # Native execution below checks values; here the independently
            # allocated buffers must still have partition-dependent stores.
            if count > 1:
                sources = [code] + [
                    node.value
                    for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "def _helion_" in node.value
                ]
                for name in ("first", "second"):
                    stores = []
                    for source in sources:
                        for call in ast.walk(ast.parse(source)):
                            if not (
                                isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Attribute)
                                and call.func.attr == "store"
                            ):
                                continue
                            pointer = (
                                call.args[0]
                                if ast.unparse(call.func) == "tl.store"
                                else call.func.value
                            )
                            if any(
                                isinstance(node, ast.Name) and node.id == name
                                for node in ast.walk(pointer)
                            ):
                                stores.append(ast.unparse(call))
                    assert stores
                    assert all(
                        "tile_id" in line or "tile_offset_0 //" in line
                        for line in stores
                    ), stores


@skipIfRefEager("requires compiler IR and explicit configurations")
@skipUnlessBackends(["cute", "triton"])
@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("block", [16, 32, 128])
def test_partition_allocation_slice_values(block, stepped):
    if stepped and _get_backend() == "cute":
        pytest.skip("CuTe rejects stepped slices; CPU test checks that rejection")
    x = torch.arange(3 * 65, device=DEVICE, dtype=torch.float32).reshape(3, 65)
    before = x.clone()
    bound = _partition_allocations.bind((x, stepped))
    config = bound.config_spec.default_config()
    config.config["pid_type"] = (
        "flat" if _get_backend() == "cute" else "persistent_blocked"
    )
    cast("list[int]", config["block_sizes"])[:] = [block, 4]
    config.config["reduction_loops"] = [None] * len(
        cast("list[object]", config["reduction_loops"])
    )
    first, second, out = bound.compile_config(config)(x, stepped)
    expected = torch.stack(
        [x[:, start : start + block].sum(-1) for start in range(0, 65, block)],
        dim=-1,
    )
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected + 1)
    selected = expected[:, ::2] if stepped else expected
    torch.testing.assert_close(out, (selected * 2 + 1).sum(-1))
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert first.untyped_storage() != second.untyped_storage()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_coordinate_pad_2d(x: torch.Tensor, width: int):
    width = hl.specialize(width)
    flat = x.reshape(-1)
    out = torch.empty((x.size(0), width), device=x.device, dtype=x.dtype)
    for row, col in hl.tile(out.shape):
        address = row.index[:, None] * x.size(1) + col.index[None, :]
        valid = col.index[None, :] < x.size(1)
        value = hl.load(flat, [address], extra_mask=valid)
        out[row, col] = torch.where(valid, value + 1, -7)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_coordinate_pad_3d(x: torch.Tensor, width: int):
    width = hl.specialize(width)
    flat = x.reshape(-1)
    out = torch.empty((x.size(0), x.size(1), width), device=x.device, dtype=x.dtype)
    for plane, row, col in hl.tile(out.shape):
        address = (
            plane.index[:, None, None] * x.size(1) + row.index[None, :, None]
        ) * x.size(2) + col.index[None, None, :]
        valid = col.index[None, None, :] < x.size(2)
        value = hl.load(flat, [address], extra_mask=valid)
        out[plane, row, col] = torch.where(valid, value + 1, -7)
    return out


def _execute_pointwise_thread_program(source, inputs):
    """Execute the actual scalar program for every launched CTA/thread on CPU."""
    import itertools
    import operator
    from types import SimpleNamespace

    writes = {}

    class Pointer:
        def __init__(self, tensor, offset=0):
            self.tensor, self.offset = tensor, int(offset)

        def __add__(self, offset):
            return Pointer(self.tensor, self.offset + int(offset))

        def load(self):
            storage = self.tensor.as_strided(
                (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
                (1,),
                storage_offset=0,
            )
            offset = self.tensor.storage_offset() + self.offset
            assert 0 <= offset < storage.numel()
            return storage[offset].item()

        def store(self, value):
            storage = self.tensor.as_strided(
                (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
                (1,),
                storage_offset=0,
            )
            offset = self.tensor.storage_offset() + self.offset
            assert 0 <= offset < storage.numel()
            key = (self.tensor.untyped_storage().data_ptr(), offset)
            writes[key] = writes.get(key, 0) + 1
            storage[offset] = value

    current = {"block": (0, 0, 0), "thread": (0, 0, 0)}
    launches = []

    def launcher(function, grid, *arguments, block):
        launches.append(block)
        args = [
            SimpleNamespace(
                iterator=Pointer(arg), layout=SimpleNamespace(stride=arg.stride())
            )
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in arguments
        ]
        for cta in itertools.product(*(range(size) for size in grid)):
            current["block"] = (*cta, *((0,) * (3 - len(cta))))
            for thread in itertools.product(*(range(size) for size in block)):
                current["thread"] = thread
                function(*args)

    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        body.append(node)
    namespace = {
        "torch": torch,
        "operator": operator,
        "_cute_python_mod": operator.mod,
        "cutlass": SimpleNamespace(Int32=int, Int64=int, Float32=float, Boolean=bool),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: current["block"], thread_idx=lambda: current["thread"]
            )
        ),
        "_default_cute_launcher": launcher,
        "_next_power_of_2": lambda value: 1 << (value - 1).bit_length(),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<pointwise-thread-model>",
            "exec",
        ),
        namespace,
    )
    wrapper = next(
        node.name for node in reversed(body) if isinstance(node, ast.FunctionDef)
    )
    result = namespace[wrapper](*inputs)
    assert len(writes) == result.numel()
    assert set(writes.values()) == {1}, "duplicate or missing output ownership"
    return result, launches


@pytest.mark.parametrize(
    "shape,width,tiles",
    [((3, 67), 73, [4, 32]), ((5, 9), 17, [4, 8]), ((3, 5, 9), 17, [2, 4, 8])],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_cute_inactive_broadcast_axes_generated_values(shape, width, tiles, reverse):
    from helion._compiler.tile_dispatch import TileStrategyDispatch

    inputs = (torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape), width)
    aliases = []
    original = TileStrategyDispatch._inactive_cute_broadcast_aliases

    def observe(dispatch, function):
        result = original(dispatch, function)
        aliases.append(result)
        return result

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch.object(TileStrategyDispatch, "_inactive_cute_broadcast_aliases", observe),
    ):
        bound = _cpu_bind(
            _broadcast_coordinate_pad_2d
            if len(shape) == 2
            else _broadcast_coordinate_pad_3d,
            inputs,
        )
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = tiles
        config.config["loop_orders"] = [
            list(reversed(range(len(shape)))) if reverse else list(range(len(shape)))
        ]
        source = bound.to_code(config)
    assert any(aliases)
    validate_thread_axis_accesses(ast.parse(source).body)
    actual, launches = _execute_pointwise_thread_program(source, inputs)
    expected = torch.full((*shape[:-1], width), -7.0)
    expected[..., : shape[-1]] = inputs[0] + 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(len(block) == 3 for block in launches)


@pytest.mark.parametrize("axis", [-1, 3, 4])
def test_cute_pointwise_invalid_axis_rejects_structurally(axis):
    from helion._compiler.cute.backend import _pointwise_grid_thread_dims

    with pytest.raises(exc.BackendUnsupported, match="physical thread axes"):
        _pointwise_grid_thread_dims(
            {axis: 4},
            {axis},
            [1, 1, 1],
            has_pointwise_fact=True,
            has_nested_device_loops=False,
            has_synthetic_free_axes=False,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_axis_free_iota(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        index = hl.arange(x.size(1))
        out[row, :] = x[row, :] + index[None, :]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_axis_nested(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        for col in hl.tile(x.size(1)):
            out[row, col] = x[row, col] + row.index[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_axis_matmul(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty((x.size(0), y.size(1)), dtype=x.dtype, device=x.device)
    for row, col in hl.tile(out.shape):
        out[row, col] = hl.dot(x[row, :], y[:, col])
    return out


@pytest.mark.parametrize("kind", ["reduction", "free_iota", "matmul"])
def test_cute_inactive_broadcast_axes_preserve_other_owners(kind):
    from helion._compiler.tile_dispatch import TileStrategyDispatch

    cases = {
        "reduction": (
            helion.kernel(
                reduction_sum.fn,
                backend="cute",
                static_shapes=True,
                autotune_effort="none",
            ),
            (torch.ones(3, 8),),
        ),
        "free_iota": (_broadcast_axis_free_iota, (torch.ones(3, 8),)),
        "matmul": (
            _broadcast_axis_matmul,
            (
                torch.ones(16, 16, dtype=torch.float16),
                torch.ones(16, 16, dtype=torch.float16),
            ),
        ),
    }
    kernel, inputs = cases[kind]
    aliases = []
    original = TileStrategyDispatch._inactive_cute_broadcast_aliases

    def observe(dispatch, function):
        result = original(dispatch, function)
        aliases.append(result)
        return result

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        bound = _cpu_bind(kernel, inputs)
        config = bound.config_spec.default_config()
        with patch.object(
            TileStrategyDispatch, "_inactive_cute_broadcast_aliases", observe
        ):
            actual = bound.to_code(config)
        with patch.object(
            TileStrategyDispatch, "_inactive_cute_broadcast_aliases", return_value=set()
        ):
            previous = bound.to_code(config)
    assert aliases and not any(aliases)
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(previous))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "shape,width,tiles", [((3, 67), 73, [4, 32]), ((3, 5, 9), 17, [2, 4, 8])]
)
def test_cute_inactive_broadcast_axes_native(shape, width, tiles):
    x = torch.arange(math.prod(shape), device=DEVICE, dtype=torch.float32).reshape(
        shape
    )
    kernel = (
        _broadcast_coordinate_pad_2d
        if len(shape) == 2
        else _broadcast_coordinate_pad_3d
    )
    bound = kernel.bind((x, width))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = tiles
    actual = bound.compile_config(config)(x, width)
    expected = torch.full((*shape[:-1], width), -7.0, device=DEVICE)
    expected[..., : shape[-1]] = x + 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@helion.kernel(static_shapes=True, autotune_effort="none")
def _scaled_tile_slice(
    x: torch.Tensor, divide: hl.constexpr, factor: hl.constexpr
) -> torch.Tensor:
    rows, width = x.shape
    logical_width = width * factor if divide else width // factor
    block = hl.register_block_size(logical_width)
    out = torch.empty((rows, logical_width), dtype=x.dtype, device=x.device)
    for row, col in hl.tile([rows, logical_width], block_size=[1, block]):
        if divide:
            packed = x[
                row,
                col.begin // factor : col.begin // factor + col.block_size // factor,
            ]
            values = torch.stack([packed] * factor, dim=-1).reshape(
                row.block_size, col.block_size
            )
        else:
            packed = x[
                row, col.begin * factor : col.begin * factor + col.block_size * factor
            ]
            values = packed.reshape(row.block_size, col.block_size, factor).sum(-1)
        out[row, col] = values
    return out


@skipIfRefEager("requires compiler IR and explicit configurations")
@pytest.mark.parametrize("backend", ["cute", "triton"])
@pytest.mark.parametrize("divide", [False, True])
@pytest.mark.parametrize("factor", [2, 4])
def test_scaled_tile_slices_preserve_shape_algebra(backend, divide, factor):
    kernel = helion.kernel(
        _scaled_tile_slice.fn,
        backend=backend,
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(kernel, (torch.ones(3, 64), divide, factor))
        loads = [
            node
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is hl.load
        ]
        assert len(loads) == 1
        # Reshape above must keep the algebraic relation to the registered
        # tile width; an unrelated reduction symbol cannot satisfy it.
        with bound.env, bound.host_function:
            width = bound.env.block_sizes[0].var
            expected = width // factor if divide else width * factor
            assert bound.env.known_equal(loads[0].meta["val"].size(-1), expected)
        for block in (8, 16, 32):
            config = bound.config_spec.default_config()
            config.config["block_sizes"] = [block]
            if backend == "cute" and not divide:
                # This pre-existing non-matmul affine-load limitation is
                # independent of the shape relation fixed here.
                with pytest.raises(exc.BackendUnsupported, match="affine hl.arange"):
                    bound.to_code(config)
            else:
                ast.parse(bound.to_code(config))


@skipUnlessBackends(["cute", "triton"])
@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("block", [8, 32])
def test_scaled_tile_slice_values(factor, block):
    # Noncontiguous storage exercises slice addresses as well as reshape sizes.
    x = torch.arange(6 * 128, device=DEVICE, dtype=torch.float32).reshape(6, 128)[
        ::2, ::2
    ]
    before = x.clone()
    _, actual = code_and_output(
        _scaled_tile_slice, (x, True, factor), block_sizes=[block]
    )
    torch.testing.assert_close(actual, x.repeat_interleave(factor, -1), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=False, disable_autotuner_heuristics=True)
def _singleton_cache_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), x.size(1)), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0)):
        out[row, :] = x[row, :] + 1
    return out


@pytest.mark.parametrize("widths", [(1, 5, 1, 9), (5, 1, 5, 1)])
@pytest.mark.parametrize("block", [2, 4])
def test_backed_singleton_slice_binding_cache(widths, block):
    # Keep pointer/stride/shape alignment residues equal so an unrelated
    # vectorization guard cannot hide a missing singleton specialization.
    kernel = helion.kernel(
        _singleton_cache_copy.fn,
        backend="cute",
        static_shapes=False,
        disable_autotuner_heuristics=True,
    )
    storage = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16)
    before = storage.clone()
    bound_by_width = {}
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        for width in widths:
            x = storage[:, :width]
            bound = kernel.bind((x,))
            if width in bound_by_width:
                assert bound is bound_by_width[width]
            bound_by_width[width] = bound
            config = bound.config_spec.default_config()
            config.config["block_sizes"] = [block]
            actual, _ = _execute_pointwise_thread_program(bound.to_code(config), (x,))
            torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)
            torch.testing.assert_close(storage, before, rtol=0, atol=0)
        if widths[0] == 1:
            assert bound_by_width[1] is not bound_by_width[5]
            assert bound_by_width[1].env.specialized_vars


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("widths", [(1, 5, 1, 9), (5, 1, 5, 1)])
@pytest.mark.parametrize("block", [2, 4])
def test_backed_singleton_slice_binding_values(widths, block):
    kernel = helion.kernel(
        _singleton_cache_copy.fn,
        backend="cute",
        static_shapes=False,
        disable_autotuner_heuristics=True,
    )
    storage = torch.arange(3 * 16, device=DEVICE, dtype=torch.float32).reshape(3, 16)
    before = storage.clone()
    for width in widths:
        x = storage[:, :width]
        bound = kernel.bind((x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [block]
        actual = bound.compile_config(config)(x)
        torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)
        torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize("hint", [1, 5])
def test_singleton_guard_only_specializes_true_backed_size(hint):
    from torch._dynamo.source import LocalSource

    from helion._compiler.compile_environment import CompileEnvironment

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        env = CompileEnvironment(
            torch.device("cpu"), helion.Settings(backend="cute", static_shapes=False)
        )
        with env:
            size = env.input_symint(hint, LocalSource("size"))
            symbols = set(size._sympy_().free_symbols)
            assert symbols
            assert env.is_singleton_size(size) == (hint == 1)
            assert env.specialized_vars == (symbols if hint == 1 else set())


@pytest.mark.parametrize("derived", [False, True])
def test_singleton_guard_does_not_specialize_configured_hint_one(derived):
    from helion._compiler.compile_environment import CompileEnvironment

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        env = CompileEnvironment(
            torch.device("cpu"), helion.Settings(backend="cute", static_shapes=False)
        )
        with env:
            tile = env.create_unbacked_symint(32 if derived else 1)
            size = (tile + 31) // 32 if derived else tile
            replacements = dict(env.shape_env.replacements)
            guards = tuple(env.shape_env.guards)
            assert not env.is_singleton_size(size)
            assert not env.specialized_vars
            assert env.shape_env.replacements == replacements
            assert tuple(env.shape_env.guards) == guards
            assert env.is_singleton_size(tile * 0 + 1)
            assert not env.specialized_vars


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_broadcast_gather(col, val, table, flat: hl.constexpr):
    m, n, k = col.shape
    _, p, q = table.shape
    out = torch.empty((m, n, p, q), dtype=val.dtype, device=val.device)
    flat_table = table.reshape(-1)
    for mi, ni, pi, qi in hl.tile([m, n, p, q]):
        acc = hl.zeros([mi, ni, pi, qi], dtype=torch.float32)
        for ki in hl.tile(k):
            column = col[mi, ni, ki]
            if flat:
                index = (
                    column[:, :, :, None, None] * (p * q)
                    + pi.index[None, None, :, None] * q
                    + qi.index[None, None, None, :]
                )
                selected = hl.load(flat_table, [index])
            else:
                selected = table[
                    column[:, :, :, None, None],
                    pi.index[None, None, :, None],
                    qi.index[None, None, None, :],
                ]
            acc += (val[mi, ni, ki][:, :, :, None, None] * selected).sum(2)
        out[mi, ni, pi, qi] = acc
    return out


@pytest.mark.parametrize("shape", [(3, 5, 7, 9, 11), (4, 3, 8, 5, 6)])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_cute_nested_broadcast_alias_values(shape, flat, reverse, static_shapes):
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.tile_dispatch import TileStrategyDispatch

    m, n, k, p, q = shape
    column = torch.arange(m * n * k).reshape(m, n, k) % k
    # Small integer-valued FP32 data makes every modeled add/multiply exact.
    values = (torch.arange(m * n * k).reshape(m, n, k) % 5).float()
    table = (torch.arange(k * p * q).reshape(k, p, q) % 7).float()
    inputs = column, values, table, flat
    aliases = []
    original = TileStrategyDispatch._inactive_cute_broadcast_aliases

    def observe(dispatch, function):
        result = original(dispatch, function)
        env = CompileEnvironment.current()
        outer = set(function.codegen.host_function.device_ir.grid_block_ids[0])
        assert result.isdisjoint(env.config_spec.reduction_loops.valid_block_ids())
        assert all(env.canonical_block_id(block) in outer for block in result)
        aliases.append(result)
        return result

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch.object(TileStrategyDispatch, "_inactive_cute_broadcast_aliases", observe),
    ):
        kernel = helion.kernel(
            _nested_broadcast_gather.fn,
            backend="cute",
            static_shapes=static_shapes,
            autotune_effort="none",
        )
        bound = _cpu_bind(kernel, inputs)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [2, 2, 4, 4, 4]
        config.config["loop_orders"] = [[3, 2, 1, 0] if reverse else [0, 1, 2, 3]]
        source = bound.to_code(config)
    assert any(aliases)
    actual, launches = _execute_pointwise_thread_program(source, inputs)
    expected = (values[..., None, None] * table[column]).sum(2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(len(block) == 3 and math.prod(block) <= 1024 for block in launches)


def test_cute_nested_pointwise_outer_alias_values():
    inputs = (torch.arange(3 * 17, dtype=torch.float32).reshape(3, 17),)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_broadcast_axis_nested, inputs)
        source = bound.to_code(helion.Config(block_sizes=[2, 8]))
    actual, _ = _execute_pointwise_thread_program(source, inputs)
    torch.testing.assert_close(
        actual, inputs[0] + torch.arange(3)[:, None], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "guard", ["second_root", "scalar_loop", "child_owner", "executable", "free_iota"]
)
def test_cute_nested_broadcast_alias_scope_guards(guard):
    import dataclasses

    from helion._compiler.device_ir import ForLoopGraphInfo
    from helion._compiler.device_ir import RootGraphInfo
    from helion._compiler.tile_dispatch import TileStrategyDispatch

    column = torch.arange(3 * 5 * 7).reshape(3, 5, 7) % 7
    inputs = (
        column,
        torch.ones_like(column, dtype=torch.float32),
        torch.ones(7, 9, 11),
        False,
    )
    original = TileStrategyDispatch._inactive_cute_broadcast_aliases
    checked = []

    def observe(dispatch, function):
        result = original(dispatch, function)
        assert result
        graphs = function.codegen.codegen_graphs
        child = next(graph for graph in graphs if type(graph) is ForLoopGraphInfo)
        root = next(graph for graph in graphs if isinstance(graph, RootGraphInfo))
        if guard == "second_root":
            replacement = [*graphs, root.copy()]
            context = patch.object(function.codegen, "codegen_graphs", replacement)
        elif guard == "scalar_loop":
            replacement = [
                dataclasses.replace(graph, block_ids=[]) if graph is child else graph
                for graph in graphs
            ]
            context = patch.object(function.codegen, "codegen_graphs", replacement)
        elif guard == "child_owner":
            context = patch.object(
                function.codegen.host_function.device_ir,
                "grid_block_ids",
                [child.block_ids],
            )
        elif guard == "executable":
            context = patch.object(
                bound.env.config_spec.reduction_loops,
                "valid_block_ids",
                return_value=[
                    *bound.env.config_spec.reduction_loops.valid_block_ids(),
                    *result,
                ],
            )
        else:
            changed_root = root.copy()
            changed_root.graph.call_function(torch.ops.aten.arange.default, (4,))
            replacement = [changed_root if graph is root else graph for graph in graphs]
            context = patch.object(function.codegen, "codegen_graphs", replacement)
        with context:
            assert not original(dispatch, function)
        checked.append(guard)
        return result

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch.object(TileStrategyDispatch, "_inactive_cute_broadcast_aliases", observe),
    ):
        bound = _cpu_bind(_nested_broadcast_gather, inputs)
        bound.to_code(helion.Config(block_sizes=[2, 2, 4, 4, 4]))
    assert checked == [guard]


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_cute_nested_broadcast_alias_native(flat, reverse):
    m, n, k, p, q = 3, 5, 7, 9, 11
    column = torch.arange(m * n * k, device=DEVICE).reshape(m, n, k) % k
    values = (torch.arange(m * n * k, device=DEVICE).reshape(m, n, k) % 5).float()
    table = (torch.arange(k * p * q, device=DEVICE).reshape(k, p, q) % 7).float()
    inputs = column, values, table, flat
    before = tuple(tensor.clone() for tensor in inputs[:3])
    bound = _nested_broadcast_gather.bind(inputs)
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [2, 2, 4, 4, 4]
    config.config["loop_orders"] = [[3, 2, 1, 0] if reverse else [0, 1, 2, 3]]
    actual = bound.compile_config(config)(*inputs)
    torch.testing.assert_close(
        actual, (values[..., None, None] * table[column]).sum(2), rtol=0, atol=0
    )
    torch.testing.assert_close(inputs[:3], before, rtol=0, atol=0)


def _integer_modulo_ir_program(
    dtype, *, raw=False, right_dtype=None, constant=None, operand_dtype=None
):
    """Capture real SDK scalar operations; no native compilation or device."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    from helion._compiler.cute.integer_helpers import python_mod

    right_dtype = dtype if right_dtype is None else right_dtype
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "integer_modulo", ([dtype.mlir_type, right_dtype.mlir_type], [])
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                left = dtype(block.arguments[0])
                right = (
                    right_dtype(block.arguments[1]) if constant is None else constant
                )
                if operand_dtype is not None:
                    left, right = operand_dtype(left), operand_dtype(right)
                result = left % right if raw else python_mod(left, right)
                func.ReturnOp([])
        identifiers = {value: index for index, value in enumerate(block.arguments)}
        program = []
        for operation in block.operations:
            op = operation.operation
            if op.name == "func.return":
                continue
            assert len(op.results) == 1
            value = op.results[0]
            identifiers[value] = len(identifiers)
            attributes = {
                name: int(op.attributes[name].value)
                for name in ("value", "predicate")
                if name in op.attributes
            }
            program.append(
                (
                    identifiers[value],
                    op.name,
                    [identifiers[argument] for argument in op.operands],
                    ir.IntegerType(value.type).width,
                    attributes,
                )
            )
        return program, identifiers[result.ir_value()], str(module)


def _execute_integer_modulo_ir(
    program, output, width, signed, left, right, right_width=None
):
    """Interpret emitted integer IR, checking poison rather than hiding it."""
    values = {0: left, 1: right}
    widths = {0: width, 1: width if right_width is None else right_width}

    def signed_value(value, bits):
        value = int(value) & ((1 << bits) - 1)
        return value - (1 << bits) if value & (1 << (bits - 1)) else value

    for destination, operation, arguments, bits, attributes in program:
        operands = [values[index] for index in arguments]
        if operation == "arith.constant":
            value = attributes["value"]
        elif operation == "arith.addi":
            value = sum(operands)
        elif operation == "arith.remsi":
            dividend, divisor = (signed_value(value, bits) for value in operands)
            assert divisor != 0 and not (
                dividend == -(1 << (bits - 1)) and divisor == -1
            ), "undefined raw signed remainder"
            value = (abs(dividend) % abs(divisor)) * (-1 if dividend < 0 else 1)
        elif operation == "arith.remui":
            value = (operands[0] & ((1 << bits) - 1)) % (
                operands[1] & ((1 << bits) - 1)
            )
        elif operation == "arith.andi":
            value = operands[0] & operands[1]
        elif operation == "arith.select":
            value = operands[1] if operands[0] else operands[2]
        elif operation == "arith.cmpi":
            predicate = attributes["predicate"]
            operand_bits = widths[arguments[0]]
            first, second = operands
            if predicate in (2, 3, 4, 5):
                first, second = (
                    signed_value(value, operand_bits) for value in operands
                )
            else:
                first, second = (
                    value & ((1 << operand_bits) - 1) for value in operands
                )
            value = (
                first == second,
                first != second,
                first < second,
                first <= second,
                first > second,
                first >= second,
                first < second,
                first <= second,
                first > second,
                first >= second,
            )[predicate]
        elif operation in ("arith.bitcast", "arith.trunci", "arith.extui"):
            value = operands[0]
        elif operation == "arith.extsi":
            value = signed_value(operands[0], widths[arguments[0]])
        else:
            raise AssertionError(operation)
        values[destination] = int(value) & ((1 << bits) - 1)
        widths[destination] = bits
    return signed_value(values[output], widths[output]) if signed else values[output]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cute_integer_remainders(left, right, c_style: hl.constexpr):
    result = torch.empty_like(left)
    for tile in hl.tile(left.size(0)):
        if c_style:
            result[tile] = torch.fmod(left[tile], right[tile])
        else:
            result[tile] = torch.remainder(left[tile], right[tile])
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cute_integer_modulo_offsets(values):
    result = torch.zeros_like(values)
    for row in hl.grid(values.size(0)):
        offset = (-row * 5) % values.size(1)
        for columns in hl.tile(values.size(1)):
            positions = (columns.index + offset) % values.size(1)
            mask = ((-columns.index) % 3) != 0
            result[row, columns] = hl.load(values, [row, positions], extra_mask=mask)
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cute_wrapped_remainder(left, divisor: hl.constexpr):
    result = torch.empty_like(left)
    for tile in hl.tile(left.size(0)):
        result[tile] = torch.remainder(left[tile], divisor)
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cute_promoted_remainder(left, right, output_dtype: hl.constexpr):
    result = torch.empty(left.shape, dtype=output_dtype, device=left.device)
    for rows, columns in hl.tile(left.shape):
        if right.numel() == 1:
            value = right[0]
        else:
            value = right[columns]
        result[rows, columns] = torch.remainder(left[rows, columns], value)
    return result


class TestCuteIntegerModuloCPU(unittest.TestCase):
    def setUp(self):
        import importlib.util

        if importlib.util.find_spec("cutlass") is None:
            self.skipTest("requires CuTe scalar IR, not a GPU")

    def test_actual_integer_ir_matches_python_and_avoids_poison(self):
        import itertools

        import cutlass

        for dtype in (
            cutlass.Int8,
            cutlass.Int16,
            cutlass.Int32,
            cutlass.Int64,
            cutlass.Uint8,
            cutlass.Uint32,
            cutlass.Uint64,
        ):
            with self.subTest(dtype=dtype):
                program, output, _source = _integer_modulo_ir_program(dtype)
                width, signed = dtype.width, dtype.signed
                low = -(1 << (width - 1)) if signed else 0
                high = (1 << (width - int(signed))) - 1
                cases = (
                    range(low, high + 1)
                    if width == 8
                    else sorted(
                        {
                            low,
                            low + 1,
                            low + 2,
                            high - 2,
                            high - 1,
                            high,
                            *(value for value in range(-9, 10) if low <= value <= high),
                        }
                    )
                )
                for left, right in itertools.product(cases, repeat=2):
                    if right:
                        self.assertEqual(
                            _execute_integer_modulo_ir(
                                program, output, width, signed, left, right
                            ),
                            left % right,
                        )
                if signed:
                    raw, raw_output, _source = _integer_modulo_ir_program(
                        dtype, raw=True
                    )
                    self.assertEqual(
                        _execute_integer_modulo_ir(raw, raw_output, width, True, -7, 3),
                        -1,
                    )
                    with self.assertRaisesRegex(AssertionError, "undefined raw"):
                        _execute_integer_modulo_ir(
                            raw, raw_output, width, True, low, -1
                        )

    def test_promoted_widths_and_python_constants(self):
        import cutlass

        for left_dtype, right_dtype in (
            (cutlass.Int8, cutlass.Int32),
            (cutlass.Int16, cutlass.Int64),
            (cutlass.Int64, cutlass.Int8),
        ):
            program, output, _source = _integer_modulo_ir_program(
                left_dtype, right_dtype=right_dtype
            )
            for left, right in ((-7, 3), (7, -3), (-7, -3), (0, -1)):
                self.assertEqual(
                    _execute_integer_modulo_ir(
                        program,
                        output,
                        left_dtype.width,
                        True,
                        left,
                        right,
                        right_dtype.width,
                    ),
                    left % right,
                )
        for dtype in (cutlass.Int8, cutlass.Int32, cutlass.Int64):
            for constant in (3, -3, 1 << 40, -(1 << 40)):
                program, output, _source = _integer_modulo_ir_program(
                    dtype, constant=constant
                )
                for left in (-7, 0, 7):
                    self.assertEqual(
                        _execute_integer_modulo_ir(
                            program, output, dtype.width, True, left, 0
                        ),
                        left % constant,
                    )

    def test_proved_positive_printer_path_is_unchanged(self):
        import sympy
        from torch.utils._sympy.functions import PythonMod

        from helion._compiler.cute.printer import cute_texpr

        nonnegative = sympy.Symbol("position", integer=True, nonnegative=True)
        positive = sympy.Symbol("extent", integer=True, positive=True)
        unknown = sympy.Symbol("offset", integer=True)
        self.assertEqual(
            cute_texpr(PythonMod(nonnegative, positive)), "((position) % (extent))"
        )
        self.assertEqual(
            cute_texpr(PythonMod(unknown, positive)), "_cute_python_mod(offset, extent)"
        )
        self.assertEqual(
            cute_texpr(PythonMod(nonnegative, -positive)),
            "_cute_python_mod(position, (-1)*extent)",
        )

    def test_tensor_wrapped_scalars_and_promoted_operands(self):
        import ast

        import cutlass

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        for torch_dtype, sdk_dtype in (
            (torch.int8, cutlass.Int8),
            (torch.int16, cutlass.Int16),
            (torch.int32, cutlass.Int32),
        ):
            values = torch.tensor([-7, 7], dtype=torch_dtype)
            for scalar in (
                torch.iinfo(torch_dtype).max + 2,
                -(torch.iinfo(torch_dtype).max + 2),
            ):
                with self.subTest(dtype=torch_dtype, scalar=scalar):
                    program, output, _ir = _integer_modulo_ir_program(
                        sdk_dtype, constant=scalar, operand_dtype=sdk_dtype
                    )
                    actual = [
                        _execute_integer_modulo_ir(
                            program, output, sdk_dtype.width, True, int(value), 0
                        )
                        for value in values
                    ]
                    self.assertEqual(actual, torch.remainder(values, scalar).tolist())
                    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
                        bound = _cpu_bind(_cute_wrapped_remainder, (values, scalar))
                        source = bound.to_code(bound.config_spec.default_config())
                    calls = [
                        n
                        for n in ast.walk(ast.parse(source))
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id == "_cute_python_mod"
                    ]
                    self.assertTrue(calls)
                    for call in calls:
                        self.assertEqual(
                            [ast.unparse(arg.func) for arg in call.args],
                            [f"cutlass.{sdk_dtype.__name__}"] * 2,
                        )
        left = torch.tensor([[-7, 7], [-9, 9]], dtype=torch.int8)
        for right in (
            torch.tensor(129, dtype=torch.int64),
            torch.tensor([129, -129], dtype=torch.int32),
        ):
            expected = torch.remainder(left, right)
            sdk_dtype = cutlass.Int8 if expected.dtype == torch.int8 else cutlass.Int32
            with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
                bound = _cpu_bind(
                    # The scalar operand is a rank-zero device value loaded
                    # from a rank-one host tensor, independent of scalar I/O.
                    _cute_promoted_remainder,
                    (left, right.reshape(-1), expected.dtype),
                )
                source = bound.to_code(bound.config_spec.default_config())
            calls = [
                n
                for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_cute_python_mod"
            ]
            self.assertTrue(calls)
            for call in calls:
                self.assertEqual(
                    [ast.unparse(arg.func) for arg in call.args],
                    [f"cutlass.{sdk_dtype.__name__}"] * 2,
                )
            right_sdk = cutlass.Int64 if right.ndim == 0 else cutlass.Int32
            program, output, _ir = _integer_modulo_ir_program(
                cutlass.Int8, right_dtype=right_sdk, operand_dtype=sdk_dtype
            )
            actual = [
                _execute_integer_modulo_ir(
                    program, output, 8, True, int(a), int(b), right_sdk.width
                )
                for a, b in zip(
                    left.flatten(), right.expand_as(left).flatten(), strict=True
                )
            ]
            self.assertEqual(actual, expected.flatten().tolist())

    def test_codegen_distinguishes_python_remainder_and_c_style_mod(self):
        import ast

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                for c_style in (False, True):
                    with self.subTest(dtype=dtype, c_style=c_style):
                        left = torch.tensor([-7, -1, 0, 1, 7], dtype=dtype)
                        right = torch.tensor([3, -3, 3, -3, 3], dtype=dtype)
                        bound = _cpu_bind(
                            _cute_integer_remainders, (left, right, c_style)
                        )
                        source = bound.to_code(bound.config_spec.default_config())
                        calls = [
                            node
                            for node in ast.walk(ast.parse(source))
                            if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == "_cute_python_mod"
                        ]
                        self.assertEqual(len(calls), int(not c_style))
            values = torch.arange(3 * 17, dtype=torch.int32).reshape(3, 17)
            bound = _cpu_bind(_cute_integer_modulo_offsets, (values,))
            config = bound.config_spec.default_config()
            config.config["block_sizes"] = [16]
            source = bound.to_code(config)
            self.assertIn("_cute_python_mod", source)


@onlyBackends("cute")
class TestCuteIntegerModuloNative(TestCase):
    def test_signed_remainder_and_fmod_values(self):
        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            for c_style in (False, True):
                with self.subTest(dtype=dtype, c_style=c_style):
                    low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
                    pairs = [
                        (-7, 3),
                        (7, -3),
                        (-7, -3),
                        (7, 3),
                        (0, -3),
                        (low, 3),
                        (high, -3),
                        (1, low),
                        (-1, low),
                    ]
                    if not c_style:
                        pairs.append((low, -1))
                    left = torch.tensor(
                        [a for a, _b in pairs], dtype=dtype, device=DEVICE
                    )
                    right = torch.tensor(
                        [b for _a, b in pairs], dtype=dtype, device=DEVICE
                    )
                    expected = torch.tensor(
                        [
                            (abs(a) % abs(b)) * (-1 if a < 0 else 1)
                            if c_style
                            else a % b
                            for a, b in pairs
                        ],
                        dtype=dtype,
                        device=DEVICE,
                    )
                    _source, actual = code_and_output(
                        _cute_integer_remainders,
                        (left, right, c_style),
                        block_sizes=[16],
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_signed_modulo_scalar_offsets_and_tensor_masks(self):
        for width in (17, 65, 2051):
            with self.subTest(width=width):
                values = torch.arange(
                    3 * width, dtype=torch.int32, device=DEVICE
                ).reshape(3, width)
                expected = torch.zeros_like(values)
                for row in range(3):
                    positions = (
                        torch.arange(width, device=DEVICE) + (-row * 5) % width
                    ) % width
                    mask = (-torch.arange(width, device=DEVICE)) % 3 != 0
                    expected[row] = torch.where(mask, values[row, positions], 0)
                _source, actual = code_and_output(
                    _cute_integer_modulo_offsets, (values,), block_sizes=[16]
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wrapped_scalars_and_tensor_promotion(self):
        for dtype in (torch.int8, torch.int16, torch.int32):
            values = torch.tensor([-7, 7], dtype=dtype, device=DEVICE)
            for scalar in (torch.iinfo(dtype).max + 2, -(torch.iinfo(dtype).max + 2)):
                with self.subTest(dtype=dtype, scalar=scalar):
                    _source, actual = code_and_output(
                        _cute_wrapped_remainder, (values, scalar), block_sizes=[16]
                    )
                    torch.testing.assert_close(
                        actual, torch.remainder(values, scalar), rtol=0, atol=0
                    )
        left = torch.tensor([[-7, 7], [-9, 9]], dtype=torch.int8, device=DEVICE)
        right = torch.tensor([129, -129], dtype=torch.int32, device=DEVICE)
        expected = torch.remainder(left, right)
        _source, actual = code_and_output(
            _cute_promoted_remainder, (left, right, expected.dtype)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_rows(x: torch.Tensor, mode: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        source = x[row, :] * 3 + 1
        lane = hl.arange(source.size(0))
        if mode == "duplicate":
            index = lane % 3
        elif mode == "xor":
            index = (lane ^ 1) % x.size(1)
        elif mode == "data":
            index = source.to(torch.int64) % x.size(1)
        else:
            index = (lane * 5 + 3) % x.size(1)
        result = torch.gather(source, -1, index.to(torch.int64))
        out[row, :] = result + source
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_leading(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        source = x[row, :, :] * 3 + 1
        index = source.to(torch.int64) % x.size(2)
        first = torch.gather(source, -1, index)
        other = first + source
        index2 = (source.to(torch.int64) * 5 + 2) % x.size(2)
        second = torch.gather(other, -1, index2)
        out[row, :, :] = first + second
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_bad(x: torch.Tensor, mode: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        source = x[row, :]
        if mode == "negative":
            index = source.to(torch.int64) % x.size(1) - 1
        elif mode == "upper":
            index = source.to(torch.int64) % x.size(1) + 1
        elif mode == "raw":
            index = source.to(torch.int64)
        else:
            index = (source.to(torch.int64) % x.size(1)).to(torch.int32)
        out[row, :] = torch.gather(source, -1, index)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_shift(x: torch.Tensor):
    out = torch.empty((x.size(0), x.size(1) - 1), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        source = x[row, :]
        index = hl.arange(x.size(1) - 1) + 1
        out[row, :] = torch.gather(source, -1, index.to(torch.int64))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_maps(x: torch.Tensor, mode: hl.constexpr, offset: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        source = x[row, :]
        lane = hl.arange(1024)
        if mode == "down":
            index = torch.minimum(lane + offset, lane | 31)
        elif mode == "up":
            index = torch.maximum(lane - offset, lane & -32)
        elif mode == "xor":
            index = lane ^ offset
        elif mode == "end":
            index = lane | 31
        elif mode == "pack":
            index = (lane & 31) * 32
        elif mode == "prefix":
            index = lane // 32
        else:
            index = lane * 0 + offset * 32
        out[row, :] = torch.gather(source, -1, index.to(torch.int64))
    return out


def _bounded_gather_codegen(kernel, args):
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config["cute_fragment_bounded_gather"] = True
        source = bound.to_code(config)
    return bound, source


class TestCuteBoundedGatherCPU(TestCase):
    def test_logical_exchange_maps_and_exact_float_payloads(self):
        from test.test_atomic_ops import _simulate_register_load_program

        lane = torch.arange(1024)
        cases = (
            ("down", 16, torch.minimum(lane + 16, lane | 31)),
            ("up", 16, torch.maximum(lane - 16, lane & -32)),
            ("xor", 16, lane ^ 16),
            ("end", 0, lane | 31),
            ("pack", 0, (lane & 31) * 32),
            ("prefix", 0, lane // 32),
            ("constant", 0, lane * 0),
            ("constant", 31, lane * 0 + 31 * 32),
        )
        bits = (
            torch.tensor(
                [
                    0,
                    -2147483648,
                    1,
                    0x7FFFFF,
                    0x800000,
                    0x7F800000,
                    -8388608,
                    0x7FC00000,
                ],
                dtype=torch.int32,
            )
            .repeat(256)
            .reshape(2, 1024)
        )
        x = bits.view(torch.float32)
        for mode, offset, index in cases:
            with self.subTest(mode=mode, offset=offset):
                _bound, code = _bounded_gather_codegen(
                    _bounded_gather_maps, (x, mode, offset)
                )
                out = torch.empty_like(x)
                _simulate_register_load_program(
                    code,
                    x,
                    128,
                    host_tensors={"out": out},
                    lane_order=list(reversed(range(128))),
                )
                torch.testing.assert_close(
                    out.view(torch.int32), bits[:, index], rtol=0, atol=0
                )

    def test_all_exchange_index_range_proofs(self):
        from helion._compiler.cute.bounded_gather import integer_bounds

        graph = torch.fx.Graph()
        fake = torch.empty(1024, dtype=torch.int64)

        def call(target, *args, **kwargs):
            node = graph.call_function(target, args, kwargs)
            node.meta["val"] = fake
            return node

        lane = call(torch.ops.prims.iota.default, 1024, start=0, step=1)
        end = call(torch.ops.aten.bitwise_or.Scalar, lane, 31)
        begin = call(torch.ops.aten.bitwise_and.Scalar, lane, -32)
        maps = [
            end,
            call(torch.ops.aten.div.Scalar_mode, lane, 32, rounding_mode="floor"),
        ]
        maps.append(
            call(
                torch.ops.aten.mul.Scalar,
                call(torch.ops.aten.bitwise_and.Scalar, lane, 31),
                32,
            )
        )
        for offset in (1, 2, 4, 8, 16):
            maps.extend(
                (
                    call(
                        torch.ops.aten.minimum.default,
                        call(torch.ops.aten.add.Scalar, lane, offset),
                        end,
                    ),
                    call(
                        torch.ops.aten.maximum.default,
                        call(torch.ops.aten.sub.Scalar, lane, offset),
                        begin,
                    ),
                    call(torch.ops.aten.bitwise_xor.Scalar, lane, offset),
                )
            )
        for warp in range(32):
            maps.append(
                call(
                    torch.ops.aten.add.Scalar,
                    call(torch.ops.aten.mul.Scalar, lane, 0),
                    warp * 32,
                )
            )
        for node in maps:
            interval = integer_bounds(node)
            self.assertIsNotNone(interval)
            assert interval is not None
            self.assertLessEqual(0, interval[0])
            self.assertLess(interval[1], 1024)

    def test_padded_indices_do_not_read_outside_source(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for width in (18, 34, 66):
            with self.subTest(width=width):
                x = torch.arange(3 * width).reshape(3, width).int()
                _bound, code = _bounded_gather_codegen(_bounded_gather_shift, (x,))
                for order in (list(range(128)), list(reversed(range(128)))):
                    out = torch.full((3, width - 1), -999, dtype=x.dtype)
                    _simulate_register_load_program(
                        code, x, 128, host_tensors={"out": out}, lane_order=order
                    )
                    torch.testing.assert_close(out, x[:, 1:], rtol=0, atol=0)

    def test_tail_permutations_duplicates_and_source_lifetime(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ):
            for width, mode in (
                (17, "rotate"),
                (65, "duplicate"),
                (64, "xor"),
                (33, "data"),
            ):
                with self.subTest(dtype=dtype, width=width, mode=mode):
                    x = torch.arange(2 * width).reshape(2, width).to(dtype)
                    bound, code = _bounded_gather_codegen(
                        _bounded_gather_rows, (x, mode)
                    )
                    self.assertTrue(
                        bound.config_spec.cute_fragment_bounded_gather_root_ids
                    )
                    source = x * 3 + 1
                    lane = torch.arange(width).expand(2, width)
                    if mode == "duplicate":
                        index = lane % 3
                    elif mode == "xor":
                        index = lane ^ 1
                    elif mode == "data":
                        index = source.long() % width
                    else:
                        index = (lane * 5 + 3) % width
                    expected = torch.gather(source, -1, index) + source
                    for order in (list(range(128)), list(reversed(range(128)))):
                        out = torch.full_like(x, -999)
                        _simulate_register_load_program(
                            code, x, 128, host_tensors={"out": out}, lane_order=order
                        )
                        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_independent_leading_axes_and_repeated_gather_lifetimes(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for shape in ((2, 3, 17), (1, 5, 65)):
            for dtype in (torch.float32, torch.int64):
                with self.subTest(shape=shape, dtype=dtype):
                    x = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
                    _bound, code = _bounded_gather_codegen(
                        _bounded_gather_leading, (x,)
                    )
                    source = x * 3 + 1
                    first = torch.gather(source, -1, source.long() % shape[-1])
                    expected = first + torch.gather(
                        first + source, -1, (source.long() * 5 + 2) % shape[-1]
                    )
                    for order in (list(range(128)), list(reversed(range(128)))):
                        out = torch.full_like(x, -999)
                        _simulate_register_load_program(
                            code, x, 128, host_tensors={"out": out}, lane_order=order
                        )
                        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_unproved_bounds_and_invalid_index_dtype_reject(self):
        for mode in ("negative", "upper", "raw", "int32"):
            with (
                self.subTest(mode=mode),
                self.assertRaises((exc.InvalidConfig, exc.TorchOpTracingError)),
            ):
                _bounded_gather_codegen(_bounded_gather_bad, (torch.ones(2, 17), mode))

    def test_integer_range_wrap_and_modulo_proofs(self):
        from helion._compiler.cute.bounded_gather import integer_bounds

        for dtype in (torch.int32, torch.int64):
            graph = torch.fx.Graph()
            raw = graph.placeholder("raw")
            raw.meta["val"] = torch.empty(17, dtype=dtype)
            modulo = graph.call_function(torch.ops.aten.remainder.Scalar, (raw, 17))
            modulo.meta["val"] = torch.empty(17, dtype=dtype)
            self.assertEqual(integer_bounds(modulo), (0, 16))
            bad = graph.call_function(
                torch.ops.aten.add.Scalar, (modulo, torch.iinfo(dtype).max)
            )
            bad.meta["val"] = torch.empty(17, dtype=dtype)
            self.assertIsNone(integer_bounds(bad))
            shift = graph.call_function(torch.ops.aten.sub.Scalar, (modulo, 1))
            shift.meta["val"] = torch.empty(17, dtype=dtype)
            self.assertEqual(integer_bounds(shift), (-1, 15))

    def test_strided_inputs_and_empty_or_unproved_domains(self):
        from test.test_atomic_ops import _simulate_register_load_program

        storage = torch.arange(3 * 66).reshape(3, 66).int()
        for x in (storage[:, 1::2], storage.t()):
            with self.subTest(stride=x.stride()):
                _bound, code = _bounded_gather_codegen(
                    _bounded_gather_rows, (x, "data")
                )
                source = x * 3 + 1
                expected = torch.gather(source, -1, source.long() % x.size(1)) + source
                out = torch.empty_like(x)
                _simulate_register_load_program(code, x, 128, host_tensors={"out": out})
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
        with self.assertRaises(
            (exc.InvalidConfig, exc.BackendUnsupported, exc.TorchOpTracingError)
        ):
            _bounded_gather_codegen(_bounded_gather_rows, (torch.empty(2, 0), "data"))

    def test_default_strict_and_deferred_population_prefix(self):
        from copy import deepcopy
        import random

        from test.cute_population_contracts import checked_initial_population

        from helion._compiler import autotuner_heuristics
        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        key = "cute_fragment_bounded_gather"
        args = (torch.ones(2, 33), "data")

        def capture(enabled):
            with patch(
                "helion._compiler.autotuner_heuristics.register_fragment_bounded_gather_coverage",
                wraps=autotuner_heuristics.register_fragment_bounded_gather_coverage
                if enabled
                else None,
            ) as hook:
                if not enabled:
                    hook.return_value = None
                bound = _cpu_bind(_bounded_gather_rows, args)
            spec = bound.config_spec
            default = spec.default_config()
            self.assertNotIn(key, default)
            for value in (1, None, "yes"):
                invalid = deepcopy(default)
                invalid.config[key] = value
                with self.assertRaises(exc.InvalidConfig):
                    spec.normalized_config(invalid)
            records = []
            for seed in (0, 91):
                for strategy in (
                    InitialPopulationStrategy.FROM_RANDOM,
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                ):
                    random.seed(seed)
                    with bound.env:
                        search = PatternSearch(
                            bound,
                            args,
                            initial_population=100,
                            initial_population_strategy=strategy,
                        )
                        rows = [
                            dict(search.config_gen.canonicalize_flat(row)[1])
                            for row in checked_initial_population(search)
                        ]
                    records.append((rows, random.getstate()))
            if enabled:
                group = spec.compiler_coverage_groups[-1]
                self.assertEqual(group.key, key)
                self.assertTrue(group.deferred)
                requested = group.witnesses[0].carrier
                requested.config[key] = True
                self.assertIn("fragment_buffer", bound.to_code(requested))
            return records

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            previous, current = capture(False), capture(True)
        for (old, old_rng), (new, new_rng) in zip(previous, current, strict=True):
            self.assertEqual(old, new[: len(old)])
            self.assertEqual(old_rng, new_rng)
            self.assertTrue(any(row.get(key) is True for row in new[len(old) :]))


@onlyBackends("cute")
class TestCuteBoundedGatherNative(TestCase):
    def test_cross_lane_dtypes_and_tail(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ):
            with self.subTest(dtype=dtype):
                x = (torch.arange(66, device=DEVICE).reshape(2, 33) % 17).to(dtype)
                source = x * 3 + 1
                expected = torch.gather(source, -1, source.long() % 33) + source
                before = x.clone()
                _, actual = code_and_output(
                    _bounded_gather_rows,
                    (x, "data"),
                    cute_fragment_bounded_gather=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_leading_axes_and_repeated_producer(self):
        with self.subTest(shape=(2, 3, 17)):
            x = torch.arange(102, device=DEVICE).reshape(2, 3, 17).long() + 2**24
            source = x * 3 + 1
            first = torch.gather(source, -1, source % 17)
            expected = first + torch.gather(first + source, -1, (source * 5 + 2) % 17)
            before = x.clone()
            _, actual = code_and_output(
                _bounded_gather_leading,
                (x,),
                cute_fragment_bounded_gather=True,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_padding_and_exchange_maps(self):
        with self.subTest(mode="padded"):
            x = torch.arange(54, device=DEVICE).reshape(3, 18).int()
            _, actual = code_and_output(
                _bounded_gather_shift,
                (x,),
                cute_fragment_bounded_gather=True,
            )
            torch.testing.assert_close(actual, x[:, 1:], rtol=0, atol=0)
        lane = torch.arange(1024, device=DEVICE)
        x = torch.arange(2048, device=DEVICE).reshape(2, 1024).float()
        for mode, index in (
            ("down", torch.minimum(lane + 16, lane | 31)),
            ("up", torch.maximum(lane - 16, lane & -32)),
            ("xor", lane ^ 16),
        ):
            with self.subTest(mode=mode):
                _, actual = code_and_output(
                    _bounded_gather_maps,
                    (x, mode, 16),
                    cute_fragment_bounded_gather=True,
                )
                torch.testing.assert_close(actual, x[:, index], rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _indexed_bounded_gather(x: torch.Tensor, mode: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))
        if mode == "masked" or mode == "filled":
            valid = (lane % 3 != 0) & (lane + 1 < x.size(1))
            loaded = hl.load(x, [row, lane + 1], extra_mask=valid)
            if mode == "filled":
                source = torch.where(valid, loaded, -7)
            else:
                source = loaded
        elif mode == "reverse":
            source = x[row, x.size(1) - 1 - lane]
        else:
            source = x[row, lane]
        index = (lane * 5 + 3) % x.size(1)
        first = torch.gather(source, 0, index.long())
        if mode == "repeated":
            second = torch.gather(first + source, 0, (lane % 3).long())
            out[row, lane] = second + first
        else:
            out[row, lane] = first
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _indexed_bounded_leading(x: torch.Tensor, broadcast: hl.constexpr):
    out = torch.empty_like(x)
    for batch in hl.grid(x.size(0)):
        rows = hl.arange(x.size(1))
        columns = hl.arange(x.size(2))
        if broadcast:
            source = x[batch, rows[:, None], columns[None, :]]
        else:
            source = x[batch, rows, columns]
        index = ((rows[:, None] * 0 + columns[None, :] * 5 + 3) % x.size(2)).long()
        out[batch, rows, columns] = torch.gather(source, -1, index)
    return out


def _indexed_bounded_expected(x, mode):
    lane = torch.arange(x.size(1), device=x.device)
    if mode == "masked" or mode == "filled":
        source = torch.full_like(x, -7 if mode == "filled" else 0)
        valid = (lane + 1 < x.size(1)) & (lane % 3 != 0)
        source[:, valid] = x[:, lane[valid] + 1]
    elif mode == "reverse":
        source = x.flip(-1)
    else:
        source = x
    first = source[:, (lane * 5 + 3) % x.size(1)]
    if mode == "repeated":
        return (first + source)[:, lane % 3] + first
    return first


class TestCuteIndexedGatherCPU(TestCase):
    def test_indexed_masked_tail_and_repeated_producers(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for width, mode in (
            (17, "masked"),
            (33, "reverse"),
            (64, "repeated"),
            (1024, "plain"),
            (64, "filled"),
        ):
            for dtype in (torch.float32, torch.int64):
                with self.subTest(width=width, mode=mode, dtype=dtype):
                    x = torch.arange(3 * width).reshape(3, width).to(dtype)
                    _bound, code = _bounded_gather_codegen(
                        _indexed_bounded_gather, (x, mode)
                    )
                    expected = _indexed_bounded_expected(x, mode)
                    for order in (list(range(128)), list(reversed(range(128)))):
                        out = torch.full_like(x, -999)
                        _simulate_register_load_program(
                            code, x, 128, host_tensors={"out": out}, lane_order=order
                        )
                        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_indexed_cartesian_and_broadcast_leading_axes(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for shape in ((2, 3, 17), (1, 5, 33), (2, 1, 17), (2, 3, 1)):
            for broadcast in (False, True):
                with self.subTest(shape=shape, broadcast=broadcast):
                    x = torch.arange(math.prod(shape)).reshape(shape).float()
                    _bound, code = _bounded_gather_codegen(
                        _indexed_bounded_leading, (x, broadcast)
                    )
                    expected = x[:, :, (torch.arange(shape[-1]) * 5 + 3) % shape[-1]]
                    out = torch.full_like(x, -999)
                    _simulate_register_load_program(
                        code,
                        x,
                        128,
                        host_tensors={"out": out},
                        lane_order=list(reversed(range(128))),
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_indexed_strides_offsets_and_float_payload(self):
        from test.test_atomic_ops import _simulate_register_load_program

        storage = torch.arange(3 * 66).reshape(3, 66).int()
        for x in (storage[:, 1::2], storage.t()):
            with self.subTest(stride=x.stride()):
                _bound, code = _bounded_gather_codegen(
                    _indexed_bounded_gather, (x, "reverse")
                )
                out = torch.empty_like(x)
                _simulate_register_load_program(code, x, 128, host_tensors={"out": out})
                torch.testing.assert_close(
                    out, _indexed_bounded_expected(x, "reverse"), rtol=0, atol=0
                )
        bits = (
            torch.tensor(
                [
                    0,
                    -2147483648,
                    1,
                    0x7FFFFF,
                    0x800000,
                    0x7F800000,
                    -8388608,
                    0x7FC00000,
                ],
                dtype=torch.int32,
            )
            .repeat(8)
            .reshape(2, 32)
        )
        x = bits.view(torch.float32)
        _bound, code = _bounded_gather_codegen(_indexed_bounded_gather, (x, "reverse"))
        out = torch.empty_like(x)
        _simulate_register_load_program(code, x, 128, host_tensors={"out": out})
        torch.testing.assert_close(
            out.view(torch.int32),
            _indexed_bounded_expected(x, "reverse").view(torch.int32),
            rtol=0,
            atol=0,
        )

    def test_index_values_do_not_define_domains_and_unknowns_decline(self):
        import sympy

        from helion._compiler.cute.bounded_gather import indexed_load_shapes

        bound, _code = _bounded_gather_codegen(
            _indexed_bounded_gather, (torch.ones(2, 17), "masked")
        )
        graph = torch.fx.Graph()
        index = graph.placeholder("index")
        index.meta["val"] = torch.empty(32, dtype=torch.int64)
        with bound.env:

            def infer(node):
                return (sympy.Integer(17),) if node is index else None

            self.assertEqual(
                indexed_load_shapes(bound.env, [index], infer, sympy.sympify),
                {0: (sympy.Integer(17),)},
            )
            self.assertIsNone(
                indexed_load_shapes(
                    bound.env, [index], lambda node: None, sympy.sympify
                )
            )
            # Equal padded capacities cannot reconcile unequal logical axes.
            left = graph.placeholder("left")
            right = graph.placeholder("right")
            left.meta["val"] = torch.empty(2, 32, dtype=torch.int64)
            right.meta["val"] = torch.empty(2, 32, dtype=torch.int64)
            shapes = {
                left: (sympy.Integer(2), sympy.Integer(17)),
                right: (sympy.Integer(2), sympy.Integer(18)),
            }
            self.assertIsNone(
                indexed_load_shapes(bound.env, [left, right], shapes.get, sympy.sympify)
            )
            # A logical singleton cannot reuse a non-singleton physical axis.
            self.assertIsNone(
                indexed_load_shapes(
                    bound.env, [index], lambda node: (sympy.Integer(1),), sympy.sympify
                )
            )


@onlyBackends("cute")
class TestCuteIndexedGatherNative(TestCase):
    def test_indexed_tail_masks_and_repeated(self):
        for mode in ("masked", "reverse", "repeated", "filled"):
            with self.subTest(mode=mode):
                width = 64 if mode in ("repeated", "filled") else 33
                x = torch.arange(3 * width, device=DEVICE).reshape(3, width).float()
                before = x.clone()
                _, actual = code_and_output(
                    _indexed_bounded_gather,
                    (x, mode),
                    cute_fragment_bounded_gather=True,
                )
                torch.testing.assert_close(
                    actual, _indexed_bounded_expected(x, mode), rtol=0, atol=0
                )
                torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_indexed_leading_mapping(self):
        for broadcast in (False, True):
            with self.subTest(broadcast=broadcast):
                x = torch.arange(102, device=DEVICE).reshape(2, 3, 17).long()
                before = x.clone()
                _, actual = code_and_output(
                    _indexed_bounded_leading,
                    (x, broadcast),
                    cute_fragment_bounded_gather=True,
                )
                expected = x[:, :, (torch.arange(17, device=DEVICE) * 5 + 3) % 17]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_bounded_exchange(x: torch.Tensor, steps: int, mode: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))[:]
        left = hl.full([x.size(1)], 1, dtype=x.dtype)
        right = x[row, :]
        for _step in range(steps):
            if mode == "swap":
                previous = left
                left = right
                right = previous
            else:
                left = left + x[row, lane]
                index = (left.long() * 0 + lane * 5 + 3) % x.size(1)
                left = torch.gather(left, -1, index)
        if mode == "swap":
            index = (lane * 5 + 3) % x.size(1)
            out[row, :] = torch.gather(left + right * 2, -1, index.long())
        else:
            out[row, :] = left + right
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_bounded_index_drift(x: torch.Tensor, clamp: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1)).long()
        source = x[row, :]
        result = x[row, :]
        for _step in range(2):
            index = index + 1
            if clamp:
                selected = index % x.size(1)
            else:
                selected = index
            result = torch.gather(source, -1, selected)
        out[row, :] = result
    return out


class TestCuteLoopGatherDomainsCPU(TestCase):
    def test_simultaneous_carries_keep_parallel_assignment(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for steps in (0, 1, 2, 3):
            with self.subTest(steps=steps):
                x = torch.arange(128).reshape(2, 64).long()
                _bound, code = _bounded_gather_codegen(
                    _loop_bounded_exchange, (x, steps, "swap")
                )
                left, right = torch.ones_like(x), x
                for _ in range(steps):
                    left, right = right, left
                expected = (left + right * 2)[:, (torch.arange(64) * 5 + 3) % 64]
                out = torch.empty_like(x)
                _simulate_register_load_program(
                    code,
                    x,
                    128,
                    host_tensors={"out": out},
                    scalar_args={"steps": steps},
                    lane_order=list(reversed(range(128))),
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_loop_entry_backedges_zero_trip_and_tails(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for width in (17, 64):
            for steps in (0, 1, 3):
                for dtype in (torch.float32, torch.int64):
                    with self.subTest(width=width, steps=steps, dtype=dtype):
                        x = torch.arange(2 * width).reshape(2, width).to(dtype)
                        _bound, code = _bounded_gather_codegen(
                            _loop_bounded_exchange, (x, steps, "gather")
                        )
                        expected = torch.ones_like(x)
                        index = (torch.arange(width) * 5 + 3) % width
                        for _ in range(steps):
                            expected = (expected + x)[:, index]
                        expected += x
                        for order in (list(range(128)), list(reversed(range(128)))):
                            out = torch.full_like(x, -999)
                            _simulate_register_load_program(
                                code,
                                x,
                                128,
                                host_tensors={"out": out},
                                lane_order=order,
                                scalar_args={"steps": steps},
                            )
                            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_mutable_index_range_does_not_inherit_entry(self):
        from test.test_atomic_ops import _simulate_register_load_program

        x = torch.arange(128).reshape(2, 64).int()
        with self.assertRaises(exc.InvalidConfig):
            _bounded_gather_codegen(_loop_bounded_index_drift, (x, False))
        _bound, code = _bounded_gather_codegen(_loop_bounded_index_drift, (x, True))
        out = torch.empty_like(x)
        _simulate_register_load_program(
            code,
            x,
            128,
            host_tensors={"out": out},
            lane_order=list(reversed(range(128))),
        )
        torch.testing.assert_close(
            out, x[:, (torch.arange(64) + 2) % 64], rtol=0, atol=0
        )

    def test_actual_phi_mapping_and_all_backedges_are_required(self):
        import operator

        from helion._compiler.cute.gather_domains import loop_domain_facts
        from helion._compiler.device_ir import ForLoopGraphInfo
        from helion._compiler.device_ir import RootGraphInfo
        from helion.language import _tracing_ops
        from helion.language import creation_ops

        bound, _code = _bounded_gather_codegen(
            _indexed_bounded_gather, (torch.ones(2, 64), "plain")
        )
        cases = (
            "valid",
            "stale_metadata",
            "arity",
            "multiple_calls",
            "missing_phi",
            "duplicate_entry",
            "backedge_shape",
            "backedge_dtype",
            "logical_backedge",
            "placeholder_dtype",
            "output_arity",
            "phi_position",
            "duplicate_graph",
            "misplaced_graph",
            "nested",
        )
        for bad in cases:
            with self.subTest(case=bad), bound.env:
                root = torch.fx.Graph()
                a = root.call_function(
                    creation_ops.full, ([18 if bad == "logical_backedge" else 64], 0)
                )
                b = root.call_function(creation_ops.full, ([64], 1))
                lane = root.call_function(torch.ops.prims.iota.default, (64,))
                for n in (a, b, lane):
                    n.meta["val"] = torch.empty(64, dtype=torch.int64)
                child = torch.fx.Graph()
                # Read-only lane deliberately precedes the two mutable entries.
                lp, bp, ap = (child.placeholder(n) for n in ("lane", "b", "a"))
                for n in (lp, bp, ap):
                    n.meta["val"] = torch.empty(64, dtype=torch.int64)
                if bad == "placeholder_dtype":
                    ap.meta["val"] = torch.empty(64, dtype=torch.float32)
                first = child.call_function(
                    creation_ops.full,
                    (
                        [
                            32
                            if bad == "backedge_shape"
                            else 17
                            if bad == "logical_backedge"
                            else 64
                        ],
                        2,
                    ),
                )
                first.meta["val"] = torch.empty(
                    32 if bad == "backedge_shape" else 64,
                    dtype=torch.float32 if bad == "backedge_dtype" else torch.int64,
                )
                second = child.call_function(_tracing_ops._new_var, (bp,))
                second.meta["val"] = torch.empty(64, dtype=torch.int64)
                if bad == "nested":
                    child.call_function(_tracing_ops._for_loop, (2, [0], [2], []))
                child.output([first] if bad == "output_arity" else [first, second])
                captures = [lane, b, a]
                call = root.call_function(
                    _tracing_ops._for_loop,
                    (1, [0], [2], captures[:-1] if bad == "arity" else captures),
                )
                i0 = root.call_function(operator.getitem, (call, 0))
                i1 = root.call_function(operator.getitem, (call, 1))
                for n in (i0, i1):
                    n.meta["val"] = torch.empty(64, dtype=torch.int64)
                p0 = root.call_function(
                    _tracing_ops._phi, (i0, a) if bad == "phi_position" else (a, i0)
                )
                p0.meta["val"] = torch.empty(64, dtype=torch.int64)
                if bad != "missing_phi":
                    p1 = root.call_function(
                        _tracing_ops._phi, (a if bad == "duplicate_entry" else b, i1)
                    )
                    p1.meta["val"] = torch.empty(64, dtype=torch.int64)
                if bad == "multiple_calls":
                    root.call_function(_tracing_ops._for_loop, (1, [0], [2], captures))
                root.output([])
                info = ForLoopGraphInfo(
                    graph_id=1,
                    graph=child,
                    node_args=[lane, a, b] if bad == "stale_metadata" else captures,
                    block_ids=[],
                )
                graphs = [RootGraphInfo(graph_id=0, graph=root), info]
                if bad == "duplicate_graph":
                    graphs.append(info)
                if bad == "misplaced_graph":
                    graphs.reverse()
                facts = loop_domain_facts(bound.env, graphs)
                if bad in ("valid", "stale_metadata"):
                    self.assertIn(lp, facts.shapes)
                    self.assertEqual(facts.readonly_ranges[lp], (0, 63))
                    self.assertNotIn(ap, facts.readonly_ranges)
                    self.assertNotIn(bp, facts.readonly_ranges)
                    self.assertIn(p0, facts.shapes)
                else:
                    self.assertEqual(facts.shapes, {})
                    self.assertEqual(facts.readonly_ranges, {})


@onlyBackends("cute")
class TestCuteLoopGatherDomainsNative(TestCase):
    def test_loop_exchange_and_zero_trip(self):
        for steps in (0, 1, 3):
            with self.subTest(steps=steps):
                x = torch.arange(128, device=DEVICE).reshape(2, 64).int()
                expected = torch.ones_like(x)
                index = (torch.arange(64, device=DEVICE) * 5 + 3) % 64
                for _ in range(steps):
                    expected = (expected + x)[:, index]
                before = x.clone()
                _, actual = code_and_output(
                    _loop_bounded_exchange,
                    (x, steps, "gather"),
                    cute_fragment_bounded_gather=True,
                )
                torch.testing.assert_close(actual, expected + x, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _asm_bounded_gather(x: torch.Tensor, mode: hl.constexpr):
    out = torch.empty_like(x, dtype=torch.float32 if mode == "convert" else x.dtype)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))[:]
        if mode == "convert":
            value = hl.inline_asm_elementwise(
                "cvt.rn.f32.s32 $0, $1;",
                "=f,r",
                [x[row, :]],
                dtype=torch.float32,
                is_pure=True,
                pack=1,
            )
        else:
            scalar = hl.full([], 3, dtype=torch.int32)
            bias = hl.inline_asm_elementwise(
                "add.s32 $0, $1, $2;",
                "=r,r,r",
                [scalar, scalar],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            value = hl.inline_asm_elementwise(
                "add.s32 $0, $1, $2;",
                "=r,r,r",
                [x[row, :], bias],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
        out[row, :] = torch.gather(value, -1, ((lane * 5 + 3) % x.size(1)).long())
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _asm_bounded_leading(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(2))[None, :]
        bias = hl.full([x.size(1), 1], 7, dtype=torch.int32)
        value = hl.inline_asm_elementwise(
            "add.s32 $0, $1, $2;",
            "=r,r,r",
            [x[row, :, :], bias],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        index = ((value.long() * 0 + lane * 5 + 3) % x.size(2)).long()
        out[row, :, :] = torch.gather(value, -1, index)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _asm_bounded_loop(x: torch.Tensor, steps: int):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))[:]
        value = hl.full([x.size(1)], 1, dtype=torch.int32)
        for _step in range(steps):
            value = hl.inline_asm_elementwise(
                "add.s32 $0, $1, $2;",
                "=r,r,r",
                [value, x[row, :]],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            value = torch.gather(value, -1, ((lane * 5 + 3) % x.size(1)).long())
        out[row, :] = value
    return out


def _asm_gather_model_source(code):
    """Model only these tests' declared integer add/conversion scalar ops."""
    import ast

    class Model(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id != "_cute_inline_asm_elementwise"
            ):
                return node
            keywords = {item.arg: item.value for item in node.keywords}
            assert ast.literal_eval(keywords["is_pure"]) is True
            asm = ast.literal_eval(keywords["asm"])
            operands = node.args[0].elts
            if asm == "add.s32 $0, $1, $2;":
                assert len(operands) == 2
                expression = ast.BinOp(operands[0], ast.Add(), operands[1])
            else:
                assert asm == "cvt.rn.f32.s32 $0, $1;" and len(operands) == 1
                expression = operands[0]
            return ast.Call(keywords["dtype"], [expression], [])

    return ast.unparse(ast.fix_missing_locations(Model().visit(ast.parse(code))))


class TestCuteInlineAsmGatherCPU(TestCase):
    def test_exact_contract_and_opaque_range_declines(self):
        import operator

        import sympy

        from helion._compiler.cute.bounded_gather import integer_bounds
        from helion._compiler.cute.computed_fragment import _fragment_logical_shape
        from helion.language import creation_ops
        from helion.language import inline_asm_ops

        bound, _code = _bounded_gather_codegen(
            _indexed_bounded_gather, (torch.ones(2, 64), "plain")
        )
        cases = (
            "valid",
            "scalar",
            "conversion",
            "unknown",
            "logical_mismatch",
            "singleton_mismatch",
            "physical_output",
            "output_dtype",
            "tuple",
            "tuple_getitem",
            "effectful",
            "pack2",
            "pack4",
            "empty",
        )
        for case in cases:
            with self.subTest(case=case), bound.env:
                graph = torch.fx.Graph()
                shape = [] if case == "scalar" else [17]
                capacity = () if case == "scalar" else (32,)
                x = graph.call_function(creation_ops.full, (shape, 3))
                x.meta["val"] = torch.empty(capacity, dtype=torch.int32)
                y = graph.call_function(creation_ops.full, (shape, 5))
                y.meta["val"] = torch.empty(capacity, dtype=torch.int32)
                if case == "unknown":
                    y = graph.placeholder("unknown")
                    y.meta["val"] = torch.empty(capacity, dtype=torch.int32)
                if case == "logical_mismatch":
                    y.args = ([18], 5)
                if case == "singleton_mismatch":
                    y.args = ([1], 5)
                dtype = torch.float32 if case == "conversion" else torch.int32
                declared = (
                    (dtype, dtype) if case in ("tuple", "tuple_getitem") else dtype
                )
                operands = [] if case == "empty" else [x, y]
                pack = 2 if case == "pack2" else 4 if case == "pack4" else 1
                node = graph.call_function(
                    inline_asm_ops.inline_asm_elementwise,
                    (
                        "opaque program",
                        "constraints",
                        operands,
                        declared,
                        case != "effectful",
                        pack,
                    ),
                )
                node.meta["val"] = torch.empty(
                    (64,) if case == "physical_output" else capacity,
                    dtype=torch.float64 if case == "output_dtype" else dtype,
                )
                if case in ("tuple", "tuple_getitem"):
                    node.meta["val"] = (node.meta["val"], node.meta["val"])
                if case == "tuple_getitem":
                    parent = node
                    node = graph.call_function(operator.getitem, (parent, 0))
                    node.meta["val"] = parent.meta["val"][0]
                self.assertIsNone(_fragment_logical_shape(bound.env, node))
                actual = _fragment_logical_shape(bound.env, node, pure_inline_asm=True)
                if case in ("valid", "scalar", "conversion"):
                    self.assertEqual(actual, tuple(map(sympy.Integer, shape)))
                else:
                    self.assertIsNone(actual)
                # A proved domain does not prove any opaque assembly value.
                self.assertIsNone(integer_bounds(node))

    def test_scalar_vector_dtype_tails_and_leading_broadcast(self):
        from test.test_atomic_ops import _simulate_register_load_program

        cases = [
            (
                _asm_bounded_gather,
                (torch.arange(3 * width).reshape(3, width).int(), mode),
            )
            for width in (17, 33)
            for mode in ("scalar", "convert")
        ]
        cases += [
            (_asm_bounded_leading, (torch.arange(2 * 3 * 17).reshape(2, 3, 17).int(),))
        ]
        for kernel, args in cases:
            with self.subTest(
                kernel=kernel.fn.__name__, shape=args[0].shape, mode=args[1:]
            ):
                x = args[0]
                _bound, code = _bounded_gather_codegen(kernel, args)
                source = _asm_gather_model_source(code)
                if kernel is _asm_bounded_leading:
                    expected = (x + 7)[
                        :, :, (torch.arange(x.size(-1)) * 5 + 3) % x.size(-1)
                    ]
                else:
                    expected = x.float() if args[1] == "convert" else x + 6
                    expected = expected[
                        :, (torch.arange(x.size(-1)) * 5 + 3) % x.size(-1)
                    ]
                for order in (list(range(128)), list(reversed(range(128)))):
                    out = torch.empty_like(expected)
                    _simulate_register_load_program(
                        source, x, 128, host_tensors={"out": out}, lane_order=order
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_loop_assembly_domains_and_zero_trip(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for width in (17, 64):
            for steps in (0, 1, 3):
                with self.subTest(width=width, steps=steps):
                    x = torch.arange(2 * width).reshape(2, width).int()
                    _bound, code = _bounded_gather_codegen(
                        _asm_bounded_loop, (x, steps)
                    )
                    expected = torch.ones_like(x)
                    for _ in range(steps):
                        expected = (expected + x)[
                            :, (torch.arange(width) * 5 + 3) % width
                        ]
                    out = torch.empty_like(x)
                    _simulate_register_load_program(
                        _asm_gather_model_source(code),
                        x,
                        128,
                        host_tensors={"out": out},
                        scalar_args={"steps": steps},
                        lane_order=list(reversed(range(128))),
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@onlyBackends("cute")
class TestCuteInlineAsmGatherNative(TestCase):
    def test_scalar_vector_tail_and_loop(self):
        for width, mode in ((17, "scalar"), (33, "convert")):
            with self.subTest(width=width, mode=mode):
                x = torch.arange(2 * width, device=DEVICE).reshape(2, width).int()
                before = x.clone()
                _, actual = code_and_output(
                    _asm_bounded_gather, (x, mode), cute_fragment_bounded_gather=True
                )
                expected = x.float() if mode == "convert" else x + 6
                expected = expected[
                    :, (torch.arange(width, device=DEVICE) * 5 + 3) % width
                ]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
        with self.subTest(leading=True):
            x = torch.arange(102, device=DEVICE).reshape(2, 3, 17).int()
            before = x.clone()
            _, actual = code_and_output(
                _asm_bounded_leading, (x,), cute_fragment_bounded_gather=True
            )
            expected = (x + 7)[:, :, (torch.arange(17, device=DEVICE) * 5 + 3) % 17]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x, before, rtol=0, atol=0)
        for steps in (0, 3):
            with self.subTest(steps=steps):
                x = torch.arange(34, device=DEVICE).reshape(2, 17).int()
                before = x.clone()
                _, actual = code_and_output(
                    _asm_bounded_loop, (x, steps), cute_fragment_bounded_gather=True
                )
                expected = torch.ones_like(x)
                for _ in range(steps):
                    expected = (expected + x)[
                        :, (torch.arange(17, device=DEVICE) * 5 + 3) % 17
                    ]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_warp_scan(x: torch.Tensor, reverse: hl.constexpr):
    out = torch.empty_like(x)
    reused = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))
        values = hl.load(x, [row, lane])
        index = ((lane * 3 + 5) % x.size(1)).long()
        chosen = torch.gather(values, 0, index)
        out[row, lane] = hl.cumsum(chosen, dim=0, reverse=reverse)
        reused[row, lane] = torch.gather(values, 0, index)
    return out, reused


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather_mixed_scan(x: torch.Tensor, unsafe: hl.constexpr):
    partial = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        partial[row, :] = hl.cumsum(x[row, :] + 1, dim=-1)
    hl.barrier()
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))
        values = hl.load(partial, [row, lane])
        if unsafe:
            index = (lane + 1).long()
        else:
            index = ((lane * 3 + 5) % x.size(1)).long()
        out[row, lane] = hl.cumsum(torch.gather(values, 0, index), dim=0)
    return out


class TestCuteGatherWarpScanCPU(TestCase):
    def test_explicit_options_and_wide_generated_model(self):
        from test.test_cute_computed_fragment import _simulate_fragment_warp_reduction

        for dtype in (torch.int32, torch.int64):
            for reverse in (False, True):
                with self.subTest(dtype=dtype, reverse=reverse):
                    x = (torch.arange(2 * 2049).reshape(2, 2049) % 29 - 14).to(dtype)
                    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
                        bound = _cpu_bind(_bounded_gather_warp_scan, (x, reverse))
                        for gather in (False, True):
                            for scan in (False, True):
                                config = bound.config_spec.default_config()
                                config.config.update(
                                    pid_type="flat",
                                    cute_fragment_bounded_gather=gather,
                                    cute_fragment_warp_scan=scan,
                                )
                                before = dict(config)
                                if not gather:
                                    with self.assertRaises(
                                        (exc.InvalidConfig, exc.BackendUnsupported)
                                    ):
                                        bound.to_code(config)
                                    self.assertEqual(dict(config), before)
                                    continue
                                code = bound.to_code(config)
                                out, reused = torch.empty_like(x), torch.empty_like(x)
                                _simulate_fragment_warp_reduction(
                                    code,
                                    {"x": x},
                                    {"out": out, "reused": reused},
                                    2,
                                    allow_lane_stores=True,
                                )
                                selected = x[:, (torch.arange(2049) * 3 + 5) % 2049]
                                expected = (
                                    selected.flip(-1).cumsum(-1, dtype=dtype).flip(-1)
                                    if reverse
                                    else selected.cumsum(-1, dtype=dtype)
                                )
                                self.assertTrue(torch.equal(out, expected))
                                self.assertTrue(torch.equal(reused, selected))
                                self.assertEqual("shuffle_sync_up" in code, scan)

    def test_dependent_carrier_and_strict_mutation(self):
        from helion._compiler.autotuner_heuristics.cute_fragment_warp_scan import (
            register_gather_warp_scan_coverage,
        )

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(
                _bounded_gather_warp_scan, (torch.ones(2, 2049).int(), False)
            )
            spec = bound.config_spec
            self.assertFalse(spec.cute_fragment_warp_scan_root_ids)
            groups = spec.compiler_coverage_groups
            self.assertEqual(
                [g.key for g in groups][-2:],
                ["cute_fragment_bounded_gather", "cute_fragment_warp_scan"],
            )
            group = groups[-1]
            self.assertEqual(len(group.dependencies), 1)
            dependency = group.dependencies[0]
            self.assertEqual(dependency.key, "cute_fragment_bounded_gather")
            self.assertIs(dependency.value, True)
            carrier = group.witnesses[0].carrier
            self.assertIs(carrier.get(dependency.key), True)
            requested = helion.Config.from_dict(carrier.config | {group.key: True})
            spec.create_config_generation().strict_config_pair(requested)
            self.assertIn("shuffle_sync_up", bound.to_code(requested))
            for missing in (False, 1, None):
                bad = helion.Config.from_dict(
                    requested.config | {dependency.key: missing}
                )
                before = dict(bad)
                with self.assertRaises(exc.InvalidConfig):
                    spec.create_config_generation().strict_config_pair(bad)
                self.assertEqual(dict(bad), before)
            register_gather_warp_scan_coverage(bound.env, bound.host_function.device_ir)
            self.assertEqual(spec.compiler_coverage_groups, groups)

    def test_mixed_roots_and_unsupported_gather(self):
        from helion._compiler.autotuner_heuristics.cute_fragment_warp_scan import (
            active_warp_scan_roots,
        )

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for unsafe in (False, True):
                with self.subTest(unsafe=unsafe):
                    bound = _cpu_bind(
                        _bounded_gather_mixed_scan, (torch.ones(2, 65).int(), unsafe)
                    )
                    spec = bound.config_spec
                    self.assertEqual(len(spec.cute_fragment_warp_scan_root_ids), 1)
                    self.assertEqual(
                        len(spec.cute_fragment_warp_scan_requirements),
                        0 if unsafe else 1,
                    )
                    self.assertEqual(
                        sum(
                            g.key == "cute_fragment_warp_scan"
                            for g in spec.compiler_coverage_groups
                        ),
                        1,
                    )
                    for gather in (False, True):
                        config = spec.default_config()
                        config.config.update(
                            pid_type="flat",
                            cute_fragment_bounded_gather=gather,
                            cute_fragment_warp_scan=True,
                        )
                        self.assertEqual(
                            len(active_warp_scan_roots(bound.env, config)),
                            2 if gather and not unsafe else 1,
                        )
                        if gather and unsafe:
                            with self.assertRaisesRegex(
                                exc.InvalidConfig, "proved bounded last-axis gather"
                            ):
                                bound.to_code(config)
                            continue
                        code = bound.to_code(config)
                        phases = [
                            node.value
                            for node in ast.walk(ast.parse(code))
                            if isinstance(node, ast.Constant)
                            and isinstance(node.value, str)
                            and "@cute.kernel" in node.value
                        ]
                        self.assertEqual(len(phases), 2)
                        # The unproved gather retains ordinary lowering;
                        # only eligible phases use the fragment warp scan.
                        self.assertEqual(
                            sum("fragment_scan_lane" in phase for phase in phases),
                            2 if gather and not unsafe else 1,
                        )

    def test_mixed_phases_execute_and_keep_publication(self):
        from test.test_cute_computed_fragment import _simulate_fragment_warp_reduction

        x = (torch.arange(130).reshape(2, 65) % 11 - 5).int()
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(_bounded_gather_mixed_scan, (x, False))
            for scan in (False, True):
                with self.subTest(scan=scan):
                    config = bound.config_spec.default_config()
                    config.config.update(
                        pid_type="flat",
                        cute_fragment_bounded_gather=True,
                        cute_fragment_warp_scan=scan,
                    )
                    code = bound.to_code(config)
                    phases = [
                        node.value
                        for node in ast.walk(ast.parse(code))
                        if isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and "@cute.kernel" in node.value
                    ]
                    self.assertEqual(len(phases), 2)
                    partial = torch.full_like(x, -999)
                    out = torch.full_like(x, -777)
                    for phase in phases:
                        _simulate_fragment_warp_reduction(
                            phase,
                            {"x": x, "partial": partial},
                            {"partial": partial, "out": out},
                            2,
                            allow_lane_stores=True,
                        )
                        self.assertEqual("shuffle_sync_up" in phase, scan)
                    expected_partial = (x + 1).cumsum(-1, dtype=x.dtype)
                    expected = expected_partial[
                        :, (torch.arange(65) * 3 + 5) % 65
                    ].cumsum(-1, dtype=x.dtype)
                    self.assertTrue(torch.equal(partial, expected_partial))
                    self.assertTrue(torch.equal(out, expected))

    def test_unsupported_capacity_and_resource_decline(self):
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for width, dtype in (
                (0, torch.int32),
                (2049, torch.float32),
                (65537, torch.int64),
            ):
                with self.subTest(width=width, dtype=dtype):
                    bound = _cpu_bind(
                        _bounded_gather_warp_scan,
                        (torch.empty(1, width, dtype=dtype), False),
                    )
                    config = bound.config_spec.default_config()
                    config.config.update(
                        pid_type="flat",
                        cute_fragment_bounded_gather=True,
                        cute_fragment_warp_scan=True,
                    )
                    before = dict(config)
                    with self.assertRaises(exc.InvalidConfig):
                        bound.to_code(config)
                    self.assertEqual(dict(config), before)


@onlyBackends("cute")
class TestCuteGatherWarpScanNative(TestCase):
    def test_wide_tails_reverse_and_reuse(self):
        for dtype in (torch.int32, torch.int64):
            for reverse in (False, True):
                with self.subTest(dtype=dtype, reverse=reverse):
                    x = (
                        torch.arange(3 * 2049, device=DEVICE).reshape(3, 2049) % 29 - 14
                    ).to(dtype)
                    before = x.clone()
                    _, (out, reused) = code_and_output(
                        _bounded_gather_warp_scan,
                        (x, reverse),
                        cute_fragment_bounded_gather=True,
                        cute_fragment_warp_scan=True,
                    )
                    selected = x[:, (torch.arange(2049, device=DEVICE) * 3 + 5) % 2049]
                    expected = (
                        selected.flip(-1).cumsum(-1, dtype=dtype).flip(-1)
                        if reverse
                        else selected.cumsum(-1, dtype=dtype)
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    torch.testing.assert_close(reused, selected, rtol=0, atol=0)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_mixed_ordered_roots(self):
        x = (torch.arange(3 * 65, device=DEVICE).reshape(3, 65) % 11 - 5).int()
        before = x.clone()
        _, out = code_and_output(
            _bounded_gather_mixed_scan,
            (x, False),
            block_sizes=[1],
            pid_type="flat",
            cute_fragment_bounded_gather=True,
            cute_fragment_warp_scan=True,
        )
        expected = (
            (x + 1)
            .cumsum(-1, dtype=x.dtype)[
                :, (torch.arange(65, device=DEVICE) * 3 + 5) % 65
            ]
            .cumsum(-1, dtype=x.dtype)
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _all_axis_bounded_gather(
    x: torch.Tensor, op: hl.constexpr, mode: hl.constexpr, keep: hl.constexpr
):
    dtype = torch.int64 if op in ("sum", "prod") and x.dtype == torch.int32 else x.dtype
    out = torch.empty_like(x, dtype=dtype)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(-1))
        if x.ndim == 3:
            leading = hl.arange(x.size(1))
            value = x[row, leading, lane]
            index = ((leading[:, None] * 0 + lane[None, :] * 5 + 3) % x.size(-1)).long()
        else:
            value = x[row, lane]
            index = ((lane * 5 + 3) % x.size(-1)).long()
        if op == "min":
            if mode == "omitted":
                reduced = value.amin(keepdim=keep)
            elif mode == "none":
                reduced = value.amin(dim=None, keepdim=keep)
            else:
                reduced = value.amin(dim=[], keepdim=keep)
        elif op == "max":
            if mode == "omitted":
                reduced = value.amax(keepdim=keep)
            elif mode == "none":
                reduced = value.amax(dim=None, keepdim=keep)
            else:
                reduced = value.amax(dim=[], keepdim=keep)
        elif op == "sum":
            reduced = value.sum(dim=None, keepdim=True)
        else:
            reduced = value.prod()
        source = value + reduced
        result = torch.gather(source, -1, index)
        if x.ndim == 3:
            out[row, leading, lane] = result
        else:
            out[row, lane] = result
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _all_axis_bounded_loop(x: torch.Tensor, steps: int):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))[:]
        value = x[row, :]
        carry = hl.full([], 0, dtype=x.dtype)
        for _step in range(steps):
            carry = (value + carry).amin()
            value = torch.gather(value + carry, -1, ((lane * 5 + 3) % x.size(1)).long())
        out[row, :] = value
    return out


class TestCuteAllAxisGatherCPU(TestCase):
    def test_canonical_dims_keepdim_and_promotion(self):
        from test.test_atomic_ops import _simulate_register_load_program

        cases = [
            ((2, 16), op, mode, keep)
            for op in ("min", "max")
            for mode in ("omitted", "none", "empty")
            for keep in (False, True)
        ] + [
            ((2, 16), "sum", "none", True),
            ((2, 16), "prod", "omitted", False),
        ]
        for shape, op, mode, keep in cases:
            with self.subTest(shape=shape, op=op, mode=mode, keep=keep):
                x = torch.arange(math.prod(shape)).reshape(shape).remainder(3).int()
                if op == "prod":
                    x = x.remainder(2) + 1
                if op in ("min", "max"):
                    x = x.float()
                _bound, code = _bounded_gather_codegen(
                    _all_axis_bounded_gather, (x, op, mode, keep)
                )
                rows = []
                for value in x:
                    if op == "min":
                        reduced = value.amin(keepdim=keep)
                    elif op == "max":
                        reduced = value.amax(keepdim=keep)
                    elif op == "sum":
                        reduced = value.sum(dim=None, keepdim=True)
                    else:
                        reduced = value.prod()
                    source = value + reduced
                    rows.append(
                        source[..., (torch.arange(x.size(-1)) * 5 + 3) % x.size(-1)]
                    )
                expected = torch.stack(rows)
                out = torch.empty_like(expected)
                _simulate_register_load_program(
                    code,
                    x,
                    128,
                    host_tensors={"out": out},
                    lane_order=list(reversed(range(128))),
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_scalar_carry_zero_trip_and_padded_decline(self):
        from test.test_atomic_ops import _simulate_register_load_program

        x = torch.arange(32).reshape(2, 16).float()
        for steps in (0, 1, 3):
            with self.subTest(steps=steps):
                _bound, code = _bounded_gather_codegen(
                    _all_axis_bounded_loop, (x, steps)
                )
                rows = []
                for original in x:
                    value, carry = original, torch.tensor(0.0)
                    for _ in range(steps):
                        carry = (value + carry).amin()
                        value = (value + carry)[(torch.arange(16) * 5 + 3) % 16]
                    rows.append(value)
                out = torch.empty_like(x)
                _simulate_register_load_program(
                    code,
                    x,
                    128,
                    host_tensors={"out": out},
                    scalar_args={"steps": steps},
                    lane_order=list(reversed(range(128))),
                )
                torch.testing.assert_close(out, torch.stack(rows), rtol=0, atol=0)
        with self.assertRaises(exc.InvalidConfig):
            _bounded_gather_codegen(_all_axis_bounded_loop, (torch.ones(2, 17), 2))

    def test_schema_volume_dtype_and_existing_multiaxis_declines(self):
        import sympy

        from helion._compiler.cute.bounded_gather import all_axis_reduction_shape
        from helion._compiler.cute.computed_fragment import _fragment_logical_shape
        from helion._compiler.inductor_lowering import ReductionLowering

        x = torch.arange(32).reshape(2, 16).float()
        bound, _code = _bounded_gather_codegen(
            _all_axis_bounded_gather, (x, "min", "omitted", False)
        )
        with bound.env, bound.host_function:
            node = next(
                n
                for graph in bound.host_function.device_ir.graphs
                for n in graph.graph.nodes
                if n.target is torch.ops.aten.amin.default
                and isinstance(n.meta.get("lowering"), ReductionLowering)
            )

            def infer(n):
                return _fragment_logical_shape(
                    bound.env, n, scalar_indexed_loads=True, tensor_indexed_loads=True
                )

            def dimension(size):
                return bound.env.specialize_expr(sympy.sympify(size))

            def prove():
                return all_axis_reduction_shape(node, infer, dimension)

            self.assertEqual(prove(), ())
            args, kwargs, value = node.args, node.kwargs, node.meta["val"]
            mutations = (
                ((args[0], [0]), {}),
                ((args[0], None), {}),
                ((args[0], [], True), {}),
                ((args[0], [], 1), {}),
                (args, {"self": args[0]}),
                (args, {"unexpected": 1}),
                ((*args, [], False, 7), {}),
            )
            for changed_args, changed_kwargs in mutations:
                with self.subTest(args=changed_args[1:], kwargs=changed_kwargs):
                    try:
                        node.args, node.kwargs = changed_args, changed_kwargs
                        self.assertIsNone(prove())
                    finally:
                        node.args, node.kwargs = args, kwargs
            data = node.meta["lowering"].buffer.data
            for field, changed in (("reduction_ranges", [15]), ("ranges", [2])):
                with self.subTest(field=field):
                    original = getattr(data, field)
                    try:
                        object.__setattr__(data, field, changed)
                        self.assertIsNone(prove())
                    finally:
                        object.__setattr__(data, field, original)
            for changed in (torch.empty((), dtype=torch.float64), torch.empty(1)):
                try:
                    node.meta["val"] = changed
                    self.assertIsNone(prove())
                finally:
                    node.meta["val"] = value
            for logical in ((sympy.Integer(0),), (sympy.Integer(15),), ()):
                with self.subTest(logical=logical):
                    self.assertIsNone(
                        all_axis_reduction_shape(
                            node, lambda _node, shape=logical: shape, dimension
                        )
                    )
            self.assertEqual(prove(), ())
        # The unchanged lowerer rejects truly multiple reduction dimensions.
        for op, mode, keep in (("min", "empty", True), ("max", "none", False)):
            with (
                self.subTest(op=op),
                self.assertRaisesRegex(
                    NotImplementedError, "multiple reduction dimensions"
                ),
            ):
                _bounded_gather_codegen(
                    _all_axis_bounded_gather, (torch.ones(2, 4, 8), op, mode, keep)
                )


@onlyBackends("cute")
class TestCuteAllAxisGatherNative(TestCase):
    def test_reductions_and_scalar_loop(self):
        for op, mode, keep in (
            ("min", "none", False),
            ("max", "empty", True),
            ("sum", "none", True),
            ("prod", "omitted", False),
        ):
            with self.subTest(op=op):
                x = torch.arange(32, device=DEVICE).reshape(2, 16).remainder(3)
                x = x.int() if op in ("sum", "prod") else x.float()
                if op == "prod":
                    x = x.remainder(2) + 1
                before = x.clone()
                _, actual = code_and_output(
                    _all_axis_bounded_gather,
                    (x, op, mode, keep),
                    cute_fragment_bounded_gather=True,
                )
                if op == "min":
                    reduced = x.amin(-1, keepdim=True)
                elif op == "max":
                    reduced = x.amax(-1, keepdim=True)
                elif op == "sum":
                    reduced = x.sum(-1, keepdim=True)
                else:
                    reduced = x.prod(-1, keepdim=True)
                expected = (x + reduced)[
                    :, (torch.arange(16, device=DEVICE) * 5 + 3) % 16
                ]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
        for steps in (0, 3):
            with self.subTest(steps=steps):
                x = torch.arange(32, device=DEVICE).reshape(2, 16).float()
                before = x.clone()
                _, actual = code_and_output(
                    _all_axis_bounded_loop,
                    (x, steps),
                    cute_fragment_bounded_gather=True,
                )
                expected, carry = x.clone(), torch.zeros((2, 1), device=DEVICE)
                for _ in range(steps):
                    carry = (expected + carry).amin(-1, keepdim=True)
                    expected = (expected + carry)[
                        :, (torch.arange(16, device=DEVICE) * 5 + 3) % 16
                    ]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_bounded_exchange(x: torch.Tensor, outer_steps: int, inner_steps: int):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))[:]
        value = x[row, :]
        bias = hl.full([], 1, dtype=x.dtype)
        for _outer in range(outer_steps):
            for _inner in range(inner_steps):
                index = ((lane ^ 1) % x.size(1)).long()
                value = torch.gather(value + bias, -1, index)
                bias = bias + 1
            value = value + x[row, lane]
        out[row, :] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_bounded_packets(x: torch.Tensor, steps: int):
    out = torch.empty((x.size(0), x.size(1), 32), device=x.device, dtype=x.dtype)
    for batch in hl.grid(x.size(0)):
        leading = hl.arange(x.size(1))
        lane = hl.arange(32)
        value = hl.full([x.size(1), 32], 0, dtype=x.dtype)
        for _outer in range(steps):
            for packet in range((x.size(2) + 31) // 32):
                column = packet * 32 + lane
                loaded = hl.load(
                    x, [batch, leading, column], extra_mask=column < x.size(2)
                )
                index = ((leading[:, None] * 0 + (lane[None, :] ^ 7)) % 32).long()
                value = torch.gather(value + loaded, -1, index)
        out[batch, leading, lane] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_bounded_index_drift(x: torch.Tensor, clamp: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1)).long()
        value = x[row, :]
        for _outer in range(2):
            index = index + 1
            for _inner in range(2):
                if clamp:
                    selected = index % x.size(1)
                else:
                    selected = index
                value = torch.gather(value + 1, -1, selected)
        out[row, :] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_bounded_swapped_index(x: torch.Tensor, clamp: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1))[:].long()
        other = hl.full([x.size(1)], x.size(1), dtype=torch.int64)
        value = x[row, :]
        for _outer in range(2):
            previous = index
            index = other
            other = previous
            for _inner in range(2):
                if clamp:
                    selected = index % x.size(1)
                else:
                    selected = index
                value = torch.gather(value + 1, -1, selected)
        out[row, :] = value
    return out


class TestCuteNestedForGatherCPU(TestCase):
    def test_nested_scalar_vector_carries_and_zero_trips(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for width, dtype, outer, inner in (
            (17, torch.float32, 0, 3),
            (17, torch.int64, 3, 0),
            (17, torch.int32, 2, 3),
            (32, torch.float32, 3, 2),
            (64, torch.int64, 2, 3),
        ):
            with self.subTest(width=width, dtype=dtype, outer=outer, inner=inner):
                x = torch.arange(2 * width).reshape(2, width).to(dtype)
                _bound, code = _bounded_gather_codegen(
                    _nested_bounded_exchange, (x, outer, inner)
                )
                expected, bias = x.clone(), 1
                index = (torch.arange(width) ^ 1) % width
                for _ in range(outer):
                    for _ in range(inner):
                        expected = (expected + bias)[:, index]
                        bias += 1
                    expected += x
                for order in (list(range(128)), list(reversed(range(128)))):
                    out = torch.full_like(x, -999)
                    _simulate_register_load_program(
                        code,
                        x,
                        128,
                        host_tensors={"out": out},
                        lane_order=order,
                        scalar_args={"outer_steps": outer, "inner_steps": inner},
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_nested_packet_tails_and_independent_leading_domains(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for shape, steps in (((2, 3, 65), 2), ((1, 5, 33), 1), ((2, 1, 17), 0)):
            with self.subTest(shape=shape, steps=steps):
                x = torch.arange(math.prod(shape)).reshape(shape).int()
                _bound, code = _bounded_gather_codegen(
                    _nested_bounded_packets, (x, steps)
                )
                expected = torch.zeros((*shape[:2], 32), dtype=x.dtype)
                index = torch.arange(32) ^ 7
                for _ in range(steps):
                    for packet in range((shape[2] + 31) // 32):
                        chunk = torch.zeros_like(expected)
                        part = x[:, :, packet * 32 : (packet + 1) * 32]
                        chunk[:, :, : part.size(2)] = part
                        expected = (expected + chunk)[:, :, index]
                out = torch.full_like(expected, -999)
                _simulate_register_load_program(
                    code,
                    x,
                    128,
                    host_tensors={"out": out},
                    scalar_args={"steps": steps},
                    lane_order=list(reversed(range(128))),
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_nested_mutable_range_requires_bound_at_use(self):
        from test.test_atomic_ops import _simulate_register_load_program

        x = torch.arange(64).reshape(2, 32).int()
        with self.assertRaises(exc.InvalidConfig):
            _bounded_gather_codegen(_nested_bounded_index_drift, (x, False))
        _bound, code = _bounded_gather_codegen(_nested_bounded_index_drift, (x, True))
        out = torch.empty_like(x)
        _simulate_register_load_program(code, x, 128, host_tensors={"out": out})
        expected = x.clone()
        for outer in range(2):
            for _ in range(2):
                expected = (expected + 1)[:, (torch.arange(32) + outer + 1) % 32]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_nested_current_edges_all_backedges_and_transaction_rollback(self):
        import operator

        from helion._compiler.cute.gather_domains import loop_domain_facts
        from helion._compiler.device_ir import ForLoopGraphInfo
        from helion._compiler.device_ir import RootGraphInfo
        from helion.language import _tracing_ops
        from helion.language import creation_ops

        bound, _code = _bounded_gather_codegen(
            _indexed_bounded_gather, (torch.ones(2, 64), "plain")
        )
        for bad in (
            "valid",
            "stale_node_args",
            "inner_shape",
            "inner_dtype",
            "outer_shape",
            "outer_dtype",
            "logical_drift",
            "multiple_calls",
            "dynamic_if",
            "dynamic_while",
            "alias_mutation",
            "prefix_mutation",
            "foreign_capture",
            "forward_capture",
            "missing_phi",
            "wrong_arity",
            "recursive",
            "other_control_reference",
            "unknown_initialization",
            "wrapped_iota",
            "wrapped_carry",
            "wrapped_dtype",
            "wrapped_alias_write",
            "shared_graph_object",
        ):
            with self.subTest(case=bad), bound.env:
                root, outer, inner = (torch.fx.Graph() for _ in range(3))

                def meta(node, width=64, dtype=torch.int64):
                    node.meta["val"] = torch.empty(width, dtype=dtype)
                    return node

                entry = meta(
                    root.call_function(
                        torch.ops.aten.empty.memory_format
                        if bad == "unknown_initialization"
                        else creation_ops.full,
                        ([64],)
                        if bad == "unknown_initialization"
                        else ([17 if bad == "logical_drift" else 64], 0),
                    )
                )
                lane = meta(root.call_function(torch.ops.prims.iota.default, (64,)))
                op, ol = (meta(outer.placeholder(name)) for name in ("value", "lane"))
                ip, il = (meta(inner.placeholder(name)) for name in ("value", "lane"))
                result = meta(
                    inner.call_function(
                        creation_ops.full,
                        (
                            [
                                32
                                if bad == "inner_shape"
                                else 18
                                if bad == "logical_drift"
                                else 64
                            ],
                            1,
                        ),
                    ),
                    32 if bad == "inner_shape" else 64,
                    torch.float32 if bad == "inner_dtype" else torch.int64,
                )
                inner.output([result])
                capture_index = ol
                if bad.startswith("wrapped_"):
                    if bad == "wrapped_carry":
                        # Its entry has an exact iota range, but it is an outer
                        # mutable carry, unlike the separate readonly lane.
                        entry.target = torch.ops.prims.iota.default
                        entry.args = (64,)
                    capture_index = meta(
                        outer.call_function(
                            _tracing_ops._new_var,
                            (op if bad == "wrapped_carry" else ol,),
                        ),
                        dtype=torch.float32 if bad == "wrapped_dtype" else torch.int64,
                    )
                if bad == "wrapped_alias_write":
                    alias = meta(
                        outer.call_function(
                            torch.ops.aten.alias.default, (capture_index,)
                        )
                    )
                    outer.call_function(torch.ops.aten.add_.Scalar, (alias, 1))
                child = outer.call_function(
                    _tracing_ops._for_loop, (2, [0], [2], [op, capture_index])
                )
                item = meta(outer.call_function(operator.getitem, (child, 0)))
                phi = meta(outer.call_function(_tracing_ops._phi, (op, item)))
                returned = phi
                if bad in ("outer_shape", "outer_dtype"):
                    returned = meta(
                        outer.call_function(
                            creation_ops.full, ([32 if bad == "outer_shape" else 64], 1)
                        ),
                        32 if bad == "outer_shape" else 64,
                        torch.float32 if bad == "outer_dtype" else torch.int64,
                    )
                if bad == "missing_phi":
                    outer.erase_node(phi)
                    returned = item
                if bad == "multiple_calls":
                    outer.call_function(_tracing_ops._for_loop, (2, [0], [2], [op, ol]))
                if bad in ("dynamic_if", "dynamic_while", "recursive"):
                    target = {
                        "dynamic_if": _tracing_ops._if,
                        "dynamic_while": _tracing_ops._while_loop,
                        "recursive": _tracing_ops._for_loop,
                    }[bad]
                    outer.call_function(target, (1, [0], [2], [op, ol]))
                if bad == "alias_mutation":
                    alias = meta(
                        outer.call_function(torch.ops.aten.alias.default, (ol,))
                    )
                    outer.call_function(torch.ops.aten.add_.Scalar, (alias, 1))
                if bad == "prefix_mutation":
                    alias = meta(
                        root.call_function(torch.ops.aten.alias.default, (lane,))
                    )
                    root.call_function(torch.ops.aten.add_.Scalar, (alias, 1))
                outer.output([returned])
                call = root.call_function(
                    _tracing_ops._for_loop, (1, [0], [2], [entry, lane])
                )
                oi = meta(root.call_function(operator.getitem, (call, 0)))
                po = meta(root.call_function(_tracing_ops._phi, (entry, oi)))
                if bad == "foreign_capture":
                    call.args = (1, [0], [2], [entry, il])
                if bad == "forward_capture":
                    late = meta(root.call_function(torch.ops.prims.iota.default, (64,)))
                    call.args = (1, [0], [2], [entry, late])
                if bad == "other_control_reference":
                    root.call_function(_tracing_ops._if, (True, 2, 2, [], []))
                root.output([])
                graphs = [
                    RootGraphInfo(graph_id=0, graph=root),
                    ForLoopGraphInfo(
                        graph_id=1,
                        graph=outer,
                        node_args=[lane, entry]
                        if bad == "stale_node_args"
                        else [entry, lane],
                        block_ids=[],
                    ),
                    ForLoopGraphInfo(
                        graph_id=2,
                        graph=inner,
                        node_args=[op]
                        if bad == "wrong_arity"
                        else [ol, op]
                        if bad == "stale_node_args"
                        else [op, ol],
                        block_ids=[],
                    ),
                ]
                if bad == "shared_graph_object":
                    graphs.append(
                        ForLoopGraphInfo(
                            graph_id=3,
                            graph=inner,
                            node_args=[op, capture_index],
                            block_ids=[],
                        )
                    )
                facts = loop_domain_facts(bound.env, graphs)
                if bad in ("valid", "stale_node_args", "wrapped_iota", "wrapped_carry"):
                    self.assertIn(ip, facts.shapes)
                    self.assertIn(po, facts.shapes)
                    if bad == "wrapped_carry":
                        self.assertNotIn(il, facts.readonly_ranges)
                    else:
                        self.assertEqual(facts.readonly_ranges[il], (0, 63))
                    self.assertNotIn(op, facts.readonly_ranges)
                    self.assertNotIn(ip, facts.readonly_ranges)
                    old = po.meta["val"]
                    po.meta["val"] = torch.empty(64, dtype=torch.float32)
                    rejected = loop_domain_facts(bound.env, graphs)
                    self.assertEqual(rejected.shapes, {})
                    self.assertEqual(rejected.readonly_ranges, {})
                    po.meta["val"] = old
                    self.assertEqual(loop_domain_facts(bound.env, graphs), facts)
                else:
                    self.assertEqual(facts.shapes, {})
                    self.assertEqual(facts.readonly_ranges, {})

    def test_swapped_index_carry_cannot_reuse_iota_entry_interval(self):
        from test.test_atomic_ops import _simulate_register_load_program

        x = torch.arange(64).reshape(2, 32).int()
        with self.assertRaises(exc.InvalidConfig):
            _bounded_gather_codegen(_nested_bounded_swapped_index, (x, False))
        _bound, code = _bounded_gather_codegen(_nested_bounded_swapped_index, (x, True))
        out = torch.empty_like(x)
        _simulate_register_load_program(
            code,
            x,
            128,
            host_tensors={"out": out},
            lane_order=list(reversed(range(128))),
        )
        expected = x[:, :1].expand_as(x) + 4
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@onlyBackends("cute")
class TestCuteNestedForGatherNative(TestCase):
    def test_nested_carries(self):
        for width, outer, inner in ((17, 0, 3), (17, 3, 0), (17, 2, 3), (64, 2, 3)):
            with self.subTest(width=width, outer=outer, inner=inner):
                x = torch.arange(2 * width, device=DEVICE).reshape(2, width).int()
                expected, bias = x.clone(), 1
                index = (torch.arange(width, device=DEVICE) ^ 1) % width
                for _ in range(outer):
                    for _ in range(inner):
                        expected = (expected + bias)[:, index]
                        bias += 1
                    expected += x
                before = x.clone()
                _, actual = code_and_output(
                    _nested_bounded_exchange,
                    (x, outer, inner),
                    cute_fragment_bounded_gather=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_nested_packet_tails(self):
        for shape, steps in (((2, 3, 65), 2), ((1, 5, 33), 1), ((2, 1, 17), 0)):
            with self.subTest(shape=shape, steps=steps):
                x = torch.arange(math.prod(shape), device=DEVICE).reshape(shape).int()
                expected = torch.zeros((*shape[:2], 32), device=DEVICE, dtype=x.dtype)
                index = torch.arange(32, device=DEVICE) ^ 7
                for _ in range(steps):
                    for packet in range((shape[2] + 31) // 32):
                        chunk = torch.zeros_like(expected)
                        part = x[:, :, packet * 32 : (packet + 1) * 32]
                        chunk[:, :, : part.size(2)] = part
                        expected = (expected + chunk)[:, :, index]
                before = x.clone()
                _, actual = code_and_output(
                    _nested_bounded_packets,
                    (x, steps),
                    cute_fragment_bounded_gather=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_bounded_pure_asm(x: torch.Tensor, outer_steps: int, inner_steps: int):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))[:]
        value = x[row, :]
        one = hl.full([], 1, dtype=torch.int32)
        for _outer in range(outer_steps):
            for _inner in range(inner_steps):
                adjusted = hl.inline_asm_elementwise(
                    "add.s32 $0, $1, $2;",
                    "=r,r,r",
                    [value, one],
                    dtype=torch.int32,
                    is_pure=True,
                    pack=1,
                )
                value = torch.gather(adjusted, -1, ((lane ^ 1) % x.size(1)).long())
        out[row, :] = value
    return out


class TestCuteNestedAsmGatherCPU(TestCase):
    def test_actual_normalized_pure_assembly_and_tails(self):
        from test.test_atomic_ops import _simulate_register_load_program

        for width, outer, inner in ((17, 2, 3), (32, 2, 2), (65, 0, 2), (17, 2, 0)):
            with self.subTest(width=width, outer=outer, inner=inner):
                x = torch.arange(2 * width).reshape(2, width).int()
                _bound, code = _bounded_gather_codegen(
                    _nested_bounded_pure_asm, (x, outer, inner)
                )
                expected = x.clone()
                index = (torch.arange(width) ^ 1) % width
                for _ in range(outer * inner):
                    expected = (expected + 1)[:, index]
                for order in (list(range(128)), list(reversed(range(128)))):
                    out = torch.empty_like(x)
                    _simulate_register_load_program(
                        _asm_gather_model_source(code),
                        x,
                        128,
                        host_tensors={"out": out},
                        lane_order=order,
                        scalar_args={"outer_steps": outer, "inner_steps": inner},
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_normalized_abi_effect_and_ambiguity_declines(self):
        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        from helion._compiler.cute.gather_domains import loop_domain_facts
        from helion.language import inline_asm_ops

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(_nested_bounded_pure_asm, (torch.ones(2, 32).int(), 2, 2))
            graphs = bound.host_function.device_ir.graphs
            node = next(
                node
                for graph in graphs
                for node in graph.graph.nodes
                if node.target is inline_asm_ops.inline_asm_elementwise
            )
            original_args, original_kwargs = node.args, node.kwargs
            original_value = node.meta["val"]
            self.assertEqual(len(original_args), 6)
            self.assertEqual(original_kwargs, {})
            self.assertIs(original_args[4], True)
            with bound.env, bound.host_function:
                valid = loop_domain_facts(bound.env, graphs)
                self.assertTrue(valid.shapes)
                for case in (
                    "impure",
                    "ambiguous_keyword",
                    "keyword_only_contract",
                    "arity",
                    "pack2",
                    "pack_bool",
                    "tuple_dtype",
                    "tuple_output",
                    "wrong_dtype",
                    "empty_operands",
                    "non_tensor_operand",
                ):
                    with self.subTest(case=case):
                        args = list(original_args)
                        node.kwargs = {}
                        if case == "impure":
                            args[4] = False
                        elif case == "ambiguous_keyword":
                            node.kwargs = {"is_pure": False}
                        elif case == "keyword_only_contract":
                            args = args[:3]
                            node.kwargs = {
                                "dtype": torch.int32,
                                "is_pure": True,
                                "pack": 1,
                            }
                        elif case == "arity":
                            args = args[:-1]
                        elif case == "pack2":
                            args[5] = 2
                        elif case == "pack_bool":
                            args[5] = True
                        elif case == "tuple_dtype":
                            args[3] = (torch.int32, torch.int32)
                        elif case == "tuple_output":
                            node.meta["val"] = (original_value, original_value)
                        elif case == "wrong_dtype":
                            args[3] = torch.float32
                        elif case == "empty_operands":
                            args[2] = []
                        elif case == "non_tensor_operand":
                            args[2] = [args[2][0], 1]
                        node.args = tuple(args)
                        rejected = loop_domain_facts(bound.env, graphs)
                        self.assertEqual(rejected.shapes, {})
                        self.assertEqual(rejected.readonly_ranges, {})
                        node.args, node.kwargs = original_args, original_kwargs
                        node.meta["val"] = original_value
                        self.assertEqual(loop_domain_facts(bound.env, graphs), valid)


@onlyBackends("cute")
class TestCuteNestedAsmGatherNative(TestCase):
    def test_nested_pure_assembly(self):
        for width, steps in ((17, 2), (32, 3), (65, 0)):
            with self.subTest(width=width, steps=steps):
                x = torch.arange(2 * width, device=DEVICE).reshape(2, width).int()
                before = x.clone()
                _, actual = code_and_output(
                    _nested_bounded_pure_asm,
                    (x, steps, 2),
                    cute_fragment_bounded_gather=True,
                )
                expected = x.clone()
                index = (torch.arange(width, device=DEVICE) ^ 1) % width
                for _ in range(steps * 2):
                    expected = (expected + 1)[:, index]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
