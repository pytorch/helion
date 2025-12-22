from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfCpu
from helion._testing import skipIfLowVRAM
from helion._testing import skipIfNormalMode
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
import helion.language as hl


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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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

        x = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
        y = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
        weight = torch.randn((n,), device=DEVICE, dtype=torch.float16)
        grad_out = torch.randn((m, n), device=DEVICE, dtype=torch.float16)

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

        torch.testing.assert_close(grad_x, ref_grad_x, rtol=1e-3, atol=2e-3)
        torch.testing.assert_close(grad_y, ref_grad_y, rtol=1e-3, atol=2e-3)
        # TODO(oulgen): needs mindot size mocked
        # self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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

        dtype = torch.float16
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
                total_m, n, p, device=group_offsets.device, dtype=torch.float16
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

        dtype = torch.float16
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
        self.assertExpectedJournal(code)

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
    @skipIfLowVRAM("Test requires high VRAM")
    @skipIfCpu("fails on Triton CPU backend")
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
            range_flattens=[None],
            range_multi_buffers=[None],
            range_num_stages=[],
            range_unroll_factors=[0],
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
                torch.accelerator.synchronize()
                return

            code, out = code_and_output(kernel, (x, y))
            torch.accelerator.synchronize()
            checker = self.assertIn if expect_int64_in_code else self.assertNotIn
            checker("tl.int64", code)
            torch.accelerator.synchronize()
            ref_out = torch.add(x, y)
            torch.accelerator.synchronize()
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)

        small_shape = (128, 128)
        large_shape = (51200, 51200)

        if DEVICE.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info()
            element_size = 2  # torch.bfloat16 element size in bytes
            # Worst case: inputs, kernel output, reference output, and temporary buffers.
            # Give ourselves margin by budgeting for 5 tensors of this shape.
            required_bytes = 5 * math.prod(large_shape) * element_size
            if free_bytes < required_bytes:
                required_gib = required_bytes / (1024**3)
                available_gib = free_bytes / (1024**3)
                self.skipTest(
                    f"Large BF16 add needs ~{required_gib:.1f} GiB free, only {available_gib:.1f} GiB available"
                )

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
        self.assertIn("tl.program_id(0).to(tl.int64)", code_int64)

        # Test int32 case: program_id should NOT be cast
        code_int32, result_int32 = code_and_output(add_kernel_int32, (x, y))
        self.assertNotIn(".to(tl.int64)", code_int32)
        self.assertIn("tl.program_id(0)", code_int32)

        # Both should produce correct results
        expected = x + y
        torch.testing.assert_close(result_int64, expected)
        torch.testing.assert_close(result_int32, expected)

    @skipIfRefEager("Test checks for no IMA")
    @skipIfRocm("Test takes too long on ROCm")
    @skipIfCpu("Test requires GPU")
    @skipIfLowVRAM("Test requires large memory")
    def test_large_tensor(self):
        @helion.kernel(autotune_effort="none")
        def f(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty(x.shape)
            for (b,) in hl.grid([x.shape[0]]):
                for (x_tile,) in hl.tile([x.shape[1]]):
                    out[b, x_tile] = x[b, x_tile]
            return out

        B = 2**15
        D = 2**17
        inp = torch.randn(B, D, device=DEVICE, dtype=torch.float16)
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
        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
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
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
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
        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
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
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() != "tl._experimental_make_tensor_descriptor",
        "Not using experimental tensor descriptor",
    )
    def test_reduction_tensor_descriptor_indexing_reduction_loop(self):
        x = torch.randn([64, 256], dtype=torch.float16, device=DEVICE)

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
        self.assertExpectedJournal(code)

    @skipIfCpu("")
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

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([1, N], device=DEVICE)
        expected_dst = torch.ones([1, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

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

    @skipIfRocm("failure on rocm")
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

    @skipIfNormalMode("InternalError: Negative indexes")
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

    @skipIfNormalMode(
        "RankMismatch: Cannot assign a tensor of rank 2 to a buffer of rank 3"
    )
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

    @skipIfNormalMode(
        "RankMismatch: Expected ndim=2, but got ndim=1 - tensor value assignment shape mismatch"
    )
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

    @skipIfNormalMode(
        "InternalError: AssertionError in type_propagation.py - slice indexing error"
    )
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
        self.assertExpectedJournal(code)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
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
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
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
        self.assertExpectedJournal(code)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
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
        self.assertExpectedJournal(code)

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
        a = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        b = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        c = torch.randn([m, n], device=DEVICE, dtype=torch.float16)

        # 3 loads + 1 store = 4 operations
        code, result = code_and_output(
            multi_load_kernel,
            (a, b, c),
            indexing=["pointer", "pointer", "block_ptr", "pointer"],
            block_size=[16, 16],
        )
        expected = a + b + c
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertIn("tl.load", code)
        self.assertIn("tl.make_block_ptr", code)
        self.assertExpectedJournal(code)

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
        a = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        expected = a + a + a

        # When indexing is not specified (empty list), all loads and stores default to pointer
        code1, result = code_and_output(
            many_loads_kernel,
            (a,),
            block_size=[16, 16],
        )
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertExpectedJournal(code1)

        # Single string: backward compatible mode, all loads and stores use the same strategy
        code2, result = code_and_output(
            many_loads_kernel,
            (a,),
            indexing="pointer",
            block_size=[16, 16],
        )
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertExpectedJournal(code2)

        # List: per-operation mode, must provide strategy for all loads and stores (3 loads + 1 store)
        code3, result = code_and_output(
            many_loads_kernel,
            (a,),
            indexing=["pointer", "pointer", "pointer", "pointer"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertExpectedJournal(code3)

        self.assertEqual(code1, code2)
        self.assertEqual(code2, code3)

    @skipIfRefEager("needs debugging")
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
        a = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        b = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
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
        # Verify we have both pointer loads and block_ptr store
        self.assertIn("tl.load", code1)
        self.assertIn("tl.make_block_ptr", code1)
        # Count occurrences: should have block_ptr for store
        self.assertEqual(code1.count("tl.make_block_ptr"), 1)
        self.assertExpectedJournal(code1)

        # Test 2: Different mix - block_ptr loads, pointer store
        code2, result2 = code_and_output(
            load_store_kernel,
            (a, b),
            indexing=["block_ptr", "block_ptr", "pointer"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result2, expected, rtol=1e-3, atol=1e-3)
        # Should have 2 block_ptrs for loads, regular store
        self.assertEqual(code2.count("tl.make_block_ptr"), 2)
        self.assertExpectedJournal(code2)

        # Test 3: All block_ptr
        code3, result3 = code_and_output(
            load_store_kernel,
            (a, b),
            indexing=["block_ptr", "block_ptr", "block_ptr"],
            block_size=[16, 16],
        )
        torch.testing.assert_close(result3, expected, rtol=1e-3, atol=1e-3)
        # Should have 3 block_ptrs total (2 loads + 1 store)
        self.assertEqual(code3.count("tl.make_block_ptr"), 3)
        self.assertExpectedJournal(code3)

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
        self.assertExpectedJournal(code4)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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

        expected = torch.zeros((M, N, P, Q), device=DEVICE, dtype=torch.float32)
        for i in range(M):
            for j in range(N):
                for p in range(P):
                    for q in range(Q):
                        for k in range(K):
                            expected[i, j, p, q] += val[i, j, k] * B[col[i, j, k], p, q]

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

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

        expected = torch.zeros((M, N, P, Q), device=DEVICE, dtype=torch.float32)
        for i in range(M):
            for j in range(N):
                for p in range(P):
                    for q in range(Q):
                        for k in range(K):
                            expected[i, j, p, q] += val[i, j, k] * B[col[i, j, k], p, q]

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

    @skipIfCpu("")
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
        self.assertExpectedJournal(code)

    @skipIfCpu("fails on Triton CPU backend")
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
        self.assertExpectedJournal(code)

    @skipIfCpu("fails on Triton CPU backend")
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
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
