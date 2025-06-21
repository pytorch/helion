from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


def grid_2d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        bi,
        bj,
        m,
        n,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for i in range(bi):
        for j in range(bj):
            out[i, j] = torch.mm(x[i, j], y)
    return out


class TestGrid(TestCase):
    def test_grid_1d(self):
        @helion.kernel(static_shapes=True)
        def grid_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for i in hl.grid(b):
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[i, tile_m, tile_k], y[tile_k, tile_n])
                    out[i, tile_m, tile_n] = acc
            return out

        def grid_1d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for i in range(b):
                out[i] = torch.mm(x[i], y)
            return out

        args = (
            torch.randn([8, 16, 32], device=DEVICE, dtype=torch.float16),
            torch.randn([32, 4], device=DEVICE, dtype=torch.float16),
        )
        code, result = code_and_output(grid_1d, args)
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _grid_1d_kernel(x, y, out, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    for offset_1 in range(0, 16, _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        for offset_2 in range(0, 4, _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            mask_2 = indices_2 < 4
            acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_2], 0.0, tl.float32)
            for offset_3 in range(0, 32, _BLOCK_SIZE_3):
                indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
                acc_copy = acc
                acc_copy_0 = acc_copy
                load = tl.load(x + (tl.full([1], offset_0, tl.int32)[:, None] * 512 + indices_1[:, None] * 32 + indices_3[None, :] * 1), None)
                load_1 = tl.load(y + (indices_3[:, None] * 4 + indices_2[None, :] * 1), mask_2[None, :], other=0)
                acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
            v_0 = acc.to(tl.float16)
            tl.store(out + (tl.full([1], offset_0, tl.int32)[:, None] * 64 + indices_1[:, None] * 4 + indices_2[None, :] * 1), v_0, mask_2[None, :])

def grid_1d(x: torch.Tensor, y: torch.Tensor):
    b, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_3 = 16
    _grid_1d_kernel[8,](x, y, out, _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out

def _grid_1d_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    b, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_3 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_grid_1d_kernel)(x, y, out, _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
        )

        # test again with block_ptr indexing
        code, result = code_and_output(
            grid_1d, args, block_sizes=[16, 16, 16], indexing="block_ptr"
        )
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _grid_1d_kernel(x, y, out, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    for offset_1 in range(0, 16, _BLOCK_SIZE_1):
        for offset_2 in range(0, 4, _BLOCK_SIZE_2):
            acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_2], 0.0, tl.float32)
            for offset_3 in range(0, 32, _BLOCK_SIZE_3):
                acc_copy = acc
                acc_copy_0 = acc_copy
                load = tl.reshape(tl.load(tl.make_block_ptr(x, [8, 16, 32], [512, 32, 1], [offset_0, offset_1, offset_3], [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3], [2, 1, 0]), boundary_check=[1, 2], padding_option='zero'), [_BLOCK_SIZE_1, _BLOCK_SIZE_3])
                load_1 = tl.load(tl.make_block_ptr(y, [32, 4], [4, 1], [offset_3, offset_2], [_BLOCK_SIZE_3, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
                acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
            v_0 = acc.to(tl.float16)
            tl.store(tl.make_block_ptr(out, [8, 16, 4], [64, 4, 1], [offset_0, offset_1, offset_2], [1, _BLOCK_SIZE_1, _BLOCK_SIZE_2], [2, 1, 0]), tl.reshape(v_0, [1, _BLOCK_SIZE_1, _BLOCK_SIZE_2]), boundary_check=[1, 2])

def grid_1d(x: torch.Tensor, y: torch.Tensor):
    b, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_3 = 16
    _grid_1d_kernel[8,](x, y, out, _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out

def _grid_1d_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    b, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_3 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_grid_1d_kernel)(x, y, out, _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
        )

    def test_grid_2d_idx_list(self):
        @helion.kernel(static_shapes=True)
        def grid_2d_idx_list(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bi, bj, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                bi,
                bj,
                m,
                n,
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for i, j in hl.grid([bi, bj]):
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(
                            acc, x[i, j, tile_m, tile_k], y[tile_k, tile_n]
                        )
                    out[i, j, tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([3, 4, 64, 32], device=DEVICE, dtype=torch.float16),
            torch.randn([32, 16], device=DEVICE, dtype=torch.float16),
        )

        code, result = code_and_output(grid_2d_idx_list, args)
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _grid_2d_idx_list_kernel(x, y, out, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_4: tl.constexpr):
    num_blocks_0 = 3
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1
    for offset_2 in range(0, 64, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        for offset_3 in range(0, 16, _BLOCK_SIZE_3):
            indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
            acc = tl.full([_BLOCK_SIZE_2, _BLOCK_SIZE_3], 0.0, tl.float32)
            for offset_4 in range(0, 32, _BLOCK_SIZE_4):
                indices_4 = offset_4 + tl.arange(0, _BLOCK_SIZE_4).to(tl.int32)
                acc_copy = acc
                acc_copy_0 = acc_copy
                load = tl.load(x + (tl.full([1], offset_0, tl.int32)[:, None] * 8192 + tl.full([1], offset_1, tl.int32)[:, None] * 2048 + indices_2[:, None] * 32 + indices_4[None, :] * 1), None)
                load_1 = tl.load(y + (indices_4[:, None] * 16 + indices_3[None, :] * 1), None)
                acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
            v_0 = acc.to(tl.float16)
            tl.store(out + (tl.full([1], offset_0, tl.int32)[:, None] * 4096 + tl.full([1], offset_1, tl.int32)[:, None] * 1024 + indices_2[:, None] * 16 + indices_3[None, :] * 1), v_0, None)

def grid_2d_idx_list(x: torch.Tensor, y: torch.Tensor):
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(bi, bj, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_4 = 16
    _grid_2d_idx_list_kernel[3 * 4,](x, y, out, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_4, num_warps=4, num_stages=3)
    return out

def _grid_2d_idx_list_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(bi, bj, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_4 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_grid_2d_idx_list_kernel)(x, y, out, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_4, num_warps=4, num_stages=3)""",
        )

        code, result = code_and_output(
            grid_2d_idx_list, args, block_sizes=[64, 32, 16], indexing="block_ptr"
        )
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _grid_2d_idx_list_kernel(x, y, out, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_4: tl.constexpr):
    num_blocks_0 = 3
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1
    for offset_2 in range(0, 64, _BLOCK_SIZE_2):
        for offset_3 in range(0, 16, _BLOCK_SIZE_3):
            acc = tl.full([_BLOCK_SIZE_2, _BLOCK_SIZE_3], 0.0, tl.float32)
            for offset_4 in range(0, 32, _BLOCK_SIZE_4):
                acc_copy = acc
                acc_copy_0 = acc_copy
                load = tl.reshape(tl.load(tl.make_block_ptr(x, [3, 4, 64, 32], [8192, 2048, 32, 1], [offset_0, offset_1, offset_2, offset_4], [1, 1, _BLOCK_SIZE_2, _BLOCK_SIZE_4], [3, 2, 1, 0]), boundary_check=[2, 3], padding_option='zero'), [_BLOCK_SIZE_2, _BLOCK_SIZE_4])
                load_1 = tl.load(tl.make_block_ptr(y, [32, 16], [16, 1], [offset_4, offset_3], [_BLOCK_SIZE_4, _BLOCK_SIZE_3], [1, 0]), boundary_check=[0, 1], padding_option='zero')
                acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
            v_0 = acc.to(tl.float16)
            tl.store(tl.make_block_ptr(out, [3, 4, 64, 16], [4096, 1024, 16, 1], [offset_0, offset_1, offset_2, offset_3], [1, 1, _BLOCK_SIZE_2, _BLOCK_SIZE_3], [3, 2, 1, 0]), tl.reshape(v_0, [1, 1, _BLOCK_SIZE_2, _BLOCK_SIZE_3]), boundary_check=[2, 3])

def grid_2d_idx_list(x: torch.Tensor, y: torch.Tensor):
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(bi, bj, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_3 = 32
    _BLOCK_SIZE_2 = 64
    _BLOCK_SIZE_4 = 16
    _grid_2d_idx_list_kernel[3 * 4,](x, y, out, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_4, num_warps=4, num_stages=3)
    return out

def _grid_2d_idx_list_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(bi, bj, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_3 = 32
    _BLOCK_SIZE_2 = 64
    _BLOCK_SIZE_4 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_grid_2d_idx_list_kernel)(x, y, out, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_4, num_warps=4, num_stages=3)""",
        )

    def test_grid_2d_idx_nested(self):
        @helion.kernel(static_shapes=True)
        def grid_2d_idx_nested(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bi, bj, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                bi,
                bj,
                m,
                n,
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for i in hl.grid(bi):
                for j in hl.grid(bj):
                    for tile_m, tile_n in hl.tile([m, n]):
                        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                        for tile_k in hl.tile(k):
                            acc = torch.addmm(
                                acc, x[i, j, tile_m, tile_k], y[tile_k, tile_n]
                            )
                        out[i, j, tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([3, 4, 64, 32], device=DEVICE, dtype=torch.float16),
            torch.randn([32, 16], device=DEVICE, dtype=torch.float16),
        )
        code, result = code_and_output(grid_2d_idx_nested, args)
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _grid_2d_idx_nested_kernel(x, y, out, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_4: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    for offset_1 in range(0, 4, 1):
        for offset_2 in range(0, 64, _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            for offset_3 in range(0, 16, _BLOCK_SIZE_3):
                indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
                acc = tl.full([_BLOCK_SIZE_2, _BLOCK_SIZE_3], 0.0, tl.float32)
                for offset_4 in range(0, 32, _BLOCK_SIZE_4):
                    indices_4 = offset_4 + tl.arange(0, _BLOCK_SIZE_4).to(tl.int32)
                    acc_copy = acc
                    acc_copy_0 = acc_copy
                    load = tl.load(x + (tl.full([1], offset_0, tl.int32)[:, None] * 8192 + tl.full([1], offset_1, tl.int32)[:, None] * 2048 + indices_2[:, None] * 32 + indices_4[None, :] * 1), None)
                    load_1 = tl.load(y + (indices_4[:, None] * 16 + indices_3[None, :] * 1), None)
                    acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
                v_0 = acc.to(tl.float16)
                tl.store(out + (tl.full([1], offset_0, tl.int32)[:, None] * 4096 + tl.full([1], offset_1, tl.int32)[:, None] * 1024 + indices_2[:, None] * 16 + indices_3[None, :] * 1), v_0, None)

def grid_2d_idx_nested(x: torch.Tensor, y: torch.Tensor):
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(bi, bj, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_4 = 16
    _grid_2d_idx_nested_kernel[3,](x, y, out, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_4, num_warps=4, num_stages=3)
    return out

def _grid_2d_idx_nested_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty(bi, bj, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_4 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_grid_2d_idx_nested_kernel)(x, y, out, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_4, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
