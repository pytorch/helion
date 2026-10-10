from __future__ import annotations

import unittest

import torch

import helion
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfXPU
from helion._testing import skipUnlessTensorDescriptor
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl
from helion.runtime.settings import _get_backend


def _supports_epilogue_subtile_autotune() -> bool:
    return supports_tensor_descriptor() and torch.cuda.get_device_capability() >= (
        10,
        0,
    )


def _assert_split_codegen(test_case: TestCase, code: str, store_count: int) -> None:
    if _get_backend() == "cute":
        test_case.assertIn("split_smem", code)
        test_case.assertEqual(code.count(".store("), store_count)
        return

    test_case.assertIn("tl.split", code)
    test_case.assertEqual(code.count("tl.store("), store_count)
    test_case.assertNotIn("tl.join", code)


def _assert_no_split_codegen(test_case: TestCase, code: str) -> None:
    if _get_backend() == "cute":
        test_case.assertNotIn("split_smem", code)
        return

    test_case.assertNotIn("tl.split", code)


def _assert_descriptor_store_codegen(
    test_case: TestCase,
    code: str,
    descriptor_store_count: int,
    *,
    pointer_store_count: int | None = None,
) -> None:
    test_case.assertIn("tl.split", code)
    test_case.assertEqual(code.count("_desc.store("), descriptor_store_count)
    if pointer_store_count is None:
        test_case.assertNotIn("tl.store(", code)
    else:
        test_case.assertEqual(code.count("tl.store("), pointer_store_count)


def _assert_descriptor_atomic_codegen(
    test_case: TestCase,
    code: str,
    descriptor_atomic_count: int,
) -> None:
    test_case.assertIn("tl.split", code)
    test_case.assertEqual(code.count("_desc.atomic_add("), descriptor_atomic_count)
    test_case.assertNotIn("tl.atomic_add(", code)


@helion.kernel
def matmul_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
def matmul_with_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc + bias[tile_n]
    return out


@helion.kernel
def matmul_with_cast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out


@helion.kernel
def matmul_atomic_add(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc + bias[tile_n])
    return out


@helion.kernel
def row_slice_atomic_add(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    for tile_m in hl.tile(m):
        hl.atomic_add(out, [tile_m, slice(None)], x[tile_m, :] * 2.0)
    return out


@helion.kernel
def two_row_slice_stores(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = x.size()
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    for tile_m in hl.tile(m):
        base = x[tile_m, :]
        out0[tile_m, :] = base * 2.0
        out1[tile_m, :] = base * 3.0 + 1.0
    return out0, out1


@helion.kernel
def mixed_row_slice_stores(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = x.size()
    out_descriptor = torch.empty_like(x)
    out_pointer = torch.empty_like(x)
    for tile_m in hl.tile(m):
        base = x[tile_m, :]
        out_descriptor[tile_m, :] = base * 2.0
        out_pointer[tile_m, :] = base * 3.0 + 1.0
    return out_descriptor, out_pointer


@helion.kernel
def atomic_add_prev_used(
    y: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prev = torch.empty_like(out)
    for tile_m, tile_n in hl.tile(out.size()):
        old = hl.atomic_add(out, [tile_m, tile_n], y[tile_m, tile_n] * 2.0)
        prev[tile_m, tile_n] = old
    return prev, out


@helion.kernel
def matmul_col_scale_first(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = scale[tile_n, :].T * acc
    return out


@helion.kernel
def matmul_bias_first(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = bias[tile_n] + acc
    return out


@helion.kernel
def matmul_transposed_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [n, m], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_n, tile_m] = acc.T
    return out


@helion.kernel
def matmul_shifted_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros(
        [m, n + 8], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n.index + 8] = acc * 2.0
    return out


@helion.kernel
def matmul_int_prefix_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros(
        [2, m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[1, tile_m, tile_n] = acc * 2.0
    return out


@helion.kernel
def matmul_strided_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros(
        [m, 2 * n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n.index * 2] = acc * 2.0
    return out


@helion.kernel
def matmul_atomic_add_shifted(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n.index + 8], acc * 2.0)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def store_with_tile_indexer(
    tensor_idx: torch.Tensor, data: torch.Tensor, k: int
) -> torch.Tensor:
    m, _ = data.size()
    k = hl.specialize(k)
    out = torch.zeros([m, m, k], device=data.device, dtype=data.dtype)
    for tile_m in hl.tile(m, block_size=4):
        val = hl.load(data, [tile_m, hl.arange(k, dtype=torch.int32)])
        val_3d = val[:, None, :].expand(val.size(0), val.size(0), k)
        hl.store(
            out,
            [tensor_idx[tile_m], tile_m.index, hl.arange(k, dtype=torch.int32)],
            val_3d,
        )
    return out


@onlyBackends(["triton", "cute"])
class TestEpilogueSubtiling(TestCase):
    @skipIfRefEager("test checks generated backend code")
    def test_codegen_s2(self):
        """S=2 produces backend split code and 2 stores for bias/cast epilogues."""
        bias_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias, bias_args, block_sizes=[64, 64, 64], epilogue_subtile=2
        )
        torch.testing.assert_close(
            output, bias_args[0] @ bias_args[1] + bias_args[2], atol=1e-1, rtol=1e-2
        )
        _assert_split_codegen(self, code, 2)

        cast_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_cast, cast_args, block_sizes=[64, 64, 64], epilogue_subtile=2
        )
        torch.testing.assert_close(
            output,
            (cast_args[0] @ cast_args[1]).to(torch.float16),
            atol=1e-1,
            rtol=1e-2,
        )
        _assert_split_codegen(self, code, 2)

    @skipIfRefEager("test checks generated backend code")
    def test_codegen_s4(self):
        """S=4 produces multiple backend split steps and 4 stores."""
        bias_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias, bias_args, block_sizes=[64, 64, 64], epilogue_subtile=4
        )
        torch.testing.assert_close(
            output, bias_args[0] @ bias_args[1] + bias_args[2], atol=1e-1, rtol=1e-2
        )
        _assert_split_codegen(self, code, 4)

        cast_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_cast, cast_args, block_sizes=[64, 64, 64], epilogue_subtile=4
        )
        torch.testing.assert_close(
            output,
            (cast_args[0] @ cast_args[1]).to(torch.float16),
            atol=1e-1,
            rtol=1e-2,
        )
        _assert_split_codegen(self, code, 4)

    def test_disabled_by_default(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(matmul_simple, args, block_sizes=[64, 64, 64])
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        _assert_no_split_codegen(self, code)

    @skipIfRefEager("test checks generated backend code")
    def test_bool_true_normalizes_to_s2(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=True
        )
        torch.testing.assert_close(
            output, args[0] @ args[1] + args[2], atol=1e-1, rtol=1e-2
        )
        _assert_split_codegen(self, code, 2)

    def test_numerical_correctness_s2_vs_s4(self):
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256], device=DEVICE, dtype=torch.float32),
        )
        _, out_none = code_and_output(matmul_with_bias, args, block_sizes=[64, 64, 64])
        _, out_s2 = code_and_output(
            matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=2
        )
        _, out_s4 = code_and_output(
            matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=4
        )
        # The subtile variants must agree bit-tightly with each other: subtiling
        # only splits the store. The no-subtile config may take a different MMA
        # lowering (on cute, fp32 without epilogue_subtile runs tcgen05 as tf32
        # while subtile configs keep the exact SIMT lowering), so it is compared
        # at tf32-friendly tolerance instead.
        torch.testing.assert_close(out_s2, out_s4, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_none, out_s2, atol=1e-1, rtol=1e-2)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_descriptor_atomic_add_codegen_s2(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
            torch.zeros([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_atomic_add,
            args,
            block_sizes=[64, 64, 64],
            epilogue_subtile=2,
            atomic_indexing="tensor_descriptor",
        )
        torch.testing.assert_close(
            output, args[0] @ args[1] + args[2], atol=1e-1, rtol=1e-2
        )
        _assert_descriptor_atomic_codegen(self, code, 2)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_descriptor_atomic_add_static_slice_codegen_s4(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.zeros([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            row_slice_atomic_add,
            args,
            block_sizes=[64],
            epilogue_subtile=4,
            atomic_indexing="tensor_descriptor",
        )
        torch.testing.assert_close(output, args[0] * 2.0)
        _assert_descriptor_atomic_codegen(self, code, 4)
        self.assertIn("_desc.atomic_add([offset_0, 32]", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_descriptor_store_tiled_dims_codegen_s2(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias,
            args,
            block_sizes=[64, 64, 64],
            epilogue_subtile=2,
            indexing=["pointer", "pointer", "pointer", "tensor_descriptor"],
        )
        torch.testing.assert_close(
            output, args[0] @ args[1] + args[2], atol=1e-1, rtol=1e-2
        )
        _assert_descriptor_store_codegen(self, code, 2)
        self.assertIn("_desc.store([offset_0, offset_1 + 0]", code)
        self.assertIn("_desc.store([offset_0, offset_1 + _BLOCK_SIZE_1 // 2]", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_descriptor_store_static_slice_codegen_s2(self):
        args = (torch.randn([128, 128], device=DEVICE, dtype=torch.float32),)
        code, output = code_and_output(
            two_row_slice_stores,
            args,
            block_sizes=[64],
            epilogue_subtile=2,
            indexing=["pointer", "tensor_descriptor", "tensor_descriptor"],
        )
        torch.testing.assert_close(output[0], args[0] * 2.0)
        torch.testing.assert_close(output[1], args[0] * 3.0 + 1.0)
        _assert_descriptor_store_codegen(self, code, 4)
        self.assertIn("_desc.store([offset_0, 0]", code)
        self.assertIn("_desc.store([offset_0, 64]", code)
        self.assertNotIn("_desc.store([offset_0 + _BLOCK_SIZE_0 // 2, 0]", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_mixed_descriptor_pointer_outputs_choose_split_per_output(self):
        args = (torch.randn([128, 128], device=DEVICE, dtype=torch.float32),)
        code, output = code_and_output(
            mixed_row_slice_stores,
            args,
            block_sizes=[64],
            epilogue_subtile=2,
            indexing=["pointer", "tensor_descriptor", "pointer"],
        )
        torch.testing.assert_close(output[0], args[0] * 2.0)
        torch.testing.assert_close(output[1], args[0] * 3.0 + 1.0)
        _assert_descriptor_store_codegen(
            self, code, descriptor_store_count=2, pointer_store_count=2
        )
        self.assertIn("_desc.store([offset_0, 0]", code)
        self.assertIn("_desc.store([offset_0, 64]", code)
        self.assertIn("tl.store(out_pointer + ((offset_0 +", code)
        self.assertNotIn("_desc.store([offset_0 + _BLOCK_SIZE_0 // 2, 0]", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_atomic_add_with_return_value_is_not_epilogue_subtiled(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.zeros([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            atomic_add_prev_used,
            args,
            block_sizes=[64, 64],
            epilogue_subtile=2,
        )
        torch.testing.assert_close(output[0], torch.zeros_like(args[1]))
        torch.testing.assert_close(output[1], args[0] * 2.0)
        self.assertNotIn("tl.split", code)
        self.assertEqual(code.count("tl.atomic_add("), 1)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_split_store_dim_with_transposed_scale_first(self):
        """``scale[tile_n, :].T * acc`` splits the store along N, not M."""
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
            torch.rand([192, 1], device=DEVICE, dtype=torch.float32) + 0.5,
        )
        code, output = code_and_output(
            matmul_col_scale_first,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
        )
        torch.testing.assert_close(
            output, args[2].T * (args[0] @ args[1]), atol=1e-1, rtol=1e-2
        )
        _assert_split_codegen(self, code, 2)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_descriptor_store_with_transposed_scale_first(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
            torch.rand([192, 1], device=DEVICE, dtype=torch.float32) + 0.5,
        )
        code, output = code_and_output(
            matmul_col_scale_first,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
            indexing=["pointer", "pointer", "pointer", "tensor_descriptor"],
        )
        torch.testing.assert_close(
            output, args[2].T * (args[0] @ args[1]), atol=1e-1, rtol=1e-2
        )
        _assert_descriptor_store_codegen(self, code, 2)
        self.assertIn("_desc.store([offset_0, offset_1 + 0]", code)
        self.assertIn("_desc.store([offset_0, offset_1 + _BLOCK_SIZE_1 // 2]", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_split_store_dim_with_bias_first_s4(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
            torch.randn([192], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_bias_first,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=4,
        )
        torch.testing.assert_close(
            output, args[2] + args[0] @ args[1], atol=1e-1, rtol=1e-2
        )
        _assert_split_codegen(self, code, 4)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_descriptor_store_transposed_output(self):
        """out[tile_n, tile_m] = acc.T splits the N tile, now in store dim 0."""
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_transposed_out,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
            indexing=["pointer", "pointer", "tensor_descriptor"],
        )
        torch.testing.assert_close(output, (args[0] @ args[1]).T, atol=1e-1, rtol=1e-2)
        _assert_descriptor_store_codegen(self, code, 2)
        self.assertIn("_desc.store([offset_1 + 0, offset_0]", code)
        self.assertIn("_desc.store([offset_1 + _BLOCK_SIZE_1 // 2, offset_0]", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_split_store_keeps_subscript_offset(self):
        """The piece offset is added to an existing ``tile.index + 8`` offset."""
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_shifted_out,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
        )
        expected = torch.zeros_like(output)
        expected[:, 8:] = 2.0 * (args[0] @ args[1])
        torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-2)
        _assert_split_codegen(self, code, 2)
        self.assertIn("(offset_1 + 8 + tl.arange(", code)
        self.assertIn("(offset_1 + 8 + _BLOCK_SIZE_1 // 2 + tl.arange(", code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_split_store_with_int_prefix(self):
        """out[1, tile_m, tile_n]: the split tile sits at store dim 2, value dim 1."""
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_int_prefix_out,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
        )
        expected = torch.zeros_like(output)
        expected[1] = 2.0 * (args[0] @ args[1])
        torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-2)
        _assert_split_codegen(self, code, 2)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_atomic_add_split_keeps_subscript_offset(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
            torch.zeros([128, 200], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_atomic_add_shifted,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
        )
        expected = torch.zeros_like(output)
        expected[:, 8:] = 2.0 * (args[0] @ args[1])
        torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("tl.split", code)
        self.assertEqual(code.count("tl.atomic_add("), 2)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_no_split_for_non_tile_store_index(self):
        """out[tile_m, tile_n.index * 2] has no tile position to offset per piece."""
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 192], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_strided_out,
            args,
            block_sizes=[32, 64, 64],
            epilogue_subtile=2,
        )
        expected = torch.zeros_like(output)
        expected[:, ::2] = 2.0 * (args[0] @ args[1])
        torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-2)
        _assert_no_split_codegen(self, code)

    @onlyBackends("triton")
    @skipIfRefEager("test checks generated backend code")
    def test_no_split_when_tile_also_feeds_tensor_indexer(self):
        """idx[tile_m] would keep the full tile while the value is split."""
        args = (
            torch.arange(8, device=DEVICE, dtype=torch.int32),
            torch.randn([8, 16], device=DEVICE, dtype=torch.float32),
            16,
        )
        _, expected = code_and_output(store_with_tile_indexer, args)
        code, output = code_and_output(
            store_with_tile_indexer, args, epilogue_subtile=4
        )
        torch.testing.assert_close(output, expected)
        _assert_no_split_codegen(self, code)

    @skipIfXPU("epilogue_subtile_autotune check uses CUDA device properties")
    def test_autotune_field_enabled_for_large_k(self):
        args = (
            torch.randn([128, 1024], device=DEVICE, dtype=torch.float32),
            torch.randn([1024, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        fields = matmul_with_bias.bind(args).config_spec._flat_fields()
        if _supports_epilogue_subtile_autotune():
            self.assertIn("epilogue_subtile", fields)
            fragment = fields["epilogue_subtile"]
            assert isinstance(fragment, EnumFragment)
            self.assertEqual(fragment.choices, (None, 2))
        else:
            self.assertNotIn("epilogue_subtile", fields)

    def test_autotune_field_disabled_for_small_k(self):
        args = (
            torch.randn([128, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        fields = matmul_with_bias.bind(args).config_spec._flat_fields()
        self.assertNotIn("epilogue_subtile", fields)

    @skipIfXPU("uses torch.cuda.get_device_capability() - Blackwell is CUDA-specific")
    def test_autotune_field_large_k_allows_s4_on_blackwell(self):
        args = (
            torch.randn([128, 16384], device=DEVICE, dtype=torch.float32),
            torch.randn([16384, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        fields = matmul_with_bias.bind(args).config_spec._flat_fields()
        if not _supports_epilogue_subtile_autotune():
            self.assertNotIn("epilogue_subtile", fields)
            return

        self.assertIn("epilogue_subtile", fields)
        fragment = fields["epilogue_subtile"]
        assert isinstance(fragment, EnumFragment)
        expected_choices = (
            (None, 2, 4) if torch.cuda.get_device_capability() >= (10, 0) else (None, 2)
        )
        self.assertEqual(fragment.choices, expected_choices)


if __name__ == "__main__":
    unittest.main()
