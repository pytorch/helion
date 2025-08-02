from __future__ import annotations

from pathlib import Path
import unittest

import torch

import helion
from helion import Config
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
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


class TestMatmul(RefEagerTestDisabled, TestCase):
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
            indexing="block_ptr",
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
