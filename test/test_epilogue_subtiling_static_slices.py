from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessTensorDescriptor
import helion.language as hl


@helion.kernel(static_shapes=True)
def static_slice_store_with_prefix(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    for tile_m in hl.tile(m):
        out[1, tile_m, :] = x[tile_m, :] * 2.0
    return out


@helion.kernel(static_shapes=True)
def static_slice_store_with_middle_index(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.size()
    for tile_m in hl.tile(m):
        out[tile_m, 1, :] = x[tile_m, :] * 2.0
    return out


@onlyBackends("triton")
@skipIfRefEager("test checks generated tensor descriptor stores")
@skipUnlessTensorDescriptor("Tensor descriptor support is required")
class TestEpilogueSubtilingStaticSlices(TestCase):
    def _assert_descriptor_stores(self, code: str, split_factor: int | None) -> None:
        if split_factor is None:
            self.assertNotIn("tl.split(", code)
            self.assertEqual(code.count("_desc.store("), 1)
        else:
            self.assertIn("tl.split(", code)
            self.assertEqual(code.count("_desc.store("), split_factor)
        self.assertNotIn("tl.store(", code)

    def test_descriptor_static_slice_with_integer_prefix(self) -> None:
        """Reproduce splitting the row tile instead of the full column slice."""
        x = torch.arange(128 * 64, device=DEVICE, dtype=torch.float32).reshape(128, 64)
        for split_factor in (None, 2, 4):
            with self.subTest(split_factor=split_factor):
                out = torch.full([2, 128, 64], -7.0, device=DEVICE, dtype=x.dtype)
                expected = out.clone()
                expected[1] = x * 2.0
                code, result = code_and_output(
                    static_slice_store_with_prefix,
                    (x, out),
                    block_sizes=[32],
                    indexing=["pointer", "tensor_descriptor"],
                    epilogue_subtile=split_factor,
                )
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self._assert_descriptor_stores(code, split_factor)

    def test_descriptor_static_slice_with_middle_integer_s4(self) -> None:
        x = torch.randn([128, 64], device=DEVICE, dtype=torch.float32)
        out = torch.full([128, 2, 64], -7.0, device=DEVICE, dtype=x.dtype)
        expected = out.clone()
        expected[:, 1] = x * 2.0
        code, result = code_and_output(
            static_slice_store_with_middle_index,
            (x, out),
            block_sizes=[32],
            indexing=["pointer", "tensor_descriptor"],
            epilogue_subtile=4,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self._assert_descriptor_stores(code, 4)


if __name__ == "__main__":
    unittest.main()
