from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(autotune_effort="none")
def padded_pointwise(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    out = torch.zeros([n + 32], device=x.device, dtype=x.dtype)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n] + 1.0
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def shifted_range_pointwise(
    x: torch.Tensor, out: torch.Tensor, begin: int, end: int, shift: int
) -> torch.Tensor:
    shift = hl.specialize(shift)
    for tile_n in hl.tile(begin, end):
        out[tile_n.index + shift] = x[tile_n] + 1.0
    return out


@helion.kernel(autotune_effort="none")
def transposed_padded_pointwise(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_n.index + 8, tile_m] = (x[tile_m, tile_n] + 1.0).T
    return out


@helion.kernel(autotune_effort="none")
def device_range_pointwise(
    x: torch.Tensor, out: torch.Tensor, begin: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(begin[0], end[0]):
            out[tile_m, tile_n.index + 8] = x[tile_m, tile_n] + 1.0
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def shifted_range_atomic_add(
    x: torch.Tensor, out: torch.Tensor, begin: int, end: int
) -> torch.Tensor:
    for tile_n in hl.tile(begin, end):
        hl.atomic_add(out, [tile_n.index + 8], x[tile_n] + 1.0)
    return out


@onlyBackends("triton")
class TestEpilogueSubtilingBounds(TestCase):
    def _assert_split(self, code: str, split_factor: int, op: str = "store") -> None:
        self.assertIn("tl.split", code)
        self.assertEqual(code.count(f"tl.{op}("), split_factor)

    @skipIfRefEager("test requires epilogue subtiling")
    def test_ragged_tile_with_padded_output(self) -> None:
        """Reproduce split stores writing past the loop into output padding."""
        for n in (50, 63):
            x = torch.arange(n, device=DEVICE, dtype=torch.float32)
            expected = torch.zeros([n + 32], device=DEVICE)
            expected[:n] = x + 1.0
            for split_factor in (None, 2, 4):
                with self.subTest(n=n, split_factor=split_factor):
                    code, output = code_and_output(
                        padded_pointwise,
                        (x,),
                        block_sizes=[32],
                        indexing="pointer",
                        epilogue_subtile=split_factor,
                    )
                    if split_factor is None:
                        self.assertNotIn("tl.split", code)
                        self.assertEqual(code.count("tl.store("), 1)
                    else:
                        self._assert_split(code, split_factor)
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @skipIfRefEager("test requires epilogue subtiling")
    def test_shifted_output_with_runtime_range(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for begin, end, shift in ((7, 57, 8), (12, 51, -4)):
            expected = torch.full([136], -9.0, device=DEVICE)
            expected[begin + shift : end + shift] = x[begin:end] + 1.0
            for split_factor in (2, 4):
                with self.subTest(
                    begin=begin, end=end, shift=shift, split_factor=split_factor
                ):
                    out = torch.full_like(expected, -9.0)
                    code, output = code_and_output(
                        shifted_range_pointwise,
                        (x, out, begin, end, shift),
                        block_sizes=[32],
                        indexing="pointer",
                        epilogue_subtile=split_factor,
                    )
                    self._assert_split(code, split_factor)
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @skipIfRefEager("test requires epilogue subtiling")
    def test_transposed_output_with_two_ragged_dims(self) -> None:
        m, n = 19, 50
        x = torch.arange(m * n, device=DEVICE, dtype=torch.float32).reshape(m, n)
        expected = torch.full([n + 40, m + 16], -9.0, device=DEVICE)
        expected[8 : n + 8, :m] = (x + 1.0).T
        for split_factor in (2, 4):
            with self.subTest(split_factor=split_factor):
                out = torch.full_like(expected, -9.0)
                code, output = code_and_output(
                    transposed_padded_pointwise,
                    (x, out),
                    block_sizes=[16, 32],
                    indexing="pointer",
                    epilogue_subtile=split_factor,
                )
                self._assert_split(code, split_factor)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @skipIfRefEager("test requires epilogue subtiling")
    def test_device_loop_with_data_dependent_range(self) -> None:
        x = torch.arange(3 * 96, device=DEVICE, dtype=torch.float32).reshape(3, 96)
        begin = torch.tensor([7], device=DEVICE, dtype=torch.int32)
        end = torch.tensor([57], device=DEVICE, dtype=torch.int32)
        expected = torch.full([3, 136], -9.0, device=DEVICE)
        expected[:, 15:65] = x[:, 7:57] + 1.0
        for split_factor in (2, 4):
            with self.subTest(split_factor=split_factor):
                out = torch.full_like(expected, -9.0)
                code, output = code_and_output(
                    device_range_pointwise,
                    (x, out, begin, end),
                    block_sizes=[2, 32],
                    indexing="pointer",
                    epilogue_subtile=split_factor,
                )
                self._assert_split(code, split_factor)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @skipIfRefEager("test requires epilogue subtiling")
    def test_atomic_add_with_shifted_runtime_range(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        begin, end = 7, 57
        expected = torch.full([136], 5.0, device=DEVICE)
        expected[begin + 8 : end + 8] += x[begin:end] + 1.0
        for split_factor in (2, 4):
            with self.subTest(split_factor=split_factor):
                out = torch.full_like(expected, 5.0)
                code, output = code_and_output(
                    shifted_range_atomic_add,
                    (x, out, begin, end),
                    block_sizes=[32],
                    indexing="pointer",
                    atomic_indexing="pointer",
                    epilogue_subtile=split_factor,
                )
                self._assert_split(code, split_factor, "atomic_add")
                torch.testing.assert_close(output, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
