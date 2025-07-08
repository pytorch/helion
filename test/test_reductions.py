from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(
        [n],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel()
def sum_kernel_keepdims(x: torch.Tensor) -> torch.Tensor:
    _n, m = x.size()
    out = torch.empty(
        [1, m],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_m in hl.tile(m):
        out[:, tile_m] = x[:, tile_m].sum(0, keepdim=True)
    return out


@helion.kernel(config={"block_sizes": [1]})
def reduce_kernel(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32
) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(
        [n],
        dtype=out_dtype,
        device=x.device,
    )
    for tile_n in hl.tile(n):
        out[tile_n] = fn(x[tile_n, :], dim=-1)
    return out


class TestReductions(TestCase):
    def test_sum(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, output = code_and_output(sum_kernel, args, block_size=1)
        torch.testing.assert_close(output, args[0].sum(-1), rtol=1e-04, atol=1e-04)
        self.assertExpectedJournal(code)

    def test_sum_keepdims(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, output = code_and_output(
            sum_kernel_keepdims, args, block_size=16, indexing="block_ptr"
        )
        torch.testing.assert_close(
            output, args[0].sum(0, keepdim=True), rtol=1e-04, atol=1e-04
        )
        self.assertExpectedJournal(code)

    def test_argmin_argmax(self):
        for fn in (torch.argmin, torch.argmax):
            args = (torch.randn([512, 512], device=DEVICE), fn, torch.int64)
            code, output = code_and_output(
                reduce_kernel, args, block_size=16, indexing="block_ptr"
            )
            torch.testing.assert_close(output, args[1](args[0], dim=-1))
        self.assertExpectedJournal(code)

    def test_reduction_functions(self):
        for reduction_loop in (None, 16):
            for block_size in (1, 16):
                for indexing in ("block_ptr", "pointer"):
                    for fn in (
                        torch.amax,
                        torch.amin,
                        torch.prod,
                        torch.sum,
                        torch.mean,
                    ):
                        args = (torch.randn([512, 512], device=DEVICE), fn)
                        _, output = code_and_output(
                            reduce_kernel,
                            args,
                            block_size=block_size,
                            indexing=indexing,
                            reduction_loop=reduction_loop,
                        )
                        torch.testing.assert_close(
                            output, fn(args[0], dim=-1), rtol=1e-3, atol=1e-3
                        )

    def test_mean(self):
        args = (torch.randn([512, 512], device=DEVICE), torch.mean, torch.float32)
        self.assertExpectedJournal(reduce_kernel.bind(args)._debug_str())
        code, output = code_and_output(
            reduce_kernel, args, block_size=8, indexing="block_ptr"
        )
        torch.testing.assert_close(output, args[1](args[0], dim=-1))
        self.assertExpectedJournal(code)

    def test_sum_looped(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, output = code_and_output(
            sum_kernel, args, block_size=2, reduction_loop=64
        )
        torch.testing.assert_close(output, args[0].sum(-1), rtol=1e-04, atol=1e-04)
        self.assertExpectedJournal(code)

    def test_argmin_argmax_looped(self):
        for fn in (torch.argmin, torch.argmax):
            args = (torch.randn([512, 512], device=DEVICE), fn, torch.int64)
            code, output = code_and_output(
                reduce_kernel,
                args,
                block_size=1,
                indexing="block_ptr",
                reduction_loop=16,
            )
            torch.testing.assert_close(output, args[1](args[0], dim=-1))
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
