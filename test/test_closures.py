from __future__ import annotations

from pathlib import Path
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl

basic_kernels = import_path(Path(__file__).parent / "data/basic_kernels.py")


global_tensor = torch.randn([512], device=DEVICE)


@helion.kernel(static_shapes=False)
def sin_func_arg(a, fn) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size()):
        out[tile] = fn(torch.sin(a[tile]), tile)
    return out


class TestClosures(RefEagerTestBase, TestCase):
    def test_add_global(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, out = code_and_output(basic_kernels.use_globals, args)
        torch.testing.assert_close(
            out,
            torch.sin(args[0] + basic_kernels.global_tensor[None, :])
            + basic_kernels.global_float,
        )
        self.assertExpectedJournal(code)

    def test_fn_arg_with_global(self):
        def fn_with_global(x, tile) -> torch.Tensor:
            return x + global_tensor[tile]

        args = (torch.randn([512], device=DEVICE), fn_with_global)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + global_tensor)
        self.assertExpectedJournal(code)

    def test_fn_arg_with_global_different_file(self):
        args = (torch.randn([512], device=DEVICE), basic_kernels.add_global_float)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + basic_kernels.global_float)
        self.assertExpectedJournal(code)

    def test_fn_arg_with_closure(self):
        def fn_with_closure(x, tile) -> torch.Tensor:
            return x + closure_tensor[tile]

        closure_tensor = torch.randn([512], device=DEVICE)
        args = (torch.randn([512], device=DEVICE), fn_with_closure)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + closure_tensor)
        self.assertExpectedJournal(code)

    def test_fn_arg_with_nested_closure(self):
        def fn_with_closure_a(x, tile) -> torch.Tensor:
            return x + closure_tensor[tile]

        def fn_with_closure_b(x, tile) -> torch.Tensor:
            return fn_with_closure_a(x, tile) + int_closure

        closure_tensor = torch.randn([512], device=DEVICE)
        int_closure = 42
        args = (torch.randn([512], device=DEVICE), fn_with_closure_b)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + closure_tensor + int_closure)
        self.assertExpectedJournal(code)

    def test_fn_called_on_host(self):
        def alloc(x):
            return torch.empty_like(x)

        @helion.kernel
        def call_func_arg_on_host(a, alloc) -> torch.Tensor:
            out = alloc(a)
            for tile in hl.tile(a.size()):
                out[tile] = a[tile].sin()
            return out

        args = (torch.randn([512], device=DEVICE), alloc)
        code, out = code_and_output(call_func_arg_on_host, args)
        torch.testing.assert_close(out, args[0].sin())
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
