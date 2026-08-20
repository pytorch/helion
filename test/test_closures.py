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
from helion._testing import onlyBackends
import helion.language as hl

basic_kernels = import_path(Path(__file__).parent / "data/basic_kernels.py")


# Initialized lazily in setUp() to avoid CUDA init at import time,
# which causes "CUDA unknown error" with pytest-xdist worker spawning.
global_tensor = None


@helion.kernel(static_shapes=False)
def sin_func_arg(a, fn) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size()):
        out[tile] = fn(torch.sin(a[tile]), tile)
    return out


@onlyBackends(["triton", "cute"])
class TestClosures(RefEagerTestBase, TestCase):
    def setUp(self):
        super().setUp()
        global global_tensor
        if global_tensor is None:
            global_tensor = torch.randn([512], device=DEVICE)
        basic_kernels._init_globals()

    def test_add_global(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, out = code_and_output(basic_kernels.use_globals, args)
        torch.testing.assert_close(
            out,
            torch.sin(args[0] + basic_kernels.global_tensor[None, :])
            + basic_kernels.global_float,
        )

    def test_fn_arg_with_global(self):
        def fn_with_global(x, tile) -> torch.Tensor:
            return x + global_tensor[tile]

        args = (torch.randn([512], device=DEVICE), fn_with_global)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + global_tensor)

    def test_fn_arg_with_global_different_file(self):
        args = (torch.randn([512], device=DEVICE), basic_kernels.add_global_float)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + basic_kernels.global_float)

    def test_fn_arg_with_closure(self):
        def fn_with_closure(x, tile) -> torch.Tensor:
            return x + closure_tensor[tile]

        closure_tensor = torch.randn([512], device=DEVICE)
        args = (torch.randn([512], device=DEVICE), fn_with_closure)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + closure_tensor)

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

    def test_nested_def_simple(self):
        @helion.kernel
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            def double(a):
                return a * 2.0

            for tile in hl.tile(x.size(0)):
                out[tile] = double(x[tile])
            return out

        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x * 2.0)

    def test_nested_def_closure(self):
        @helion.kernel
        def fn(x: torch.Tensor, scale: float) -> torch.Tensor:
            out = torch.empty_like(x)

            def transform(a):
                return a * scale

            for tile in hl.tile(x.size(0)):
                out[tile] = transform(x[tile])
            return out

        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(fn, (x, 3.0))
        torch.testing.assert_close(result, x * 3.0)

    def test_nested_def_multi_stmt(self):
        @helion.kernel
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            def clamp_and_scale(a):
                clamped = torch.clamp(a, min=0.0)
                return clamped * 2.0

            for tile in hl.tile(x.size(0)):
                out[tile] = clamp_and_scale(x[tile])
            return out

        x = torch.randn(128, device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, torch.clamp(x, min=0.0) * 2.0)

    def test_nested_def_multi_args(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            def weighted_add(a, b):
                return a * 0.7 + b * 0.3

            for tile in hl.tile(x.size(0)):
                out[tile] = weighted_add(x[tile], y[tile])
            return out

        x = torch.randn(128, device=DEVICE)
        y = torch.randn(128, device=DEVICE)
        code, result = code_and_output(fn, (x, y))
        torch.testing.assert_close(result, x * 0.7 + y * 0.3)


if __name__ == "__main__":
    unittest.main()
