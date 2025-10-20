from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


class TestControlFlow(RefEagerTestBase, TestCase):
    def test_if_arg(self):
        @helion.kernel()
        def fn(x, v):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code0, result = code_and_output(
            fn,
            (x, 5),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        code1, result = code_and_output(
            fn,
            (x, 10),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertEqual(code0, code1)
        self.assertExpectedJournal(code0)

    def test_if_arg_indexed_scalar(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)

            for idx in hl.grid(x.shape[0]):
                # Since `y[idx]` is a scalar, comparing it against 0 will also create a scalar.
                if y[idx] != 0:
                    output[idx] = x[idx] * 2
                else:
                    output[idx] = x[idx]

            return output

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        y = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
        expected = torch.tensor([1.0, 4.0, 3.0, 8.0], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, y),
        )
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_if_arg_tensor_sum(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)

            for tile in hl.tile(x.shape[0]):
                # Since `y[idx]` is a tensor, comparing it against 0 will also create a tensor.
                # if condition must takes a scalar, therefore we call .sum() to reduce the tensor to a scalar.
                if (y[tile] != 0).sum():
                    output[tile] = x[tile] * 2
                if (
                    y[tile] == 0
                ).sum():  # TODO(yf225): `else:` raises MLIR error in Triton, so we use a second if.
                    output[tile] = x[tile]

            return output

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        y = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
        expected = torch.tensor([1.0, 4.0, 3.0, 8.0], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, y),
            block_size=1,
        )
        torch.testing.assert_close(result, expected)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_constant_true(self):
        @helion.kernel(
            config={
                "block_sizes": [128, 1],
                "flatten_loop": True,
                "indexing": "block_ptr",
            }
        )
        def fn(x):
            out = torch.empty_like(x)
            v = 4
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        self.assertExpectedJournal(code)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_constant_false(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            out = torch.empty_like(x)
            v = 15
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertExpectedJournal(code)

    def test_error_in_non_taken_branch(self):
        def mul_relu_block_back_spec(x, y, dz):
            z = torch.relu(x * y[:, None])
            grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
            return grad_x, grad_y

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
        def mul_relu_block_backward_kernel(
            x: torch.Tensor,
            y: torch.Tensor,
            dz: torch.Tensor,
            use_atomics: hl.constexpr = False,
        ):
            # Get tensor sizes
            m, n = x.shape
            # Create output tensor for gradients
            dx = torch.empty_like(x)

            if use_atomics:
                dy = torch.zeros_like(y)
            else:
                dy = torch.empty_like(x)

            # Use Helion to tile the computation
            for tile_i, tile_j in hl.tile([m, n]):
                # Get input tiles
                x_tile = x[tile_i, tile_j]
                y_tile = y[tile_i]
                dz_tile = dz[tile_i, tile_j]

                # For ReLU, gradient is 1 where input > 0, 0 otherwise
                relu_mask = (x_tile * y_tile[:, None]) > 0
                # Chain rule: dx = dz * relu_grad * y
                relu_grad = torch.where(relu_mask, 1, 0)
                dx[tile_i, tile_j] = dz_tile * relu_grad * y_tile[:, None]

                # Chain rule: dy = dz * relu_grad * x -> backwards of broadcast(sum)
                if use_atomics:
                    local_dy_grad = torch.sum(dz_tile * relu_grad * x_tile, dim=1)
                    hl.atomic_add(dy, [tile_i], local_dy_grad)
                else:
                    local_dy_grad = dz_tile * relu_grad * x_tile
                    dy[tile_i, tile_j] = local_dy_grad

            if use_atomics:
                return dx, dy
            return dx, dy.sum(axis=-1)

        x = torch.randn(512, 1024, device=DEVICE, requires_grad=True)
        y = torch.randn(512, device=DEVICE, requires_grad=True)
        dz = torch.randn(512, 1024, device=DEVICE)
        expected = mul_relu_block_back_spec(x, y, dz)
        torch.testing.assert_close(
            mul_relu_block_backward_kernel(x, y, dz, False),
            expected,
            atol=1e-4,
            rtol=1e-4,
        )
        code, output = code_and_output(
            mul_relu_block_backward_kernel,
            (x, y, dz, True),
        )
        self.assertExpectedJournal(code)
        torch.testing.assert_close(
            output,
            expected,
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
