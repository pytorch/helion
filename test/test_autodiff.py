from __future__ import annotations

import operator
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfCpu
from helion._testing import skipIfMTIA
from helion._testing import skipIfNotTriton
import helion.language as hl


@skipIfMTIA("autodiff not tested on MTIA")
@skipIfNotTriton("autodiff not tested on non Triton backends")
class TestAutodiff(RefEagerTestDisabled, TestCase):
    def _check_backward(
        self,
        kernel_fn,
        pytorch_fn,
        n_inputs,
        shape=(128,),
        autotune=False,
        autotune_effort="none",
    ):
        """
        Validate helion.experimental.backward against PyTorch autograd.

        Creates n_inputs random tensors of the given shape, runs the helion kernel
        forward and backward, compares gradients against PyTorch autograd, and
        checks assertExpectedJournal for both helion and triton code.

        Returns (helion_code, triton_code) for additional assertions.
        """
        inputs = [
            torch.randn(*shape, device=DEVICE, dtype=torch.float32)
            for _ in range(n_inputs)
        ]
        grad_out = torch.randn(*shape, device=DEVICE, dtype=torch.float32)

        kernel_fn(*[inp.clone() for inp in inputs])
        result = helion.experimental.backward(
            kernel_fn,
            grad_out,
            *inputs,
            return_code=True,
            autotune=autotune,
            autotune_effort=autotune_effort,
        )
        grads, helion_code, triton_code = result

        inputs_pt = [inp.requires_grad_(True) for inp in inputs]
        pytorch_fn(*inputs_pt).backward(grad_out)

        if isinstance(grads, tuple):
            for i, inp_pt in enumerate(inputs_pt):
                torch.testing.assert_close(grads[i], inp_pt.grad, rtol=1e-5, atol=1e-5)
        else:
            torch.testing.assert_close(grads, inputs_pt[0].grad, rtol=1e-5, atol=1e-5)

        self.assertIn("backward_kernel", helion_code)
        self.assertIsNotNone(triton_code)

        if not autotune:
            self.assertExpectedJournal(helion_code)
            self.assertExpectedJournal(triton_code)

        return helion_code, triton_code

    def test_add(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] + y[tile]
            return out

        self._check_backward(kernel, operator.add, 2)

    def test_mul(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile]
            return out

        self._check_backward(kernel, operator.mul, 2)

    def test_sub(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] - y[tile]
            return out

        self._check_backward(kernel, operator.sub, 2)

    def test_fma(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x, y, z = torch.broadcast_tensors(x, y, z)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile] + z[tile]
            return out

        self._check_backward(kernel, lambda x, y, z: x * y + z, 3)

    def test_x_squared(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * x[tile]
            return out

        self._check_backward(kernel, lambda x: x * x, 1)

    def test_sum_of_products(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x, y, z = torch.broadcast_tensors(x, y, z)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile] + y[tile] * z[tile]
            return out

        self._check_backward(kernel, lambda x, y, z: x * y + y * z, 3)

    def test_triple_mul(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x, y, z = torch.broadcast_tensors(x, y, z)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile] * z[tile]
            return out

        self._check_backward(kernel, lambda x, y, z: x * y * z, 3)

    def test_sin(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.sin(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.sin(x), 1)

    def test_exp(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.exp(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.exp(x), 1)

    def test_relu(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.relu(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.relu(x), 1)

    def test_log(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.log(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.log(x), 1)

    def test_tanh(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.tanh(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.tanh(x), 1)

    def test_sigmoid(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.sigmoid(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.sigmoid(x), 1)

    def test_sin_cos(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.sin(x[tile]) * torch.cos(x[tile])
            return out

        self._check_backward(kernel, lambda x: torch.sin(x) * torch.cos(x), 1)

    def test_exp_sin(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.exp(torch.sin(x[tile]))
            return out

        self._check_backward(kernel, lambda x: torch.exp(torch.sin(x)), 1)

    def test_x_times_sin(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * torch.sin(x[tile])
            return out

        self._check_backward(kernel, lambda x: x * torch.sin(x), 1)

    def test_sin_squared(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                sin_x = torch.sin(x[tile])
                out[tile] = sin_x * sin_x
            return out

        self._check_backward(kernel, lambda x: torch.sin(x) ** 2, 1)

    def test_softplus(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.log(1.0 + torch.exp(x[tile]))
            return out

        self._check_backward(kernel, lambda x: torch.log(1.0 + torch.exp(x)), 1)

    def test_exp_x_sin_y(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.exp(x[tile]) * torch.sin(y[tile])
            return out

        self._check_backward(kernel, lambda x, y: torch.exp(x) * torch.sin(y), 2)

    def test_sin_x_cos_y(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.sin(x[tile]) + torch.cos(y[tile])
            return out

        self._check_backward(kernel, lambda x, y: torch.sin(x) + torch.cos(y), 2)

    def test_backward_cache(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile]
            return out

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, device=DEVICE, dtype=torch.float32)
        grad_out = torch.randn(64, device=DEVICE, dtype=torch.float32)

        kernel(x.clone(), y.clone())
        helion.experimental.backward(kernel, grad_out, x, y)

        # Second call should hit the compiled cache on bound
        bound = kernel.bind((x, y))
        self.assertIsNotNone(bound._backward_compiled)
        helion.experimental.backward(kernel, grad_out, x, y)

    def test_load_store_load_pattern(self):
        @helion.kernel(autotune_effort="none")
        def load_store_load(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                val1 = x[tile]  # load x (original)
                x[tile] = val1 * 2  # store 2*x back to x
                val2 = x[tile]  # load x (should get 2*x)
                out[tile] = torch.sin(val2)  # compute sin(2*x)
            return out

        self._check_backward(load_store_load, lambda x: torch.sin(x * 2), 1)

    def test_error_multiple_tile_loops(self):
        @helion.kernel(autotune_effort="none")
        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.shape
            _, n = b.shape
            c = torch.zeros([m, n], dtype=a.dtype, device=a.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = c[tile_m, tile_n]
                for tile_k in hl.tile(k):
                    acc = acc + a[tile_m, tile_k] @ b[tile_k, tile_n]
                c[tile_m, tile_n] = acc
            return c

        a = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        b = torch.randn(64, 32, device=DEVICE, dtype=torch.float32)
        kernel(a, b)
        grad_out = torch.randn(32, 32, device=DEVICE, dtype=torch.float32)

        with self.assertRaises(helion.exc.AutodiffNotSupported):
            helion.experimental.backward(kernel, grad_out, a, b)

    def test_error_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn(64, 32, device=DEVICE, dtype=torch.float32)
        kernel(x)
        grad_out = torch.randn(64, device=DEVICE, dtype=torch.float32)

        with self.assertRaises(helion.exc.AutodiffNotSupported):
            helion.experimental.backward(kernel, grad_out, x)

    @skipIfCpu("fails on Triton CPU backend")
    def test_backward_autotune(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.sin(x[tile]) * y[tile]
            return out

        self._check_backward(
            kernel,
            lambda x, y: torch.sin(x) * y,
            2,
            shape=(128, 64),
            autotune=True,
            autotune_effort="quick",
        )


if __name__ == "__main__":
    unittest.main()
