from __future__ import annotations

import operator
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfMTIA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfXPU
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
        grad_shape=None,
        autotune=False,
        autotune_effort="none",
        rtol=1e-5,
        atol=1e-5,
        inputs_fn=None,
    ):
        """
        Validate helion.experimental.backward against PyTorch autograd.

        ``inputs_fn`` overrides the default randn input factory (callable
        returning a list of tensors). ``rtol``/``atol`` configure the
        tolerance passed to ``torch.testing.assert_close``.

        Returns (helion_code, triton_code) for additional assertions.
        """
        if inputs_fn is None:
            inputs = [
                torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                for _ in range(n_inputs)
            ]
        else:
            inputs = inputs_fn()
        if grad_shape is None:
            grad_shape = shape
        grad_out = torch.randn(*grad_shape, device=DEVICE, dtype=torch.float32)

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
                torch.testing.assert_close(grads[i], inp_pt.grad, rtol=rtol, atol=atol)
        else:
            torch.testing.assert_close(grads, inputs_pt[0].grad, rtol=rtol, atol=atol)

        self.assertIn("backward_kernel", helion_code)

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

    @skipIfXPU("Timeout on XPU")
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

    @skipIfXPU("Timeout on XPU")
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
        self.assertTrue(getattr(bound, "_backward_compiled_cache", None))
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

    def test_sum_reduction_square_shape(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.sum(-1), 1, shape=(32, 32), grad_shape=(32,)
        )

    def test_multi_output_reordered_stores(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            aux = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                x_tile = x[tile_m, :]
                aux[tile_m] = x_tile.sum(-1)
                out[tile_m, :] = x_tile * 2
            return out, aux

        m, n = 64, 32
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel(x.clone())
        grad_out = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        grad_aux = torch.randn(m, device=DEVICE, dtype=torch.float32)
        result = helion.experimental.backward(
            kernel, (grad_out, grad_aux), x, return_code=True
        )
        grads, helion_code, triton_code = result

        x_ref = x.clone().requires_grad_(True)
        loss = (x_ref * 2 * grad_out).sum() + (x_ref.sum(-1) * grad_aux).sum()
        loss.backward()
        torch.testing.assert_close(grads, x_ref.grad, rtol=1e-4, atol=1e-4)

        self.assertIn("backward_kernel", helion_code)

    def test_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.sum(-1), 1, shape=(64, 32), grad_shape=(64,)
        )

    def test_sum_reduction_middle_dim(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, c], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].sum(-2)
            return out

        self._check_backward(
            kernel, lambda x: x.sum(1), 1, shape=(8, 16, 32), grad_shape=(8, 32)
        )

    def test_mean_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].mean(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.mean(-1), 1, shape=(64, 32), grad_shape=(64,)
        )

    def test_weighted_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] * w[tile_m, :]).sum(-1)
            return out

        self._check_backward(
            kernel,
            lambda x, w: (x * w).sum(-1),
            2,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_sum_mul_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] * 2).sum(-1)
            return out

        self._check_backward(
            kernel, lambda x: (x * 2).sum(-1), 1, shape=(64, 32), grad_shape=(64,)
        )

    def test_exp_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.exp(x[tile_m, :]).sum(-1)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.exp(x).sum(-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_sin_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.sin(x[tile_m, :]).sum(-1)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.sin(x).sum(-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_squared_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] * x[tile_m, :]).sum(-1)
            return out

        self._check_backward(
            kernel, lambda x: (x * x).sum(-1), 1, shape=(64, 32), grad_shape=(64,)
        )

    def test_exp_mean_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.exp(x[tile_m, :]).mean(-1)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.exp(x).mean(-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_amax_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.amax(x[tile_m, :], -1)
            return out

        self._check_backward(
            kernel, lambda x: x.amax(-1), 1, shape=(64, 32), grad_shape=(64,)
        )

    def test_amin_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.amin(x[tile_m, :], -1)
            return out

        self._check_backward(
            kernel, lambda x: x.amin(-1), 1, shape=(64, 32), grad_shape=(64,)
        )

    def test_softmax(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.nn.functional.softmax(x, dim=1),
            1,
            shape=(64, 32),
        )

    def test_softmax_decomposed(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp = torch.exp(values - amax)
                sum_exp = torch.sum(exp, dim=1, keepdim=True)
                out[tile_n, :] = exp / sum_exp
            return out

        self._check_backward(
            kernel,
            lambda x: torch.nn.functional.softmax(x, dim=1),
            1,
            shape=(64, 32),
        )

    def test_batch_softmax_3d(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            b, m, n = x.shape
            out = torch.empty_like(x)
            for tile_b, tile_m in hl.tile([b, m]):
                row = x[tile_b, tile_m, :]
                mx = torch.amax(row, -1, True)
                e = torch.exp(row - mx)
                out[tile_b, tile_m, :] = e / torch.sum(e, -1, True)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.nn.functional.softmax(x, dim=-1),
            1,
            shape=(4, 16, 32),
        )

    def test_rms_norm(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                x_tile = x[tile_m, :]
                x_squared = x_tile * x_tile
                mean_x_squared = torch.mean(x_squared, dim=-1)
                inv_rms = torch.rsqrt(mean_x_squared + eps)
                out[tile_m, :] = x_tile * inv_rms[:, None]
            return out

        def ref(x):
            var = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(var + 1e-5)

        self._check_backward(kernel, ref, 1, shape=(64, 32))

    def test_layer_norm(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :]
                mean_val = torch.sum(acc, dim=-1) / n
                centered = acc - mean_val[:, None]
                var_val = torch.sum(centered * centered, dim=-1) / n
                rstd_val = torch.rsqrt(var_val + eps)
                out[tile_m, :] = centered * rstd_val[:, None]
            return out

        def ref(x):
            mean = x.mean(-1, keepdim=True)
            var = ((x - mean) ** 2).mean(-1, keepdim=True)
            return (x - mean) * torch.rsqrt(var + 1e-5)

        self._check_backward(kernel, ref, 1, shape=(64, 32))

    def test_rms_norm_multiout(self):
        @helion.kernel(autotune_effort="none")
        def rms_norm_fwd(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.size()
            assert weight.size(0) == n
            out = torch.empty_like(x)
            inv_rms = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                x_tile = x[tile_m, :].to(torch.float32)
                x_squared = x_tile * x_tile
                mean_x_squared = torch.mean(x_squared, dim=-1)
                inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
                normalized = x_tile * inv_rms_tile[:, None]
                out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(
                    out.dtype
                )
                inv_rms[tile_m] = inv_rms_tile.to(out.dtype)
            return out, inv_rms.reshape(-1, 1)

        m, n = 64, 32
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        w = torch.randn(n, device=DEVICE, dtype=torch.float32)

        out, inv_rms = rms_norm_fwd(x.clone(), w.clone())

        grad_out = torch.randn_like(out)
        grad_inv_rms = torch.randn_like(inv_rms)

        result = helion.experimental.backward(
            rms_norm_fwd,
            (grad_out, grad_inv_rms),
            x,
            w,
            return_code=True,
        )
        grads, helion_code, triton_code = result
        assert isinstance(grads, tuple)

        # Reference: use the same math as the helion kernel via PyTorch autograd
        x_ref = x.clone().to(torch.float32).requires_grad_(True)
        w_ref = w.clone().to(torch.float32).requires_grad_(True)
        variance = x_ref.pow(2).mean(-1, keepdim=True)
        inv_rms_ref = torch.rsqrt(variance + 1e-5)
        out_ref = x_ref * inv_rms_ref * w_ref
        inv_rms_out = inv_rms_ref  # shape [M, 1], matches forward output
        loss = (out_ref * grad_out).sum() + (inv_rms_out * grad_inv_rms).sum()
        loss.backward()

        torch.testing.assert_close(grads[0], x_ref.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(grads[1], w_ref.grad, rtol=1e-4, atol=1e-4)

        self.assertIn("backward_kernel", helion_code)

    def test_sum_reduction_last_dim_3d(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, b], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].sum(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.sum(-1), 1, shape=(8, 16, 32), grad_shape=(8, 16)
        )

    def test_mean_reduction_3d_last_dim(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, b], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].mean(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.mean(-1), 1, shape=(8, 16, 32), grad_shape=(8, 16)
        )

    def test_mean_reduction_middle_dim(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, c], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].mean(-2)
            return out

        self._check_backward(
            kernel, lambda x: x.mean(1), 1, shape=(8, 16, 32), grad_shape=(8, 32)
        )

    def test_amax_reduction_3d(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, b], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].amax(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.amax(-1), 1, shape=(8, 16, 32), grad_shape=(8, 16)
        )

    def test_amin_reduction_3d(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, b], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].amin(-1)
            return out

        self._check_backward(
            kernel, lambda x: x.amin(-1), 1, shape=(8, 16, 32), grad_shape=(8, 16)
        )

    def test_logsumexp_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                tile = x[tile_m, :]
                max_val = tile.amax(-1)
                out[tile_m] = (
                    torch.log(torch.exp(tile - max_val[:, None]).sum(-1)) + max_val
                )
            return out

        self._check_backward(
            kernel,
            lambda x: torch.logsumexp(x, dim=-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_abs_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.abs(x[tile_m, :]).sum(-1)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.abs(x).sum(-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_squared_mean_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                tile = x[tile_m, :]
                out[tile_m] = (tile * tile).mean(-1)
            return out

        self._check_backward(
            kernel,
            lambda x: (x * x).mean(-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_exp_amax_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.exp(x[tile_m, :]).amax(-1)
            return out

        self._check_backward(
            kernel,
            lambda x: torch.exp(x).amax(-1),
            1,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_two_input_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] * y[tile_m, :]).sum(-1)
            return out

        self._check_backward(
            kernel,
            lambda x, y: (x * y).sum(-1),
            2,
            shape=(64, 32),
            grad_shape=(64,),
        )

    def test_softmax_3d_last_dim(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty_like(x)
            for tile_a in hl.tile(a):
                tile = x[tile_a, :, :]
                max_val = tile.amax(-1)
                exp_val = torch.exp(tile - max_val[:, :, None])
                out[tile_a, :, :] = exp_val / exp_val.sum(-1)[:, :, None]
            return out

        self._check_backward(
            kernel,
            lambda x: torch.softmax(x, dim=-1),
            1,
            shape=(4, 8, 32),
        )

    def test_reciprocal_sum_reduction(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = torch.reciprocal(x[tile_m, :]).sum(-1)
            return out

        # Inputs are bounded away from zero so the reciprocal stays
        # numerically stable; the looser tolerance accounts for the
        # 1/x derivative's high sensitivity near zero.
        self._check_backward(
            kernel,
            lambda x: torch.reciprocal(x).sum(-1),
            1,
            shape=(32, 64),
            grad_shape=(32,),
            rtol=1e-4,
            atol=1e-4,
            inputs_fn=lambda: [
                torch.randn(32, 64, device=DEVICE, dtype=torch.float32).abs() + 0.1
            ],
        )

    def test_amax_reduction_middle_dim_3d(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            a, b, c = x.shape
            out = torch.empty([a, c], dtype=x.dtype, device=x.device)
            for tile_a in hl.tile(a):
                out[tile_a, :] = x[tile_a, :, :].amax(1)
            return out

        self._check_backward(
            kernel, lambda x: x.amax(1), 1, shape=(8, 16, 32), grad_shape=(8, 32)
        )

    def test_sum_reduction_dim0(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = x[:, tile_n].sum(0)
            return out

        self._check_backward(
            kernel, lambda x: x.sum(0), 1, shape=(32, 64), grad_shape=(64,)
        )

    def test_sum_reduction_keepdim(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :].sum(-1, keepdim=True)
            return out

        self._check_backward(
            kernel,
            lambda x: x.sum(-1, keepdim=True),
            1,
            shape=(64, 32),
            grad_shape=(64, 1),
        )

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
            shape=(64, 32),
            autotune=True,
            autotune_effort="quick",
        )


if __name__ == "__main__":
    unittest.main()
