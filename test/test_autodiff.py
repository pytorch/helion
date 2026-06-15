from __future__ import annotations

import contextlib
import functools
import locale
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


def _force_utf8_locale(fn):
    # Triton writes its generated launcher with the process locale encoding; force
    # UTF-8 for the duration of the test so a non-UTF-8 worker doesn't fail to
    # encode non-ASCII bytes during a cold-cache kernel compile, then restore it.
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        prev = locale.setlocale(locale.LC_CTYPE)
        for _loc in ("C.UTF-8", "en_US.UTF-8", "C.utf8", "en_US.utf8"):
            try:
                locale.setlocale(locale.LC_CTYPE, _loc)
                break
            except locale.Error:
                continue
        try:
            return fn(self, *args, **kwargs)
        finally:
            with contextlib.suppress(locale.Error):
                locale.setlocale(locale.LC_CTYPE, prev)

    return wrapper


@skipIfMTIA("autodiff not tested on MTIA")
@skipIfNotTriton("autodiff not tested on non Triton backends")
@skipIfXPU("autodiff scan-path backward aborts in torch scan-HOP autograd on XPU")
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

        kernel_fn.settings.autotune_effort = autotune_effort
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

    def test_matmul_multi_tile_loops(self):
        @helion.kernel(autotune_effort="none")
        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.shape
            _, n = b.shape
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(a.dtype, b.dtype),
                device=a.device,
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        self._check_backward(
            kernel,
            operator.matmul,
            2,
            shape=(32, 32),
            inputs_fn=lambda: [
                torch.randn(32, 64, device=DEVICE, dtype=torch.float32),
                torch.randn(64, 32, device=DEVICE, dtype=torch.float32),
            ],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_single_loop_matmul(self):
        # A matmul fused into a *single* tile loop (the weight `w` is loaded
        # fully). Its backward must tile only `m`, keep `w` whole, and accumulate
        # grad_w across tiles (split-reduction) — not block-slice `w`.
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, _k = x.size()
            _, n = w.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ w[:, :]
            return out

        self._check_backward(
            kernel,
            operator.matmul,
            2,
            shape=(64, 32),
            inputs_fn=lambda: [
                torch.randn(64, 48, device=DEVICE, dtype=torch.float32),
                torch.randn(48, 32, device=DEVICE, dtype=torch.float32),
            ],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_single_loop_matmul_fused_pointwise(self):
        # Matmul fused with a (smooth) pointwise op in one tile loop — the grad
        # flows through the pointwise op into both matmul operands. A smooth op
        # (sigmoid) avoids relu's kink, which makes element-wise comparison flaky.
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, _k = x.size()
            _, n = w.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = torch.sigmoid(x[tile_m, :] @ w[:, :])
            return out

        self._check_backward(
            kernel,
            lambda x, w: torch.sigmoid(x @ w),
            2,
            shape=(64, 32),
            inputs_fn=lambda: [
                torch.randn(64, 48, device=DEVICE, dtype=torch.float32),
                torch.randn(48, 32, device=DEVICE, dtype=torch.float32),
            ],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_single_loop_matmul_bias(self):
        # Linear layer `x @ w + b` in one tile loop — the most common pattern.
        # The bias grad sums over the tiled `m` dim, so it must be a
        # split-reduction; it must NOT make the kernel full-slice `m` (which
        # would mis-tile the matmul). w-grad and b-grad both reduce over `m`.
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, _k = x.size()
            _, n = w.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ w[:, :] + b[:]
            return out

        self._check_backward(
            kernel,
            lambda x, w, b: x @ w + b,
            3,
            shape=(64, 32),
            inputs_fn=lambda: [
                torch.randn(64, 48, device=DEVICE, dtype=torch.float32),
                torch.randn(48, 32, device=DEVICE, dtype=torch.float32),
                torch.randn(32, device=DEVICE, dtype=torch.float32),
            ],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_single_loop_batched_bmm(self):
        # Batched bmm tiled over the batch dim: both operands are tiled, so each
        # batch element's grad is per-tile (no cross-tile reduction). Must NOT
        # be rejected just because there's no shared full weight.
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            bsz, m, _k = a.size()
            _, _, n = b.size()
            out = torch.empty([bsz, m, n], dtype=torch.float32, device=a.device)
            for tile_b in hl.tile(bsz):
                out[tile_b, :, :] = torch.bmm(a[tile_b, :, :], b[tile_b, :, :])
            return out

        self._check_backward(
            kernel,
            torch.bmm,
            2,
            shape=(8, 16, 10),
            grad_shape=(8, 16, 10),
            inputs_fn=lambda: [
                torch.randn(8, 16, 12, device=DEVICE, dtype=torch.float32),
                torch.randn(8, 12, 10, device=DEVICE, dtype=torch.float32),
            ],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_single_loop_hl_dot(self):
        # Explicit hl.dot (instead of @) in a single tile loop. The compute-graph
        # extraction must lower it to an aten matmul without its None defaults
        # (acc/out_dtype) being clobbered into tensors.
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, _k = x.size()
            _, n = w.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = hl.dot(x[tile_m, :], w[:, :])
            return out

        self._check_backward(
            kernel,
            operator.matmul,
            2,
            shape=(64, 32),
            inputs_fn=lambda: [
                torch.randn(64, 48, device=DEVICE, dtype=torch.float32),
                torch.randn(48, 32, device=DEVICE, dtype=torch.float32),
            ],
            rtol=1e-2,
            atol=1e-2,
        )

    def test_single_loop_matmul_low_precision(self):
        # fp16/bf16 matmul: the weight gradient must accumulate in fp32 (and be
        # zero-initialized) — accumulating in the weight's low-precision dtype
        # previously produced NaN.
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, _k = x.size()
            _, n = w.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = (
                    x[tile_m, :].to(torch.float32) @ w[:, :].to(torch.float32)
                ).to(x.dtype)
            return out

        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(64, 48, device=DEVICE, dtype=dtype)
            w = torch.randn(48, 32, device=DEVICE, dtype=dtype)
            kernel(x.clone(), w.clone())
            out = kernel(x.clone(), w.clone())
            grad_out = torch.randn_like(out)
            grad_x, grad_w = helion.experimental.backward(
                kernel, grad_out, x.clone(), w.clone(), autotune_effort="none"
            )
            self.assertFalse(torch.isnan(grad_x).any(), f"{dtype} grad_x has NaN")
            self.assertFalse(torch.isnan(grad_w).any(), f"{dtype} grad_w has NaN")
            xp = x.clone().float().requires_grad_(True)
            wp = w.clone().float().requires_grad_(True)
            (xp @ wp).backward(grad_out.float())

            # Norm-based check: low-precision grads are correct to their dtype's
            # precision (a few small-magnitude elements carry fp16/bf16 rounding
            # noise, so element-wise atol is the wrong metric here).
            def relf(a: torch.Tensor, b: torch.Tensor) -> float:
                return (a.float() - b).norm().item() / (b.norm().item() + 1e-12)

            tol = 1e-2  # well above fp16 (~3e-4) / bf16 (~2e-3), well below "wrong"
            self.assertLess(relf(grad_x, xp.grad), tol, f"{dtype} grad_x")
            self.assertLess(relf(grad_w, wp.grad), tol, f"{dtype} grad_w")

    def test_softmax_two_pass_multi_tile_loops(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            block_size_m = hl.register_block_size(m)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_size_m):
                mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                di = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    local_amax = torch.amax(values, dim=1)
                    mi_next = torch.maximum(mi, local_amax)
                    di = di * torch.exp(mi - mi_next) + torch.exp(
                        values - mi_next[:, None]
                    ).sum(dim=1)
                    mi = mi_next
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
            return out

        self._check_backward(
            kernel,
            lambda x: torch.nn.functional.softmax(x, dim=1),
            1,
            shape=(64, 32),
        )

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

    def test_example_bmm(self):
        from examples.bmm import bmm

        # examples/bmm.py uses check(16, 512, 768, 1024)
        B, M, K, N = 4, 32, 48, 64
        self._check_backward(
            bmm,
            lambda a, b: torch.bmm(a, b),
            2,
            inputs_fn=lambda: [
                torch.randn(B, M, K, device=DEVICE, dtype=torch.float32),
                torch.randn(B, K, N, device=DEVICE, dtype=torch.float32),
            ],
            grad_shape=(B, M, N),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_example_bmm_square(self):
        from examples.bmm import bmm

        B, M, K, N = 2, 32, 32, 32
        self._check_backward(
            bmm,
            lambda a, b: torch.bmm(a, b),
            2,
            inputs_fn=lambda: [
                torch.randn(B, M, K, device=DEVICE, dtype=torch.float32),
                torch.randn(B, K, N, device=DEVICE, dtype=torch.float32),
            ],
            grad_shape=(B, M, N),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_example_softmax(self):
        from examples.softmax import softmax

        # examples/softmax.py uses check(4096, 2560)
        self._check_backward(
            softmax,
            lambda x: torch.nn.functional.softmax(x, dim=-1),
            1,
            shape=(256, 160),
        )

    def test_example_batch_softmax(self):
        from examples.batch_softmax import batch_softmax

        # examples/batch_softmax.py uses check(16, 512, 1024)
        self._check_backward(
            batch_softmax,
            lambda x: torch.nn.functional.softmax(x, dim=-1),
            1,
            shape=(4, 32, 64),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_example_geglu(self):
        from examples.geglu import geglu

        # examples/geglu.py uses 1D shape from config (typically 8192)
        self._check_backward(
            geglu,
            lambda a, b: torch.nn.functional.gelu(a, approximate="tanh") * b,
            2,
            shape=(8192,),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_example_swiglu(self):
        from examples.swiglu import swiglu_fwd

        # examples/swiglu.py uses 1D shape from config (typically 8192)
        self._check_backward(
            swiglu_fwd,
            lambda a, b: torch.nn.functional.silu(a) * b,
            2,
            shape=(8192,),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_example_matmul_layernorm(self):
        from examples.matmul_layernorm import matmul_layernorm

        # examples/matmul_layernorm.py uses check(32, 64, 200);
        # use 128 (power-of-2) for clean block_size alignment
        M, K, N = 32, 64, 128
        self._check_backward(
            matmul_layernorm,
            lambda x, y, w, b: torch.nn.functional.layer_norm(x @ y, [N], w, b),
            4,
            inputs_fn=lambda: [
                torch.randn(M, K, device=DEVICE, dtype=torch.float32),
                torch.randn(K, N, device=DEVICE, dtype=torch.float32),
                torch.randn(N, device=DEVICE, dtype=torch.float32),
                torch.randn(N, device=DEVICE, dtype=torch.float32),
            ],
            grad_shape=(M, N),
            rtol=1e-2,
            atol=1e-2,
        )

    @_force_utf8_locale
    def test_example_attention(self):
        from examples.attention import attention

        # examples/attention.py uses test(2, 32, 1024, 64, HALF_DTYPE)
        B, H, M, D = 2, 2, 64, 32
        self._check_backward(
            attention,
            lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            3,
            inputs_fn=lambda: [
                torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32),
                torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32),
                torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32),
            ],
            grad_shape=(B, H, M, D),
            rtol=1e-2,
            atol=1e-2,
        )

    @_force_utf8_locale
    def test_example_attention_non_divisible_seqlen(self):
        # Non-block-divisible sequence length: the scan zero-pads the key dim
        # and must re-apply the forward's OOB mask, else the softmax is
        # polluted. M=100 is not a multiple of typical block sizes.
        from examples.attention import attention

        B, H, M, D = 1, 2, 100, 32
        self._check_backward(
            attention,
            lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            3,
            inputs_fn=lambda: [
                torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32),
                torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32),
                torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32),
            ],
            grad_shape=(B, H, M, D),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_intermediate_buffer(self):
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            temp = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                temp[tile] = x[tile] * 2
                out[tile] = temp[tile] + 1
            return out

        self._check_backward(
            kernel,
            lambda x: x * 2 + 1,
            1,
            shape=(64, 32),
        )


if __name__ == "__main__":
    unittest.main()
