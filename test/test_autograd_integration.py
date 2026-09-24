"""Test automatic autograd integration for Helion kernels.

When HELION_EXPERIMENTAL_DIFFERENTIABLE=1 is set, calling a @helion.kernel
on tensors with requires_grad=True should produce outputs that participate
in the autograd graph -- no manual torch.autograd.Function needed.
"""

from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

import torch
from torch import Tensor
import torch.nn as nn

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfMTIA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfXPU
import helion.language as hl


@helion.kernel(autotune_effort="none")
def square_plus(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        v = x[tile]
        out[tile] = v * v + v
    return out


@helion.kernel(autotune_effort="none")
def fused_mul_add(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * y[tile] + alpha
    return out


@helion.kernel(autotune_effort="none")
def norm_and_scale(x: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    m, n = x.size()
    out = torch.empty_like(x)
    inv_rms = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        out[tile_m, :] = (row * rms[:, None]).to(x.dtype)
        inv_rms[tile_m] = rms
    return out, inv_rms


_ENV_ON = {"HELION_EXPERIMENTAL_DIFFERENTIABLE": "1"}
_ENV_OFF = {"HELION_EXPERIMENTAL_DIFFERENTIABLE": "0"}


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestSingleInput(RefEagerTestDisabled, TestCase):
    """Single tensor in, single tensor out, elementwise."""

    def test_gradient_correctness_1d(self):
        # d/dx(x*x + x) = 2x + 1
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(256, device=DEVICE, requires_grad=True)
            y = square_plus(x)
            y.sum().backward()
            expected = 2 * x.detach() + 1
            torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)

    def test_gradient_correctness_2d(self):
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(16, 64, device=DEVICE, requires_grad=True)
            y = square_plus(x)
            y.sum().backward()
            expected = 2 * x.detach() + 1
            torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)

    def test_no_grad_input_no_wrapping(self):
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(128, device=DEVICE, requires_grad=False)
            y = square_plus(x)
            self.assertFalse(y.requires_grad)


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestMultiInput(RefEagerTestDisabled, TestCase):
    """Multiple tensor inputs and scalar args."""

    def test_two_inputs_both_grad(self):
        # d/dx(x*y + alpha) = y, d/dy(x*y + alpha) = x
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = torch.randn(128, device=DEVICE, requires_grad=True)
            out = fused_mul_add(x, y, 0.5)
            out.sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(y.grad)
            torch.testing.assert_close(x.grad, y.detach(), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(y.grad, x.detach(), rtol=1e-5, atol=1e-5)

    def test_one_input_grad_one_not(self):
        # d/dx(x*y + alpha) = y; y has no grad so y.grad stays None.
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = torch.randn(128, device=DEVICE, requires_grad=False)
            out = fused_mul_add(x, y, 1.0)
            out.sum().backward()
            torch.testing.assert_close(x.grad, y, rtol=1e-5, atol=1e-5)
            self.assertIsNone(y.grad)

    def test_scalar_arg_no_grad(self):
        # d/dx(x*y + alpha) = y, d/dy(x*y + alpha) = x, d/dalpha = 1 (no grad)
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = torch.randn(128, device=DEVICE, requires_grad=True)
            out = fused_mul_add(x, y, math.pi)
            out.sum().backward()
            torch.testing.assert_close(x.grad, y.detach(), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(y.grad, x.detach(), rtol=1e-5, atol=1e-5)


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestModule(RefEagerTestDisabled, TestCase):
    """Helion kernel inside nn.Module with a standard training loop."""

    def _make_models(self):
        class HelionMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 8)

            def forward(self, x):
                x = self.fc1(x)
                x = square_plus(x)
                return self.fc2(x)

        class BaselineMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 8)

            def forward(self, x):
                x = self.fc1(x)
                x = x * x + x
                return self.fc2(x)

        helion_mlp = HelionMLP().to(DEVICE)
        baseline_mlp = BaselineMLP().to(DEVICE)

        baseline_mlp.fc1.weight.data.copy_(helion_mlp.fc1.weight.data)
        baseline_mlp.fc1.bias.data.copy_(helion_mlp.fc1.bias.data)
        baseline_mlp.fc2.weight.data.copy_(helion_mlp.fc2.weight.data)
        baseline_mlp.fc2.bias.data.copy_(helion_mlp.fc2.bias.data)

        return helion_mlp, baseline_mlp

    def test_backward_matches_baseline(self):
        with patch.dict(os.environ, _ENV_ON):
            helion_mlp, baseline_mlp = self._make_models()
            x = torch.randn(4, 16, device=DEVICE)
            target = torch.randn(4, 8, device=DEVICE)

            loss_h = nn.functional.mse_loss(helion_mlp(x), target)
            loss_b = nn.functional.mse_loss(baseline_mlp(x), target)
            loss_h.backward()
            loss_b.backward()

            for name in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
                g_h = dict(helion_mlp.named_parameters())[name].grad
                g_b = dict(baseline_mlp.named_parameters())[name].grad
                self.assertIsNotNone(g_h, f"{name}.grad is None for Helion model")
                self.assertIsNotNone(g_b, f"{name}.grad is None for baseline")
                torch.testing.assert_close(g_h, g_b, rtol=1e-2, atol=1e-2)


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestMultiOutput(RefEagerTestDisabled, TestCase):
    """Kernels returning tuple[Tensor, ...]."""

    def setUp(self):
        super().setUp()
        # The backward cache in autodiff.py is not keyed on which outputs
        # receive gradients.  Without reset, whichever test runs first
        # compiles a backward that the other test can't reuse.
        norm_and_scale.reset()

    @staticmethod
    def _rms_norm_ref(x, eps):
        row = x.float()
        rms = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + eps)
        return (row * rms).to(x.dtype), rms.squeeze(-1)

    def test_backward_through_first_output(self):
        with patch.dict(os.environ, _ENV_ON):
            x_h = torch.randn(16, 64, device=DEVICE, requires_grad=True)
            x_r = x_h.detach().clone().requires_grad_(True)
            out_h, _ = norm_and_scale(x_h, 1e-5)
            out_r, _ = self._rms_norm_ref(x_r, 1e-5)
            out_h.sum().backward()
            out_r.sum().backward()
            torch.testing.assert_close(x_h.grad, x_r.grad, rtol=1e-4, atol=1e-4)

    def test_backward_through_both_outputs(self):
        with patch.dict(os.environ, _ENV_ON):
            x_h = torch.randn(16, 64, device=DEVICE, requires_grad=True)
            x_r = x_h.detach().clone().requires_grad_(True)
            out_h, inv_rms_h = norm_and_scale(x_h, 1e-5)
            out_r, inv_rms_r = self._rms_norm_ref(x_r, 1e-5)
            (out_h.sum() + inv_rms_h.sum()).backward()
            (out_r.sum() + inv_rms_r.sum()).backward()
            torch.testing.assert_close(x_h.grad, x_r.grad, rtol=1e-4, atol=1e-4)


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestGating(RefEagerTestDisabled, TestCase):
    """Env var opt-in/opt-out contract."""

    def test_default_no_autograd(self):
        # Env var completely absent -- current behavior preserved.
        with patch.dict(os.environ):
            os.environ.pop("HELION_EXPERIMENTAL_DIFFERENTIABLE", None)
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = square_plus(x)
            self.assertIsNone(y.grad_fn)

    def test_env_var_enables_autograd(self):
        with patch.dict(os.environ, _ENV_ON):
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = square_plus(x)
            self.assertIsNotNone(y.grad_fn)


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestCoexistence(RefEagerTestDisabled, TestCase):
    """Manual torch.autograd.Function wrappers still work with the env var."""

    def _make_manual_function(self):
        @helion.kernel(autotune_effort="none")
        def sq_fwd(x: Tensor) -> Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * x[tile]
            return out

        class SqFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = sq_fwd(x)
                ctx.save_for_backward(x)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * 2 * x

        return SqFunction

    def test_manual_function_env_off(self):
        with patch.dict(os.environ, _ENV_OFF):
            SqFunction = self._make_manual_function()
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = SqFunction.apply(x)
            y.sum().backward()
            expected = 2 * x.detach()
            torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)

    def test_manual_function_env_on(self):
        # When the env var is on, the kernel call inside the user's forward()
        # also routes through _HelionAutogradFunction, creating an inner
        # autograd node.  PyTorch's apply() replaces the grad_fn on the
        # output so only the user's backward() runs -- the inner node is
        # wasted work but gradients are still correct.
        with patch.dict(os.environ, _ENV_ON):
            SqFunction = self._make_manual_function()
            x = torch.randn(128, device=DEVICE, requires_grad=True)
            y = SqFunction.apply(x)
            y.sum().backward()
            expected = 2 * x.detach()
            torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)


@skipIfMTIA("autograd integration not tested on MTIA")
@skipIfNotTriton("autograd integration not tested on non-Triton backends")
@skipIfXPU("autograd integration not tested on XPU")
class TestBackwardCacheBug(RefEagerTestDisabled, TestCase):
    """The backward cache in autodiff.py is not keyed on which outputs
    receive gradients.  Switching between output subsets without a reset
    hits a stale compiled backward."""

    @unittest.expectedFailure  # backward cache not keyed on grad_output structure
    def test_mixed_output_usage_without_reset(self):
        # TODO(#420): backward cache in autodiff.py should be keyed on
        # which outputs receive gradients so callers don't need to call
        # kernel.reset() when switching between output subsets.
        with patch.dict(os.environ, _ENV_ON):
            x1 = torch.randn(16, 64, device=DEVICE, requires_grad=True)
            out1, _ = norm_and_scale(x1, 1e-5)
            out1.sum().backward()

            x2 = torch.randn(16, 64, device=DEVICE, requires_grad=True)
            out2, inv_rms2 = norm_and_scale(x2, 1e-5)
            (out2.sum() + inv_rms2.sum()).backward()


if __name__ == "__main__":
    unittest.main()
