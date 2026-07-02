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
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import xfailIfPallas
import helion.language as hl
from helion.runtime.settings import _get_backend


@helion.kernel
def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    for tile0, tile1 in hl.tile(out0.size()):
        out0[tile0, tile1] = a[tile0, tile1] + b[tile0, None]
        out1[tile0, tile1] = a[tile0, tile1] + b[None, tile1]
    return out0, out1


def broadcast_fn_ref(a, b):
    out0 = a + b[:, None]
    out1 = a + b[None, :]
    return out0, out1


def _check_broadcast_fn(**config):
    args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
    code, (out0, out1) = code_and_output(broadcast_fn, args, **config)
    ref0, ref1 = broadcast_fn_ref(*args)
    torch.testing.assert_close(out0, ref0)
    torch.testing.assert_close(out1, ref1)
    return code


@onlyBackends(["triton", "cute", "pallas"])
class TestBroadcasting(RefEagerTestBase, TestCase):
    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_broadcast_no_flatten(self):
        args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
        assert not broadcast_fn.bind(args).config_spec.flatten_loops

    def test_broadcast1(self):
        _check_broadcast_fn(
            block_sizes=[16, 8],
        )

    def test_broadcast2(self):
        _check_broadcast_fn(block_size=[16, 8], loop_order=(1, 0))

    def test_broadcast3(self):
        _check_broadcast_fn(
            block_sizes=[64, 1],
        )

    def test_broadcast4(self):
        _check_broadcast_fn(
            block_sizes=[1, 64],
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_broadcast5(self):
        code = _check_broadcast_fn(
            block_sizes=[32, 32],
            indexing="block_ptr",
        )
        if _get_backend() == "triton":
            self.assertIn("tl.make_block_ptr", code)

    @skipIfTileIR("tt.make_tensor_ptr legalization not supported in pinned tileir")
    def test_broadcast6(self):
        code = _check_broadcast_fn(
            block_sizes=[128, 128],
            indexing="block_ptr",
        )
        if _get_backend() == "triton":
            self.assertIn("tl.make_block_ptr", code)

    @xfailIfPallas("constexpr scalar + None-broadcast unsupported")
    def test_constexpr_index(self):
        @helion.kernel
        def fn(a, idx1):
            out0 = torch.empty_like(a)
            out1 = torch.empty_like(a)
            out2 = torch.empty_like(a)
            idx0 = 11
            for tile0, tile1 in hl.tile(out0.size()):
                out0[tile0, tile1] = a[tile0, tile1] + a[tile0, 3, None]
                out1[tile0, tile1] = a[tile0, tile1] + a[idx0, tile1][None, :]
                out2[tile0, tile1] = a[tile0, tile1] + a[tile0, idx1, None]
            return out0, out1, out2

        args = (torch.randn(512, 512, device=DEVICE), 123)
        code, (out0, out1, out2) = code_and_output(fn, args, block_sizes=[16, 16])
        torch.testing.assert_close(out0, args[0] + args[0][:, 3, None])
        torch.testing.assert_close(out1, args[0] + args[0][11, None, :])
        torch.testing.assert_close(out2, args[0] + args[0][:, args[1], None])

    def test_implicit_broadcast(self):
        @helion.kernel
        def fn(a, b):
            out = torch.empty_like(a)
            for tile0, tile1 in hl.tile(a.size()):
                out[tile0, tile1] = a[tile0, tile1] + b[tile1]
            return out

        args = (torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE))
        code, out = code_and_output(fn, args, block_sizes=[16, 16])
        torch.testing.assert_close(out, sum(args))

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @skipIfXPU("Type promotion issue on XPU backend")
    def test_python_float_promotion(self):
        # Repro for https://github.com/pytorch/helion/issues/493
        # Python floats should follow PyTorch type promotion (no unintended fp64 upcast)
        @helion.kernel(config={"block_size": 16, "indexing": "block_ptr"})
        def fn(a, beta):
            for tile0 in hl.tile(a.shape[0]):
                b = a[tile0]
                a[tile0] = (1 - beta) * b
            return a

        a = torch.randn(1024, device=DEVICE)
        beta = 1.5
        args = (a, beta)

        # Expected behavior matches PyTorch promotion rules on tensors
        expected = (1 - beta) * a
        code, out = code_and_output(fn, args)
        torch.testing.assert_close(out, expected)

    def test_lerp_scalar_weight(self):
        # Repro for https://github.com/pytorch/helion/issues/448
        # Using torch.lerp with a Python scalar weight should not crash.
        @helion.kernel
        def fn(a, b, w):
            for tile0, tile1 in hl.tile(a.shape):
                a[tile0, tile1] = torch.lerp(a[tile0, tile1], b[tile0, tile1], w)
            return a

        a = torch.randn(128, 128, device=DEVICE)
        b = torch.randn(128, 128, device=DEVICE)
        w = 0.5
        args = (a.clone(), b, w)

        expected = torch.lerp(a, b, w)
        code, out = code_and_output(fn, args, block_sizes=[16, 16])
        torch.testing.assert_close(out, expected)

    @skipIfRefEager("ref-eager does not support non-last-dim implicit broadcast")
    def test_implicit_broadcast_first_dim(self):
        """b[tile0] in a 2D tile should broadcast as b[:, None]."""

        @helion.kernel
        def fn(a, b):
            out = torch.empty_like(a)
            for tile0, tile1 in hl.tile(a.size()):
                out[tile0, tile1] = a[tile0, tile1] + b[tile0]
            return out

        args = (torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE))
        code, out = code_and_output(fn, args, block_sizes=[16, 16])
        torch.testing.assert_close(out, args[0] + args[1][:, None])

    @skipIfRefEager("ref-eager does not support non-last-dim implicit broadcast")
    def test_implicit_broadcast_both_dims(self):
        """row_bias[tile0] + col_scale[tile1] in same expression."""

        @helion.kernel
        def fn(a, row_bias, col_scale):
            out = torch.empty_like(a)
            for tile0, tile1 in hl.tile(a.size()):
                out[tile0, tile1] = a[tile0, tile1] * col_scale[tile1] + row_bias[tile0]
            return out

        a = torch.randn(128, 256, device=DEVICE)
        row_bias = torch.randn(128, device=DEVICE)
        col_scale = torch.randn(256, device=DEVICE)
        code, out = code_and_output(fn, (a, row_bias, col_scale), block_sizes=[64, 64])
        torch.testing.assert_close(out, a * col_scale + row_bias[:, None])

    @skipIfRefEager("ref-eager does not support non-last-dim implicit broadcast")
    def test_implicit_broadcast_3d_middle_dim(self):
        """b[tile1] in 3D tile should broadcast along the middle dimension."""

        @helion.kernel
        def fn(a, b):
            out = torch.empty_like(a)
            for tile0, tile1, tile2 in hl.tile(a.size()):
                out[tile0, tile1, tile2] = a[tile0, tile1, tile2] + b[tile1]
            return out

        a = torch.randn(4, 32, 32, device=DEVICE)
        b = torch.randn(32, device=DEVICE)
        code, out = code_and_output(fn, (a, b), block_sizes=[4, 32, 32])
        torch.testing.assert_close(out, a + b[None, :, None])

    @skipIfRefEager("ref-eager does not support non-last-dim implicit broadcast")
    @xfailIfPallas("scatter: only indirect dim 0 is supported")
    def test_implicit_broadcast_mixed_tile_and_slice(self):
        """b[tile0, 0:K] broadcast into a[tile0, tile1, 0:K]."""

        @helion.kernel
        def fn(a, b):
            k = a.size(2)
            out = torch.empty_like(a)
            for tile0, tile1 in hl.tile([a.size(0), a.size(1)]):
                out[tile0, tile1, 0:k] = a[tile0, tile1, 0:k] + b[tile0, 0:k]
            return out

        a = torch.randn(8, 16, 2, device=DEVICE)
        b = torch.randn(8, 4, device=DEVICE)
        code, out = code_and_output(fn, (a, b), block_sizes=[8, 16])
        torch.testing.assert_close(out, a + b[:, 0:2].unsqueeze(1))

    @skipIfRefEager("ref-eager does not support non-last-dim implicit broadcast")
    def test_expand_as_first_dim(self):
        """scale[tile0].expand_as(a) exercises codegen_expand path."""

        @helion.kernel
        def fn(a, scale):
            out = torch.empty_like(a)
            for tile0, tile1 in hl.tile(a.size()):
                loaded = a[tile0, tile1]
                s = scale[tile0].expand_as(loaded)
                out[tile0, tile1] = loaded * s
            return out

        a = torch.randn(256, 128, device=DEVICE)
        scale = torch.randn(256, device=DEVICE)
        code, out = code_and_output(fn, (a, scale), block_sizes=[64, 64])
        torch.testing.assert_close(out, a * scale[:, None])


if __name__ == "__main__":
    unittest.main()
