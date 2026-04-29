from __future__ import annotations

import math
from typing import Callable
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallas
import helion.language as hl


@helion.kernel(backend="pallas", static_shapes=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.relu(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sin(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_pointwise_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(torch.sin(torch.relu(x[tile] * y[tile])))
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_affine_scalar_args(
    x: torch.Tensor,
    scale: int,
    bias: float,
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * scale + bias
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_matmul_broadcast_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], device=x.device, dtype=torch.promote_types(x.dtype, y.dtype)
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc + bias[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sum_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sum_reduce_dim0(x: torch.Tensor) -> torch.Tensor:
    _n, m = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[:, tile_m].sum(0)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sum_reduce_middle(x: torch.Tensor) -> torch.Tensor:
    b, _n, m = x.size()
    out = torch.empty([b, m], dtype=x.dtype, device=x.device)
    for tile_b, tile_m in hl.tile([b, m]):
        out[tile_b, tile_m] = x[tile_b, :, tile_m].sum(1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_sum_reduce_multiple(x: torch.Tensor) -> torch.Tensor:
    b, _n, _m = x.size()
    out = torch.empty([b], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(b):
        out[tile_b] = x[tile_b, :, :].sum([0, 1])
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_max_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_min_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amin(x[tile_n, :], dim=-1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_argmin_reduction(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=torch.int32, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.argmin(x[tile_n, :], dim=-1).to(torch.int32)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_tile_begin_end(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + tile.begin - tile.end
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inplace_add(x: torch.Tensor, y: torch.Tensor) -> None:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(out.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_arange_add(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        offsets = hl.arange(m)
        out[tile_n, :] = x[tile_n, :] + offsets[None, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inner_loop_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel with an outer grid loop and an inner device loop."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_two_pass_reduction(x: torch.Tensor) -> torch.Tensor:
    """Two inner reduction loops over the same dim: reduce to a per-row mean,
    then subtract it from each element.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = torch.zeros_like(x[tile_m, 0], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = acc + torch.sum(x[tile_m, tile_n], dim=-1)
        mean = (acc / n)[:, None]
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - mean.to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_inner_loop_add_with_scalar_access(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Kernel that mixes pipeline-tiled and scalar reads of the same tensor."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n] + x[0, 0]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kernel with an outer grid loop and a 2D inner device loop."""
    b, m, n = x.size()
    out = torch.empty_like(x)
    for tile_b in hl.tile(b):
        for tile_m, tile_n in hl.tile([m, n]):
            out[tile_b, tile_m, tile_n] = (
                x[tile_b, tile_m, tile_n] + y[tile_b, tile_m, tile_n]
            )
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_reduce_non_pow2(x: torch.Tensor) -> torch.Tensor:
    """Softmax over a non-power-of-2 reduction dim.

    Uses amax + exp + sum which forces explicit index/mask generation,
    exercising the RDIM_SIZE code path.
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row = x[tile_n, :]
        max_val = torch.amax(row, dim=-1, keepdim=True)
        exp_val = torch.exp(row - max_val)
        out[tile_n, :] = exp_val / torch.sum(exp_val, dim=-1, keepdim=True)
    return out


def _cumsum_broadcast_ref(
    a: torch.Tensor, b: torch.Tensor, block_k: int = 128
) -> torch.Tensor:
    """Eager reference for cumsum_broadcast kernels.

    running[b,m] accumulates row sums; acc[b,m,d] += running[:,:,None].
    """
    batch, m, k = a.shape
    head_dim = b.shape[-1]
    running = torch.zeros(batch, m, dtype=torch.float32, device=a.device)
    acc = torch.zeros(batch, m, head_dim, dtype=torch.float32, device=a.device)
    for kb in range(0, k, block_k):
        chunk = a[:, :, kb : kb + block_k]
        running = running + chunk.sum(-1).float()
        acc = acc + running[:, :, None]
    return acc.to(a.dtype)


def _scaled_bmm_ref(
    a: torch.Tensor, b: torch.Tensor, block_k: int = 128
) -> torch.Tensor:
    """Eager reference for scaled_bmm kernels.

    m_i[b,m] accumulates row sums; acc[b,m,d] += m_i[:,:,None].
    """
    batch, m, k = a.shape
    head_dim = b.shape[-1]
    m_i = torch.zeros(batch, m, dtype=torch.float32, device=a.device)
    acc = torch.zeros(batch, m, head_dim, dtype=torch.float32, device=a.device)
    for kb in range(0, k, block_k):
        chunk = a[:, :, kb : kb + block_k]
        m_i = m_i + chunk.sum(-1).float()
        acc = acc + m_i[:, :, None]
    return acc.to(a.dtype)


def _running_max_broadcast_ref(
    a: torch.Tensor, b: torch.Tensor, block_k: int = 128
) -> torch.Tensor:
    """Eager reference for running_max_broadcast kernel.

    scale[b,m] = running max of chunk row maxes; acc[b,m,d] += scale[:,:,None].
    """
    batch, m, k = a.shape
    head_dim = b.shape[-1]
    scale = torch.zeros(batch, m, dtype=torch.float32, device=a.device)
    acc = torch.zeros(batch, m, head_dim, dtype=torch.float32, device=a.device)
    for kb in range(0, k, block_k):
        chunk = a[:, :, kb : kb + block_k]
        scale = torch.maximum(scale, chunk.amax(-1).float())
        acc = acc + scale[:, :, None]
    return acc.to(a.dtype)


@onlyBackends(["triton", "pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallas(TestCase):
    def test_estimate_pallas_vmem_bytes(self) -> None:
        """VMEM OOM: Tests that block sizes and dtypes (fp32, bf16) are correctly estimated."""

        # Test 1: float32 (4 bytes per element)
        # 3 tensors * 2048 * 4096 * 4 bytes * 2 (multiplier) = ~201.3MB (OOM)
        args_f32 = (
            torch.randn(2048, 4096, device=DEVICE, dtype=torch.float32),
            torch.randn(2048, 4096, device=DEVICE, dtype=torch.float32),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"Ran out of memory in memory space vmem.*Estimated [0-9.]+MB exceeds",
        ):
            code_and_output(pallas_add_2d, args_f32, block_sizes=[2048, 4096])

        # Test 2: bfloat16 (2 bytes per element)
        # 3 tensors * 1024 * 4096 * 2 bytes * 2 (multiplier) = ~50.3MB (Passes safely under 64MB)
        args_bf16 = (
            torch.randn(1024, 4096, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(1024, 4096, device=DEVICE, dtype=torch.bfloat16),
        )
        try:
            code_and_output(pallas_add_2d, args_bf16, block_sizes=[1024, 4096])
        except Exception as e:
            if "Ran out of memory in memory space vmem" in str(e):
                self.fail(f"bfloat16 incorrectly threw VMEM OOM: {e}")

        # Test 3: float8_e4m3fn (1 byte per element)
        # 3 tensors * 4096 * 8192 * 1 byte * 2 (multiplier) = ~201.3MB (OOM)
        args_fp8 = (
            torch.randn(4096, 8192, device=DEVICE, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            torch.randn(4096, 8192, device=DEVICE, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"Ran out of memory in memory space vmem.*Estimated [0-9.]+MB exceeds",
        ):
            code_and_output(pallas_add_2d, args_fp8, block_sizes=[4096, 8192])

    def test_add_1d(self) -> None:
        args = (torch.randn(1024, device=DEVICE), torch.randn(1024, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=256)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_large(self) -> None:
        args = (torch.randn(4096, device=DEVICE), torch.randn(4096, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=512)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_does_not_donate_inputs(self) -> None:
        """Verify that read-only inputs are not donated by the kernel.

        Regression test: the codegen used to mark all tensor args as outputs
        (including read-only inputs rebound by broadcast_tensors), causing JAX
        to donate their buffers.  Any external reference to the inputs would
        then fail with "Buffer has been deleted or donated".
        """
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        # Save copies to compare against after the kernel call.
        x_copy = x.clone()
        y_copy = y.clone()
        code, result = code_and_output(add_kernel, (x, y), block_size=256)
        torch.testing.assert_close(result, x_copy + y_copy)
        # Only the output (index 2) should be in _output_indices, not inputs.
        self.assertIn("_output_indices=[2]", code)
        # The original inputs must still be accessible (not donated).
        torch.testing.assert_close(x, x_copy)
        torch.testing.assert_close(y, y_copy)

    def test_add_2d(self) -> None:
        args = (
            torch.randn(64, 512, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 512, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(pallas_add_2d, args, block_sizes=[8, 512])
        torch.testing.assert_close(result, args[0] + args[1])

    def test_arange(self) -> None:
        x = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
        offsets = torch.arange(64, device=DEVICE, dtype=torch.int32).float()
        code, result = code_and_output(pallas_arange_add, (x,), block_size=8)
        torch.testing.assert_close(result, x + offsets[None, :])
        self.assertIn("jnp.arange", code)

    def test_inplace_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        expected = x + y
        # Use block_size=1024 so grid=1; with grid>1 the full-array
        # access pattern causes inplace mutations to accumulate.
        code, result = code_and_output(pallas_inplace_add, (x, y), block_size=1024)
        # x should be mutated in place
        torch.testing.assert_close(x, expected)

    def test_pointwise_mul(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(pallas_mul, args, block_size=256)
        x, y = args
        torch.testing.assert_close(out, x * y)

    def test_pointwise_relu(self) -> None:
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_relu, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.relu(x))

    def test_pointwise_sin(self) -> None:
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_sin, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.sin(x))

    def test_pointwise_sigmoid(self) -> None:
        # float16 is not supported by TPU Pallas Mosaic lowering
        # ("Not implemented: offset not aligned to sublanes")
        args = (torch.randn(1024, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(pallas_sigmoid, args, block_size=256)
        (x,) = args
        torch.testing.assert_close(out, torch.sigmoid(x), rtol=1e-5, atol=1e-5)

    def test_pointwise_chain(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(pallas_pointwise_chain, args, block_size=256)
        x, y = args
        expected = torch.sigmoid(torch.sin(torch.relu(x * y)))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_args(self) -> None:
        args = (
            torch.randn(1024, device=DEVICE, dtype=torch.float32),
            3,
            1.25,
        )
        code, out = code_and_output(pallas_affine_scalar_args, args, block_size=256)
        x, scale, bias = args
        torch.testing.assert_close(out, x * scale + bias, rtol=1e-5, atol=1e-5)

    def test_sum_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_sum_reduction, (x,), block_size=16)
        self.assertIn("jnp.sum", code)
        torch.testing.assert_close(result, x.sum(-1), rtol=1e-4, atol=1e-4)

    def test_sum_reduction_large(self) -> None:
        x = torch.randn(8, 16384, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_sum_reduction, (x,), block_size=1)
        self.assertIn("jnp.sum", code)
        torch.testing.assert_close(result, x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_reduce_dim0(self) -> None:
        x = torch.randn(64, 32, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_sum_reduce_dim0, (x,), block_size=16)
        self.assertIn("jnp.sum", code)
        torch.testing.assert_close(result, x.sum(0), rtol=1e-4, atol=1e-4)

    def test_sum_reduce_middle(self) -> None:
        x = torch.randn(4, 64, 32, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_sum_reduce_middle, (x,), block_sizes=[2, 16]
        )
        self.assertIn("jnp.sum", code)
        torch.testing.assert_close(result, x.sum(1), rtol=1e-4, atol=1e-4)

    def test_sum_reduce_multiple(self) -> None:
        x = torch.randn(4, 32, 64, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(NotImplementedError):
            code_and_output(pallas_sum_reduce_multiple, (x,), block_size=2)

    def test_max_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_max_reduction, (x,), block_size=16)
        self.assertIn("jnp.max", code)
        torch.testing.assert_close(result, torch.amax(x, dim=-1), rtol=1e-4, atol=1e-4)

    def test_min_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_min_reduction, (x,), block_size=16)
        self.assertIn("jnp.min", code)
        torch.testing.assert_close(result, torch.amin(x, dim=-1), rtol=1e-4, atol=1e-4)

    def test_argmin_reduction(self) -> None:
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_argmin_reduction, (x,), block_size=16)
        self.assertIn("jnp.argmin", code)
        torch.testing.assert_close(result, torch.argmin(x, dim=-1).to(torch.int32))

    def test_tile_begin_end(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        from helion.runtime.config import Config

        bound = pallas_tile_begin_end.bind((x,))
        code = bound.to_code(Config(block_size=256))
        self.assertIn("pl.program_id", code)

    def test_dynamic_scalar_no_recompile(self) -> None:
        """Verify that changing dynamic scalar values does not trigger recompilation."""
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        pallas_affine_scalar_args.reset()

        # First call - triggers compilation
        result1 = pallas_affine_scalar_args(x, 3, 1.25)
        self.assertEqual(len(pallas_affine_scalar_args._bound_kernels), 1)

        # Second call with different scalar values - should NOT recompile
        result2 = pallas_affine_scalar_args(x, 5, 2.5)
        self.assertEqual(len(pallas_affine_scalar_args._bound_kernels), 1)

        # Verify correctness
        torch.testing.assert_close(result1, x * 3 + 1.25, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result2, x * 5 + 2.5, rtol=1e-5, atol=1e-5)

    def test_inner_loop_add(self) -> None:
        """Test kernel with outer grid loop and inner device loop."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add, args, block_sizes=[8, 128]
        )
        self.assertIn("for ", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_matmul_broadcast_bias(self) -> None:
        """Regression: bias [1, N] must not iterate grid dim 0.

        Without the dim_size <= block_size guard in _compute_block_spec_info,
        the bias BlockSpec maps grid dim i to its 1-row axis, causing an
        out-of-bounds DMA read that crashes the TPU.
        """
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_broadcast_bias, (x, y, bias), block_sizes=[64, 128, 128]
        )
        expected = (x.float() @ y.float() + bias.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # The bias block_spec_info must have None for dim 0 (not a grid index).
        self.assertIn("(None, 1)", code)

    def test_bmm(self) -> None:
        """Test BMM with default config — exercises size_matches fix.

        Without the size_matches fix, adjust_block_size_constraints cannot
        match block dims to tensor dims (4 block dims vs 3D tensors), causing
        the default config to pick block sizes that violate TPU alignment.
        """
        a = torch.randn(4, 128, 256, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4, 256, 128, device=DEVICE, dtype=torch.bfloat16)
        # No explicit block_sizes — uses default_config() which runs
        # adjust_block_size_constraints and depends on size_matches.
        code, result = code_and_output(pallas_bmm, (a, b))
        expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_bmm_fori_loop_non_divisible_k(self) -> None:
        """Test fori_loop bmm where BLOCK_K=256 doesn't evenly divide K=384."""
        a = torch.randn(4, 128, 384, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4, 384, 128, device=DEVICE, dtype=torch.bfloat16)
        _code, result = code_and_output(
            pallas_bmm,
            (a, b),
            block_sizes=[4, 128, 128, 256],
            pallas_loop_type="fori_loop",
        )
        expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_bmm_emit_pipeline_non_divisible_k(self) -> None:
        """Test emit_pipeline bmm where BLOCK_K=256 doesn't evenly divide K=384."""
        a = torch.randn(4, 128, 384, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4, 384, 128, device=DEVICE, dtype=torch.bfloat16)
        _code, result = code_and_output(
            pallas_bmm,
            (a, b),
            block_sizes=[4, 128, 128, 256],
            pallas_loop_type="emit_pipeline",
        )
        expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_emit_pipeline_codegen(self) -> None:
        """Test that pallas_loop_type='emit_pipeline' generates correct emit_pipeline code."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("pl.BlockSpec", code)
        torch.testing.assert_close(result, args[0] + args[1])
        # out is output-only, excluded from pallas_call inputs
        self.assertIn("_inplace_indices=[]", code)

    def test_fori_loop_codegen(self) -> None:
        """Test that pallas_loop_type='fori_loop' generates correct fori_loop code."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn("pltpu.make_async_copy", code)
        self.assertNotIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(result, args[0] + args[1])
        # out is output-only, excluded from pallas_call inputs
        self.assertIn("_inplace_indices=[]", code)

    def test_two_pass_reduction_emit_pipeline(self) -> None:
        """Two inner reduction loops over the same dim compile and run under
        ``pallas_loop_type='emit_pipeline'``.
        """
        x = torch.randn(256, 128, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            pallas_two_pass_reduction,
            (x,),
            block_sizes=[128, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        expected = x - x.mean(dim=-1, keepdim=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_two_pass_reduction_fori_loop(self) -> None:
        """Two inner reduction loops over the same dim compile and run under
        ``pallas_loop_type='fori_loop'``.
        """
        x = torch.randn(256, 128, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            pallas_two_pass_reduction,
            (x,),
            block_sizes=[128, 128, 128],
            pallas_loop_type="fori_loop",
        )
        expected = x - x.mean(dim=-1, keepdim=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @xfailIfPallas("Pipeline + scalar access codegen not yet supported")
    def test_pipeline_tensor_with_scalar_access(self) -> None:
        """A pipeline tensor with scalar access should keep HBM, not be overridden to SMEM."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        expected = args[0] + args[1] + args[0][0, 0]
        code, result = code_and_output(
            pallas_inner_loop_add_with_scalar_access,
            args,
            block_sizes=[8, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("_pipeline_arg_indices=", code)
        torch.testing.assert_close(result, expected)

    def test_invalid_pallas_loop_type_raises(self) -> None:
        """Invalid pallas_loop_type values must raise instead of silently falling back."""
        args = (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "Invalid pallas_loop_type 'pipeline'"):
            code_and_output(
                pallas_inner_loop_add,
                args,
                block_sizes=[8, 128],
                pallas_loop_type="pipeline",
            )

    def test_attention_default_fp32(self) -> None:
        """Test attention with default (for-loop) inner loop."""
        query = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        key = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        val = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        args = (query, key, val)

        _code, result = code_and_output(pallas_attention, args, block_sizes=[1, 32, 32])
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

        # test that we're not manually allocating and donating out tensor HBM,
        # but are instead taking over tensor returned by torch_tpu JaxCallable
        self.assertIn("out = torch.empty_like(q_view, device='meta')", _code)
        self.assertIn("out = _launcher(", _code)

    def test_attention_emit_pipeline_correctness(self) -> None:
        """Test emit_pipeline attention with loop-carried state and pre-broadcast."""
        query = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        key = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        val = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        code, result = code_and_output(
            pallas_attention,
            (query, key, val),
            block_sizes=[4, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        # m_i and l_i last dim 128 is the pre-broadcast trailing dim;
        # acc last dim 128 is head_dim (unchanged)
        self.assertIn(
            "_scratch_shapes=["
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_attention_fori_loop_correctness(self) -> None:
        """Test fori_loop attention with loop-carried state and pre-broadcast."""
        query = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        key = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        val = torch.randn(2, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        args = (query, key, val)
        code, result = code_and_output(
            pallas_attention,
            args,
            block_sizes=[4, 128, 128],
            pallas_loop_type="fori_loop",
            pallas_pre_broadcast=True,
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn("pltpu.make_async_copy", code)
        # m_i and l_i last dim 128 is the pre-broadcast trailing dim;
        # acc last dim 128 is head_dim; extra entries are DMA buffers/semaphores
        self.assertIn(
            "_scratch_shapes=["
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((), None, 'dma_semaphore'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((), None, 'dma_semaphore')]",
            code,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_attention_emit_pipeline_non_divisible(self) -> None:
        """Test emit_pipeline with seq_kv not divisible by block_k.

        Uses _explicit_indices to pass iteration index into body for
        proper mask computation on partial tiles.  Pre-broadcast still
        applies since block_k=256 is a multiple of 128.
        """
        # seq=384, block_k=256 -> 2 tiles, last is partial (128/256)
        query = torch.randn(1, 2, 128, 128, dtype=torch.float32, device=DEVICE)
        key = torch.randn(1, 2, 384, 128, dtype=torch.float32, device=DEVICE)
        val = torch.randn(1, 2, 384, 128, dtype=torch.float32, device=DEVICE)
        code, result = code_and_output(
            pallas_attention,
            (query, key, val),
            block_sizes=[2, 128, 256],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertIn("_explicit_indices=True", code)
        # m_i and l_i last dim 128 is the pre-broadcast trailing dim;
        # acc last dim 128 is head_dim (unchanged)
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_emit_pipeline_loop_order(self) -> None:
        """Test emit_pipeline with loop_order reordering.

        Without the fix, program_id mapping uses logical grid_block_ids
        order instead of pid_info order (which reflects loop_order),
        producing wrong results.
        """
        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_broadcast_bias,
            (x, y, bias),
            block_sizes=[16, 128, 64],
            loop_orders=[[1, 0]],
            pallas_loop_type="emit_pipeline",
        )
        expected = (x.float() @ y.float() + bias.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_reduce_non_pow2(self) -> None:
        """Reduction over non-power-of-2 dim should use exact size, not rounded."""
        x = torch.randn(128, 1000, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_reduce_non_pow2, (x,), block_size=128)
        expected = torch.nn.functional.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_scalar_access_1D_constexpr(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.zeros_like(x)
            for _ in hl.tile(n, block_size=4):
                out[0] = x[0]
                out[1] = x[1]
                out[2] = x[2]
                out[3] = x[3]
            return out

        x = torch.tensor([1, 2, 3, 4], device=DEVICE, dtype=torch.float32)
        result = fn(x)
        torch.testing.assert_close(result, x)

    def test_scalar_access_2D_constexpr(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.zeros_like(x)
            for _ in hl.tile([n, m], block_size=[128, 128]):
                out[42, 79] = x[42, 79]
            return out

        x = torch.ones((128, 128), device=DEVICE, dtype=torch.float32)
        result = fn(x)
        expected = torch.zeros((128, 128), device=DEVICE, dtype=torch.float32)
        expected[42, 79] = x[42, 79]
        torch.testing.assert_close(result, expected)

    def test_scalar_index_transpose(self) -> None:
        """Scalar .begin index should collapse the dimension.

        When .begin is used as a scalar subscript, the indexed
        dimension should be eliminated from the result so that
        .T produces a correct 2D permutation.
        """

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(block_sizes=[32, 32, 1]),
        )
        def scalar_index_transpose(x: torch.Tensor) -> torch.Tensor:
            B, M, N = x.shape
            out = torch.empty([B, N, M], dtype=x.dtype, device=x.device)
            for tile_m, tile_n, tile_b in hl.tile([M, N, B]):
                # tile_b has block_size=1, so .begin is used as a scalar index
                out[tile_b.begin, tile_n, tile_m] = x[tile_b.begin, tile_m, tile_n].T
            return out

        x = torch.randn(4, 64, 64, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(scalar_index_transpose, (x,))
        expected = x.permute(0, 2, 1)
        torch.testing.assert_close(result, expected)

    def test_tile_index_with_symbolic_offset(self) -> None:
        """tile.index + tile.begin * constant should codegen valid variable names.

        The offset in TileIndexWithOffsetPattern can be a sympy expression
        (e.g. tile_chunk.begin * chunk_size). The codegen must use literal_expr()
        to translate sympy symbols to their codegen variable names, otherwise
        the generated code contains undefined variables like 'u8'.

        Pattern from mamba2_chunk_state: iterates over chunks of rows, and
        within each chunk uses tile_k.index + tile_chunk.begin * chunk_size
        to compute the global row index.
        """

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
        )
        def chunked_add(x: torch.Tensor) -> torch.Tensor:
            nrows, ncols = x.shape
            chunk_size = 64
            nchunks = nrows // chunk_size
            out = torch.empty_like(x)
            for tile_col, tile_chunk in hl.tile([ncols, nchunks], block_size=[None, 1]):
                for tile_k in hl.tile(chunk_size, block_size=64):
                    # global_row = local_row_within_chunk + chunk_start
                    row = tile_k.index + tile_chunk.begin * chunk_size
                    out[row, tile_col] = x[row, tile_col] + 1.0
            return out

        # 4 chunks of 64 rows, 128 columns
        x = torch.randn(256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(chunked_add, (x,), block_sizes=[128])
        expected = x + 1.0
        torch.testing.assert_close(result, expected)

    def test_mixed_scalar_and_slice_access(self) -> None:
        """Tensor accessed both as scalar and slice should not be placed in SMEM.

        When a tensor has one access that is all-scalar (e.g. x[i, j, k])
        and another that uses a slice (e.g. x[i, j, tile]), placing it in
        SMEM causes 'Can only load scalars from SMEM' at runtime. The tensor
        must stay in VMEM to support both access patterns.
        """

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
        )
        def mixed_access(x: torch.Tensor) -> torch.Tensor:
            B, N = x.shape
            out = torch.empty_like(x)
            for tile_b, tile_n in hl.tile([B, N], block_size=[1, None]):
                # scalar access: x[tile_b.begin, N-1]
                last_val = x[tile_b.begin, N - 1]
                # slice access: x[tile_b.begin, tile_n]
                out[tile_b.begin, tile_n] = x[tile_b.begin, tile_n] + last_val
            return out

        x = torch.randn(4, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(mixed_access, (x,), block_sizes=[128])
        # x has mixed access (scalar + slice), so it must stay in VMEM
        self.assertNotIn("_smem_arg_indices", code)
        expected = x + x[:, -1:]
        torch.testing.assert_close(result, expected)

    @xfailIfPallas(
        "Mixed scalar write + slice needs tensor duplication into SMEM and VMEM"
    )
    def test_mixed_scalar_write_and_slice_access(self) -> None:
        """Tensor with both scalar write and slice access is unsupported.

        SMEM only supports scalar access; VMEM doesn't support scalar writes.
        A tensor that needs both would require duplication into SMEM (for the
        scalar write) and VMEM (for the slice access), which is not yet
        implemented.
        """

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
        )
        def mixed_write(x: torch.Tensor) -> torch.Tensor:
            B, N = x.shape
            out = torch.empty_like(x)
            for tile_b, tile_n in hl.tile([B, N], block_size=[1, None]):
                # slice read
                out[tile_b.begin, tile_n] = x[tile_b.begin, tile_n]
                # scalar write to same tensor
                out[tile_b.begin, N - 1] = x[tile_b.begin, 0]
            return out

        x = torch.randn(4, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(mixed_write, (x,), block_sizes=[128])
        expected = x.clone()
        expected[:, -1] = x[:, 0]
        torch.testing.assert_close(result, expected)

    def test_scalar_access_hl_grid(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.zeros_like(x)
            for i in hl.grid(n):
                out[i] = x[i] + 0.5
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        result = fn(x)
        expected = x + 0.5
        torch.testing.assert_close(result, expected)

    def test_scalar_access_hl_grid_offset(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty(n // 2, device=DEVICE, dtype=torch.float32)
            for i in hl.grid(n // 2):
                out[i] = x[i + n // 2] + 0.5
            return out

        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        result = fn(x)
        expected = x[x.shape[0] // 2 :] + 0.5
        torch.testing.assert_close(result, expected)

    def test_scalar_access_hl_grid_2d(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n, m) = x.size()
            out = torch.zeros_like(x)
            for i, j in hl.grid([n, m]):
                out[i, j] = x[i, j] + 0.5
            return out

        x = torch.randn((128, 128), device=DEVICE, dtype=torch.float32)
        expected = x + 0.5

        _, result = code_and_output(fn, (x,), loop_order=[0, 1])
        torch.testing.assert_close(result, expected)

        _, result = code_and_output(fn, (x,), loop_order=[1, 0])
        torch.testing.assert_close(result, expected)

    def test_scalar_access_hl_grid_2d_nested(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n, m) = x.size()
            out = torch.zeros_like(x)
            for i in hl.grid(n):
                for j in hl.grid(m):
                    out[i, j] = x[i, j] + 0.5
            return out

        x = torch.randn((128, 128), device=DEVICE, dtype=torch.float32)
        result = fn(x)
        expected = x + 0.5
        torch.testing.assert_close(result, expected)

    @xfailIfPallas("Pallas backend not correctly handling tile index with offset")
    def test_tensor_access_tile_index_offset(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.zeros(n, device=DEVICE, dtype=torch.float32)
            for tile in hl.tile(n // 2):
                out[tile] = x[tile]
                out[tile.index + n // 2] = y[tile.index + n // 2]
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, device=DEVICE, dtype=torch.float32)
        result = fn(x, y)
        torch.testing.assert_close(result, torch.concat((x[:64], y[64:])))

    @xfailIfPallas("Pallas backend not correctly handling tile index with offset")
    def test_tensor_access_tile_index_offset_2d(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            (n, m) = x.size()
            out = torch.zeros(x.size(), device=DEVICE, dtype=torch.float32)
            for tile1, tile2 in hl.tile([n // 2, m // 2]):
                out[tile1, tile2] = x[tile1, tile2]
                out[tile1.index + n // 2, tile2] = y[tile1.index + n // 2, tile2]
                out[tile1, tile2 + m // 2] = x[tile1, tile2 + m // 2]
                out[tile1.index + n // 2, tile2 + m // 2] = y[
                    tile1.index + n // 2, tile2 + m // 2
                ]
            return out

        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(fn, (x, y), block_size=[128, 128])
        torch.testing.assert_close(result, torch.concat((x[:64, :], y[64:, :])))

    def test_tensor_access_tile_id(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(x.shape[0] // 2, device=DEVICE, dtype=torch.float32)
            for t in hl.tile(x.shape[0], block_size=2):
                out[t.id] = x[t.id]
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        result = fn(x)
        torch.testing.assert_close(result, x[: x.shape[0] // 2])

    def test_tensor_access_tile_begin_end(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(x.shape[0], device=DEVICE, dtype=torch.float32)
            for t in hl.tile(x.shape[0], block_size=2):
                out[t.begin] = x[t.id]
                out[t.end - 1] = x[t.id]
            return out

        x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=DEVICE, dtype=torch.float32)
        result = fn(x)
        expected = torch.tensor(
            [0, 0, 1, 1, 2, 2, 3, 3], device=DEVICE, dtype=torch.float32
        )
        torch.testing.assert_close(result, expected)

    def test_output_only_not_inplace(self) -> None:
        """Output-only tensors should not appear in _inplace_indices.

        When _output_indices has more entries than _inplace_indices, the
        extra outputs are excluded from pallas_call inputs and
        input_output_aliases, eliminating the OpSplitMode::kSplitBoth
        graph split in torch_tpu.
        """
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_relu, (x,), block_sizes=[1024])
        torch.testing.assert_close(result, torch.relu(x))
        # out is in _output_indices but not _inplace_indices, so it's
        # excluded from pallas_call inputs (no donation, no graph split).
        self.assertIn("_output_indices=[1]", code)
        self.assertIn("_inplace_indices=[]", code)
        # Output-only allocation retargeted to device='meta' (no real HBM).
        self.assertIn("device='meta'", code)
        # Launcher return captured into output variable.
        self.assertIn("out = _launcher(", code)

    def test_new_empty_output_only(self) -> None:
        """new_empty allocations should also be recognized as output-only."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def new_empty_relu(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty(x.shape)
            for tile in hl.tile(out.size()):
                out[tile] = torch.relu(x[tile])
            return out

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(new_empty_relu, (x,), block_sizes=[1024])
        torch.testing.assert_close(result, torch.relu(x))
        self.assertIn("_inplace_indices=[]", code)
        self.assertIn("device='meta'", code)
        self.assertIn("out = _launcher(", code)

    def test_mixed_inplace_and_output_only(self) -> None:
        """Kernel with both an inplace-mutated input and an output-only tensor.

        Verifies that _inplace_indices contains only the inplace-mutated
        input (index 0), not the output-only tensor.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def inplace_and_output(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1.0
                out[tile] = x[tile] * 2.0
            return out

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        expected_out = (x + 1.0) * 2.0
        code, result = code_and_output(inplace_and_output, (x,), block_sizes=[1024])
        torch.testing.assert_close(result, expected_out)
        # 2 outputs (x and out), but only x is aliased (inplace).
        # out is excluded from pallas_call inputs.
        self.assertIn("_output_indices=[0, 1]", code)
        self.assertIn("_inplace_indices=[0]", code)
        self.assertIn("device='meta'", code)
        self.assertIn("out = _launcher(", code)

    def test_empty_like_read_stays_inplace(self) -> None:
        """An empty_like output that is also read should stay in _inplace_indices."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def read_write_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile]
                out[tile] = out[tile] + 1.0
            return out

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(read_write_kernel, (x,), block_sizes=[1024])
        torch.testing.assert_close(result, x + 1.0)
        # out is read after write, so it must be in _inplace_indices
        self.assertIn("_inplace_indices=[1]", code)
        # Not output-only, so no device='meta' retargeting.
        self.assertNotIn("device='meta'", code)

    def test_int64_tensor_raises(self) -> None:
        """Passing int64 tensors to a Pallas kernel should raise TypeError."""
        x = torch.arange(256, device=DEVICE, dtype=torch.int64)
        y = torch.arange(256, device=DEVICE, dtype=torch.int64)
        with self.assertRaises(TypeError, msg="does not support"):
            code_and_output(add_kernel, (x, y), block_size=128)

    def test_multiple_output_only(self) -> None:
        """Kernel returning two output-only tensors."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def two_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out1 = torch.empty_like(x)
            out2 = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out1[tile] = x[tile] + 1.0
                out2[tile] = x[tile] * 2.0
            return out1, out2

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, (result1, result2) = code_and_output(
            two_outputs, (x,), block_sizes=[1024]
        )
        torch.testing.assert_close(result1, x + 1.0)
        torch.testing.assert_close(result2, x * 2.0)
        # Both outputs are output-only: 2 outputs, 0 aliases
        self.assertIn("_output_indices=[1, 2]", code)
        self.assertIn("_inplace_indices=[]", code)
        self.assertIn("device='meta'", code)
        self.assertIn("out1, out2 = _launcher(", code)

    def test_fori_loop_multidim(self) -> None:
        """Test fori_loop with a 2D inner loop (nested iteration)."""
        args = (
            torch.randn(4, 64, 128, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 64, 128, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_add_3d,
            args,
            block_sizes=[1, 8, 128],
            pallas_loop_type="fori_loop",
        )
        self.assertGreaterEqual(code.count("jax.lax.fori_loop"), 2)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_default_loop_multidim_non_divisible(self) -> None:
        """Default loop with 2D inner loop where both dims are non-divisible.

        Regression test: when an output tensor is padded on multiple dims,
        _pallas_apply_ds_padding must save the original tensor reference
        only once (on the first dim), not overwrite it with the partially-
        padded tensor on subsequent dims.
        """
        args = (
            torch.randn(4, 70, 130, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 70, 130, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_add_3d,
            args,
            block_sizes=[1, 8, 128],
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_fori_loop_multidim_partial_tile(self) -> None:
        """Test fori_loop with a 2D inner loop and a partial tail tile."""
        args = (
            torch.randn(4, 70, 130, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 70, 130, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_add_3d,
            args,
            block_sizes=[1, 8, 128],
            pallas_loop_type="fori_loop",
        )
        self.assertGreaterEqual(code.count("jax.lax.fori_loop"), 2)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_fori_loop_no_dma_unaligned_inner_block(self) -> None:
        """fori_loop with inner block violating DMA alignment (last dim % 128 != 0).

        Exercises the non-DMA fallback: instead of pltpu.make_async_copy,
        codegen should emit pl.ds() slicing into the outer BlockSpec refs.
        """
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_inner_loop_add,
            args,
            block_sizes=[8, 64],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertNotIn("pltpu.make_async_copy", code)
        self.assertIn("pl.ds(", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_fori_loop_no_dma_multidim_unaligned(self) -> None:
        """Nested fori_loop with a DMA-unaligned inner block.

        2D inner loop where both inner dims are too small for DMA
        (last dim = 64 < 128).  Validates that the non-DMA pl.ds()
        path works with nested fori_loops, one per inner dim.
        """
        args = (
            torch.randn(4, 32, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 32, 64, device=DEVICE, dtype=torch.float32),
        )
        code, result = code_and_output(
            pallas_add_3d,
            args,
            block_sizes=[1, 8, 64],
            pallas_loop_type="fori_loop",
        )
        self.assertGreaterEqual(code.count("jax.lax.fori_loop"), 2)
        self.assertNotIn("pltpu.make_async_copy", code)
        self.assertIn("pl.ds(", code)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_tile_id_per_block_accumulator(self) -> None:
        """Writing to ``out[tile.id, :]`` stores one row per outer grid iter.

        This is the multi-block partial-reduction pattern used e.g. in
        ``rms_norm_bwd``: each outer grid iter computes a per-block
        accumulator and writes it into its own row of a ``[num_blocks, N]``
        output tensor, which the host then sums across ``dim=0``.

        Each grid iter ``i`` must land in row ``i``, so the kernel must
        correctly interpret the scalar ``tile.id`` index against a tensor
        whose outer dim has extent ``num_blocks`` (not ``M``).
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def per_block_reduction(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            m_block = hl.register_block_size(x.size(0))
            out = x.new_empty(
                [(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32
            )
            for mb_cta in hl.tile(m, block_size=m_block):
                acc = x.new_zeros([n], dtype=torch.float32)
                for mb in hl.tile(mb_cta.begin, mb_cta.end):
                    acc += x[mb, :].to(torch.float32).sum(0)
                out[mb_cta.id, :] = acc
            return out

        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            per_block_reduction,
            (x,),
            block_sizes=[8, 8],
            pallas_loop_type="fori_loop",
        )
        ref = x.view(8, 8, 128).sum(1)
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

    def test_squeeze_slice_access(self) -> None:
        """Test for the [None, :] indexing pattern (subscript index for slice >= tensor_ndim)"""

        @helion.kernel(backend="pallas", static_shapes=True)
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            (N,) = x.shape
            (M,) = y.shape
            out = torch.empty((N, M), dtype=x.dtype)
            for tile in hl.tile([N], block_size=[M]):
                out[tile, :] = (x[tile][:, None] < y[None, :]).to(torch.float32)
            return out

        N = 1024
        M = 128
        x = torch.randn(N, device=DEVICE, dtype=torch.float32)
        y = torch.randn(M, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(fn, (x, y))
        expected = (x[:, None] < y[None, :]).to(torch.float32)
        torch.testing.assert_close(result, expected)

    def test_matmul_1d_bias_closure(self) -> None:
        """Verifies that ops in a closure also constrain the chosen block size."""

        @helion.kernel(backend="pallas")
        def matmul_custom(
            x: torch.Tensor, y: torch.Tensor, epilogue: Callable
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
            return out

        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)

        code, result = code_and_output(
            matmul_custom, (x, y, lambda acc, tile: acc + bias[tile[1]])
        )

        expected = x.float() @ y.float() + bias.float()
        torch.testing.assert_close(
            result, expected.to(torch.bfloat16), rtol=1e-2, atol=1e-2
        )

    def test_pre_broadcast_emit_pipeline_codegen(self) -> None:
        """Pre-broadcast with emit_pipeline: scratch shapes get extra trailing dim."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def cumsum_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            head_dim = hl.specialize(b.size(-1))
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                running = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    running = running + torch.sum(chunk, -1)
                    acc = acc + running[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            cumsum_broadcast,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = _cumsum_broadcast_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_fori_loop_codegen(self) -> None:
        """Pre-broadcast with fori_loop: same transform applies."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def cumsum_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            head_dim = hl.specialize(b.size(-1))
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                running = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    running = running + torch.sum(chunk, -1)
                    acc = acc + running[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            cumsum_broadcast,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="fori_loop",
            pallas_pre_broadcast=True,
        )
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((), None, 'dma_semaphore')]",
            code,
        )
        ref = _cumsum_broadcast_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_skipped_non_multiple_of_128(self) -> None:
        """Pre-broadcast is skipped when broadcast dim is not a multiple of 128.

        Uses head_dim=64 so the broadcast target has last dim 64.
        Since 64 % 128 != 0, the transform is skipped.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def cumsum_broadcast_d64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            head_dim = hl.specialize(b.size(-1))
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                running = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    running = running + torch.sum(chunk, -1)
                    acc = acc + running[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            cumsum_broadcast_d64,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertNotIn("jnp.tile(", code)
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 64), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = _cumsum_broadcast_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_no_broadcast_no_transform(self) -> None:
        """Pre-broadcast is a no-op when loop-carried state has no broadcast usage."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def accum_sum(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.empty([n], device=x.device, dtype=x.dtype)
            for tile_n in hl.tile(n):
                acc = hl.zeros([tile_n], dtype=torch.float32)
                for tile_m in hl.tile(m):
                    acc = acc + torch.sum(x[tile_n, tile_m], -1)
                out[tile_n] = acc.to(out.dtype)
            return out

        x = torch.randn(128, 256, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            accum_sum,
            (x,),
            block_sizes=[128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertNotIn("jnp.tile(", code)
        self.assertIn(
            "_scratch_shapes=[((128,), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = x.sum(-1)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_correctness_emit_pipeline(self) -> None:
        """Pre-broadcast correctness with emit_pipeline using a bespoke kernel."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def scaled_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            _, _, n = b.size()
            head_dim = hl.specialize(n)
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                m_i = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    row_sum = torch.sum(chunk, -1)
                    m_i = m_i + row_sum
                    acc = acc + m_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            scaled_bmm,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = _scaled_bmm_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_correctness_fori_loop(self) -> None:
        """Pre-broadcast correctness with fori_loop using a bespoke kernel."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def scaled_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            _, _, n = b.size()
            head_dim = hl.specialize(n)
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                m_i = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    row_sum = torch.sum(chunk, -1)
                    m_i = m_i + row_sum
                    acc = acc + m_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            scaled_bmm,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="fori_loop",
            pallas_pre_broadcast=True,
        )
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((), None, 'dma_semaphore')]",
            code,
        )
        ref = _scaled_bmm_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_reduction_unsqueeze(self) -> None:
        """Pre-broadcast inserts unsqueeze for reduction results feeding pre-broadcast ops.

        The inner-loop reduction torch.amax(chunk, -1) produces a 2D result
        that feeds into torch.maximum(scale, ...) where scale is a pre-broadcast
        node (3D after transform).  Step 4 of _annotate_pre_broadcast must
        unsqueeze the reduction result to [..., 1] so JAX broadcast works.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def running_max_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            head_dim = hl.specialize(b.size(-1))
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                scale = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    row_max = torch.amax(chunk, -1)
                    scale = torch.maximum(scale, row_max)
                    acc = acc + scale[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            running_max_broadcast,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem')]",
            code,
        )
        self.assertIn("unsqueeze_default = row_max[:, :, None]", code)
        ref = _running_max_broadcast_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_dynamic_shapes(self) -> None:
        """Pre-broadcast with static_shapes=False exercises the SymInt codegen path.

        When head_dim is not specialized, the inner FX graph carries it as a
        backed SymInt.  The _pre_broadcast_tile codegen must handle SymInt
        target_size and emit a valid tile expression.
        """

        @helion.kernel(backend="pallas", static_shapes=False)
        def cumsum_broadcast_dynamic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            head_dim = b.size(-1)
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                running = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    running = running + torch.sum(chunk, -1)
                    acc = acc + running[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out

        # head_dim=256 > PRE_BROADCAST_SIZE=128 and a multiple of it
        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 256, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            cumsum_broadcast_dynamic,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 256), 'jnp.float32', 'vmem')]",
            code,
        )
        ref = _cumsum_broadcast_ref(a, b, block_k=128)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_double_outer_use(self) -> None:
        """Pre-broadcast value used twice via [:, :, None] in the outer scope.

        Regression test: the outer rewrite must not append PRE_BROADCAST_SIZE
        to the same base node twice when it has multiple subscript users.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def double_use(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            batch, m, k = a.size()
            head_dim = hl.specialize(b.size(-1))
            out = torch.empty([batch, m, head_dim], device=a.device, dtype=a.dtype)
            for tile_b, tile_m in hl.tile([batch, m]):
                running = hl.zeros([tile_b, tile_m], dtype=torch.float32)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    chunk = a[tile_b, tile_m, tile_k]
                    running = running + torch.sum(chunk, -1)
                    acc = acc + running[:, :, None]
                result = acc + running[:, :, None] * running[:, :, None]
                out[tile_b, tile_m, :] = result.to(out.dtype)
            return out

        a = torch.randn(2, 128, 256, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            double_use,
            (a, b),
            block_sizes=[2, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn(
            "_scratch_shapes=["
            "((2, 128, 128), 'jnp.float32', 'vmem'), "
            "((2, 128, 128), 'jnp.float32', 'vmem')]",
            code,
        )
        # Eager reference
        block_k = 128
        running = torch.zeros(2, 128, dtype=torch.float32, device=a.device)
        acc_ref = torch.zeros(2, 128, 128, dtype=torch.float32, device=a.device)
        for kb in range(0, 256, block_k):
            chunk = a[:, :, kb : kb + block_k]
            running = running + chunk.sum(-1).float()
            acc_ref = acc_ref + running[:, :, None]
        ref = (acc_ref + running[:, :, None] * running[:, :, None]).to(a.dtype)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)


@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasIndirectGather(TestCase):
    @staticmethod
    def _gather_2d_kernel():
        @helion.kernel(backend="pallas", static_shapes=True)
        def gather(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [indices.size(0), table.size(1)],
                dtype=table.dtype,
                device=table.device,
            )
            for tile_b, tile_e in hl.tile([indices.size(0), table.size(1)]):
                out[tile_b, tile_e] = table[indices[tile_b], tile_e]
            return out

        return gather

    def test_gather_fp32_uses_highest_precision(self) -> None:
        gather = self._gather_2d_kernel()
        table = torch.randn(16, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather, (indices, table), block_sizes=[128, 64])
        self.assertIn("one_hot", code)
        self.assertIn("HIGHEST", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    def test_gather_bf16_skips_highest(self) -> None:
        gather = self._gather_2d_kernel()
        table = torch.randn(16, 64, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather, (indices, table), block_sizes=[128, 64])
        self.assertIn("one_hot", code)
        self.assertNotIn("HIGHEST", code)
        self.assertNotIn("astype(jnp.float32)", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    def test_gather_2d_index_tile(self) -> None:
        """Regression: 2D index tile must contract the last axis, not axis 1."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def gather(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [indices.size(0), indices.size(1), table.size(1)],
                dtype=table.dtype,
                device=table.device,
            )
            for tile_b, tile_s, tile_e in hl.tile(
                [indices.size(0), indices.size(1), table.size(1)]
            ):
                out[tile_b, tile_s, tile_e] = table[indices[tile_b, tile_s], tile_e]
            return out

        table = torch.randn(16, 128, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 16, (8, 128), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gather, (indices, table), block_sizes=[8, 128, 128]
        )
        self.assertIn("one_hot", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    def test_gather_over_vmem_budget_raises(self) -> None:
        """Table above VMEM budget fails fast with a clear message."""
        gather = self._gather_2d_kernel()
        table = torch.randn(65537, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 65537, (256,), device=DEVICE, dtype=torch.int32)
        with self.assertRaisesRegex(Exception, "exceeds the .* VMEM threshold"):
            code_and_output(gather, (indices, table), block_sizes=[128, 64])

    def test_gather_vmem_budget_uses_block_size(self) -> None:
        """Tiling broadcast dims shrinks the VMEM block.

        Full table is over the threshold but the resident block after tiling
        the broadcast dim fits, so the check must pass.
        """
        gather = self._gather_2d_kernel()
        # Full table = 8192 * 1024 * 4 = 32 MiB (over the 16 MiB limit).
        # Resident VMEM block with BE=256 = 8192 * 256 * 4 = 8 MiB, fits.
        table = torch.randn(8192, 1024, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 8192, (256,), device=DEVICE, dtype=torch.int32)
        code_and_output(gather, (indices, table), block_sizes=[128, 256])

    def test_gather_integer_table_rejected(self) -> None:
        """Gather on non-floating tables raises at plan time."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def gather(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [indices.size(0), table.size(1)],
                dtype=table.dtype,
                device=table.device,
            )
            for tile_b, tile_e in hl.tile([indices.size(0), table.size(1)]):
                out[tile_b, tile_e] = table[indices[tile_b], tile_e]
            return out

        table = torch.randint(0, 100, (16, 64), device=DEVICE, dtype=torch.int32)
        indices = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.int32)
        with self.assertRaisesRegex(Exception, "must be floating point"):
            code_and_output(gather, (indices, table), block_sizes=[128, 64])

    def test_scatter_raises(self) -> None:
        """Indirect store has no Pallas strategy; plan_tiling must raise."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def scatter(
            out: torch.Tensor, values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            for tile_b, tile_e in hl.tile([values.size(0), values.size(1)]):
                out[indices[tile_b], tile_e] = values[tile_b, tile_e]
            return out

        out = torch.zeros(16, 64, device=DEVICE, dtype=torch.float32)
        values = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.arange(8, device=DEVICE, dtype=torch.int32)
        with self.assertRaisesRegex(Exception, "indirect store"):
            code_and_output(scatter, (out, values, indices), block_sizes=[8, 64])

    def test_gather_1d_index_bumps_block_to_tpu_alignment(self) -> None:
        """Block size on a 1D int32 index must be bumped to 128."""
        gather = self._gather_2d_kernel()
        table = torch.randn(1024, 256, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 1024, (1024,), device=DEVICE, dtype=torch.int32)
        # If the bump didn't happen, the generated code would slice with
        # `pl.ds(offset_0, 8)`. That string must not appear.
        code, result = code_and_output(gather, (indices, table), block_sizes=[8, 64])
        self.assertNotIn("pl.ds(offset_0, 8)", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
