from __future__ import annotations

import math
from typing import Callable
import unittest

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallas
from helion._testing import xfailIfPallasInterpret
from helion._testing import xfailIfPallasTpu
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
def pallas_matmul_f32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """f32 matmul kernel mirroring the perf harness's helion variant.

    Used by the pin tests for the degenerate shape families (M=1, N=1,
    K=1, and combinations). On TPU the default ``lax.dot_general`` for
    f32 inputs falls back to bf16-internal multiplication; the Pallas
    lowering must request ``precision=jax.lax.Precision.HIGHEST`` so the
    MXU does full f32 multiplications.
    """
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], device=x.device, dtype=torch.float32)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_matmul_bf16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16 matmul kernel mirroring the perf harness's helion variant.

    Used by the pin test that verifies the bf16 2D path emits ``pl.dot``
    (the Pallas-native MXU primitive used by the hand-written reference
    kernel in ``examples/pallas_perf/matmul_pallas.py``) rather than the
    slower ``lax.dot_general`` fallback.
    """
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], device=x.device, dtype=torch.promote_types(x.dtype, y.dtype)
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
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
def pallas_bmm_subrange_k(
    A: torch.Tensor, B: torch.Tensor, k_start: int, k_end: int
) -> torch.Tensor:
    """BMM where the K reduction only covers [k_start, k_end)."""
    b, m, k = A.size()
    b2, k2, n = B.size()
    out = torch.zeros(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k_start, k_end):
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
def pallas_scalar_lookup_in_pipeline(
    biases: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Per-program scalar lookup from a small 1-D table combined with an
    inner pipeline loop. Each of the ``G`` outer programs reads its own
    ``biases[g]`` and broadcasts it across the inner pipeline body."""
    G = biases.size(0)
    M = x.size(0)
    for g in hl.grid(G):
        b = biases[g]
        for tile_m in hl.tile(M):
            out[tile_m] = x[tile_m] + b
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
    k_view = k_in.reshape([-1, n_dim, head_dim])
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
            # scaling Q in-loop on-demand reduces spillage, faster than keeping pre-scaled Q
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            # Keep scores in fp32 to match SDPA tolerances on bf16/fp16 inputs.
            # same as hl.dot(q, k, out_dtype=torch.float32)
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_row_scale_mul(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Elementwise multiply ``x [M, N]`` by per-row scale ``r [M, 1]``.

    Iterates rows with a two-level tiling: an outer CTA tile and an inner
    ``hl.tile(begin, end)`` that becomes the per-Pallas-loop-type body.
    """
    m, _ = x.shape
    out = torch.empty_like(x)
    for mb_cta in hl.tile(m, block_size=8):
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            out[mb, :] = x[mb, :] * r[mb, :]
    return out


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


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_chunked_add(x: torch.Tensor) -> torch.Tensor:
    """Iterates over chunks of rows; uses tile_k.index + tile_chunk.begin * chunk_size
    to compute the global row index (TileIndexWithOffsetPattern)."""
    nrows, ncols = x.shape
    chunk_size = 64
    nchunks = nrows // chunk_size
    out = torch.empty_like(x)
    for tile_col, tile_chunk in hl.tile([ncols, nchunks], block_size=[None, 1]):
        for tile_k in hl.tile(chunk_size, block_size=64):
            row = tile_k.index + tile_chunk.begin * chunk_size
            out[row, tile_col] = x[row, tile_col] + 1.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_rand_add(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Kernel that uses hl.rand to generate random values and add them to x."""
    out = torch.empty_like(x)
    (m,) = x.size()
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m] + hl.rand([tile_m], seed=seed)
    return out


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

    @xfailIfPallasInterpret(
        "torch.float8_e4m3fn has no JAX dtype mapping in interpret mode; "
        "conversion errors before the VMEM check fires"
    )
    def test_estimate_pallas_vmem_bytes_fp8(self) -> None:
        """VMEM OOM at fp8 (1 byte/element)."""
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

    def test_store_slice_1d(self) -> None:
        """Store value sliced when block_size > tensor dim (1D)."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def fill_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = hl.full([tile], 1.0, dtype=x.dtype)
            return out

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(fill_kernel, (x,), block_size=4096)
        self.assertIn("[:1024]", code)
        torch.testing.assert_close(result, torch.ones_like(x))

    def test_store_slice_2d(self) -> None:
        """Store value sliced on the dim where block_size > tensor dim (2D)."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def fill_2d(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.full([tile_m, tile_n], 1.0, dtype=x.dtype)
            return out

        # 100 < 128, 256 == 256 → only dim 0 needs slicing
        x = torch.randn(100, 256, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(fill_2d, (x,), block_size=[128, 256])
        self.assertIn("[:100, :]", code)
        torch.testing.assert_close(result, torch.ones_like(x))

        # 100 < 128, 200 < 256 → both dims need slicing
        x2 = torch.randn(100, 200, device=DEVICE, dtype=torch.float32)
        code2, result2 = code_and_output(fill_2d, (x2,), block_size=[128, 256])
        self.assertIn("[:100, :200]", code2)
        torch.testing.assert_close(result2, torch.ones_like(x2))

    def test_store_slice_skips_pl_ds_dim(self) -> None:
        """Store value is not sliced on dimensions indexed with pl.ds()."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def fill_inner_loop(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    out[tile_m, tile_n] = hl.full([tile_m, tile_n], 1.0, dtype=x.dtype)
            return out

        x = torch.randn(64, 32, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            fill_inner_loop,
            (x,),
            block_size=[128, 64],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("pl.ds(", code)
        self.assertIn("[:64, :]", code)
        self.assertNotIn("[:64, :32]", code)
        torch.testing.assert_close(result, torch.ones_like(x))

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

    def test_pallas_matmul_bf16_emits_pl_dot(self) -> None:
        """bf16 1024x1024x1024: 2D MXU-legal tile should emit ``pl.dot``.

        The hand-written reference kernel in
        ``examples/pallas_perf/matmul_pallas.py`` uses ``pl.dot`` for the
        bf16 square-matmul body; the Mosaic backend lowers it directly to
        the TPU MXU. The previous Helion path emitted the equivalent
        ``lax.dot_general(..., preferred_element_type=jnp.float32)``,
        which the same Mosaic backend serializes through a slower
        general dot path. The pin asserts:

        * ``pl.dot(`` appears in the generated code (the routing rule
          fired);
        * the slower ``lax.dot_general(`` fallback is absent for this
          2D bf16 tile (i.e. ``preferred_element_type=jnp.float32`` is no
          longer needed because ``pl.dot`` returns f32 on bf16 inputs
          automatically).

        The shape is the headline 1024^3 anchor and the block matches
        ``BLOCK_CONFIGS[1] = (128, 128, 128)`` from
        ``examples/pallas_perf/matmul_configs.py``.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16, (x, y), block_sizes=[128, 128, 128]
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        self.assertIn("pl.dot(", code)
        self.assertNotIn("lax.dot_general(", code)
        self.assertNotIn("preferred_element_type=jnp.float32", code)

    def test_pallas_matmul_bf16_inplace_accumulator(self) -> None:
        """bf16 1024x1024x1024: K-loop accumulator stays in VMEM scratch.

        The hand-written reference kernel in
        ``examples/pallas_perf/matmul_pallas.py`` accumulates with
        ``acc_ref[...] += pl.dot(x_val, y_val)`` directly into the VMEM
        scratch, letting Mosaic keep the accumulator on the MXU between
        iterations. The previous Helion path emitted the value-flow form
        ``acc_N = scratch[...] + dot(...)`` plus
        ``scratch[...] = acc_N[...]``, which forced an extra register
        binding per K iteration before the value made it back to VMEM.
        The pin asserts:

        * ``scratch_0[...] += pl.dot(`` appears in the generated code
          (the rewrite fired);
        * the externalised value-flow form
          ``scratch_0[...] = acc`` (with a trailing identifier rather than
          the ``acc[...]`` scratch initialiser) is absent inside the
          ``_pipeline_body`` closure.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16, (x, y), block_sizes=[128, 128, 128]
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # The augmented assignment lands the accumulator addition straight
        # back into the VMEM scratch — this is the marker that the rewrite
        # in `_write_back_loop_carried` fired.
        self.assertIn("scratch_0[...] += pl.dot(", code)
        # The externalised write-back from the prior lowering (the bind
        # cost we are removing) used a trailing ``acc_N[...]`` slice.
        # Match the specific form to avoid false negatives against the
        # one-time ``scratch_0[...] = acc[...]`` initialiser that lives
        # outside the pipeline body.
        self.assertNotIn("scratch_0[...] = acc_1[...]", code)

    def test_pallas_matmul_bf16_outer_grid_no_reduction_dim(self) -> None:
        """bf16 1024x1024x1024: outer-grid carries no reduction axis.

        The hand-written reference kernel in
        ``examples/pallas_perf/matmul_pallas.py`` uses a 3-axis outer grid
        ``(grid_m, grid_n, grid_k)`` and marks K as ``"arbitrary"``. Helion
        instead keeps the K reduction inside ``pltpu.emit_pipeline``, so
        the outer ``pl.pallas_call`` grid only carries M and N — both of
        which truly are independent and stay ``"parallel"`` (the launcher
        default). The pin asserts:

        * The host wrapper does not pass
          ``_reduction_grid_dims=`` to the launcher for this kernel — the
          codegen helper ``_compute_reduction_grid_dims`` returns an empty
          list when no outer-grid axis is a reduction (M and N are not
          ``reduction=True`` block sizes). When the list is empty the
          ``build_launcher_args`` builder omits the kwarg entirely, and
          the launcher falls back to marking every outer-grid axis
          ``"parallel"``.

        If the K loop were ever lifted into the outer grid (a structural
        refactor toward the hand-written 3-axis pattern), the helper would
        flag K's grid dim and this assertion would need updating to
        ``assertIn("_reduction_grid_dims=", code)`` with the K axis index.

        Block sizes are the headline anchor ``(128, 128, 128)`` from
        ``examples/pallas_perf/matmul_configs.py``. ``pallas_loop_type``
        is forced to ``"emit_pipeline"`` so the assertion targets the
        launcher path that the headline benchmark uses; the ``"unroll"``
        path uses a different launcher that wouldn't carry the kwarg
        either.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[128, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # No outer-grid axis is a reduction, so the kwarg should be absent.
        self.assertNotIn("_reduction_grid_dims=", code)

    def test_pallas_matmul_bf16_emit_pipeline_outer_vmem_strip(self) -> None:
        """bf16 1024x1024x1024: emit_pipeline outer in_specs become VMEM strips.

        The hand-written reference kernel in
        ``examples/pallas_perf/matmul_pallas.py`` does NOT route its
        pipelined ``x`` / ``y`` operands through ``pl.BlockSpec(
        memory_space=pltpu.HBM)`` — it keeps them as VMEM strips
        ``(bm, K)`` / ``(K, bn)`` indexed by the outer grid coordinate, so
        the inner emit_pipeline slices ``bk``-wide chunks from VMEM
        instead of DMAing each chunk from HBM per inner iteration.  A Deep
        Replan pod probe at bf16 block 128 measured a ~1.9x speedup from
        this single change (478 us → 249 us).

        Helion's previous lowering always forced the HBM ref form via
        ``_pallas_build_pipeline_specs``.  This pin asserts both halves of
        the fix:

        * The launcher receives ``_pipeline_vmem_strip_indices=`` so it
          knows which pipelined args to route to the new strip BlockSpec
          (the existing ``_pallas_make_block_spec`` already produces
          ``(bm, K) lambda i, j: (i, 0)`` for the matmul tensors —
          ``_pallas_build_pipeline_specs`` just stops forcing them to
          HBM when their id is in this list).
        * The inner emit_pipeline ``pl.BlockSpec`` lambda uses ``0`` for
          the outer-grid M / N dim instead of ``_outer_pid_0`` /
          ``_outer_pid_1``, because the outer ref is now already pre-
          sliced for the outer-grid coordinate.

        Block sizes ``(128, 128, 128)`` (well inside the
        ``_OUTER_VMEM_STRIP_BUDGET_BYTES = 10MB`` budget for bf16 1024³)
        and ``pallas_loop_type='emit_pipeline'`` force the path the
        marker pins.  A larger block that would push the strip working
        set past the budget would fall back to the HBM ref form — that
        guard is intentional and not exercised here.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[128, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Launcher-side: pipelined matmul operands are tagged so the
        # launcher emits a VMEM strip BlockSpec for them rather than an
        # HBM ref.
        self.assertIn("_pipeline_vmem_strip_indices=", code)
        # Inner emit_pipeline BlockSpec: the outer-grid pid (``_outer_pid_0``
        # / ``_outer_pid_1``) is replaced by ``0`` for VMEM-strip
        # operands.  Match strict substrings of the emitted lambdas so a
        # regression that put the pid back in would fail.  The matmul
        # pipelines both ``x`` (outer M dim → ``(0, _j)``) and ``y``
        # (outer N dim → ``(_j, 0)``).
        self.assertIn("lambda _j: (0, _j)", code)
        self.assertIn("lambda _j: (_j, 0)", code)
        # And the outer pid variables stay unused inside the
        # ``_pipeline_body`` for these strip operands — i.e. they no
        # longer appear inside the BlockSpec lambdas.
        self.assertNotIn("lambda _j: (_outer_pid_0,", code)
        self.assertNotIn("lambda _j: (_outer_pid_1,", code)

    def test_pallas_matmul_bf16_outer_grid_lifts_k_axis(self) -> None:
        """bf16 1024x1024x1024: ``pallas_loop_type='outer_grid'`` lifts the
        inner K loop into the outer ``pl.pallas_call`` grid.

        The hand-written reference kernel in
        ``examples/pallas_perf/matmul_pallas.py`` uses a 3-axis outer grid
        ``(grid_m, grid_n, grid_k)`` with
        ``dimension_semantics=("parallel", "parallel", "arbitrary")``, a
        ``pl.when(pl.program_id(2) == 0)`` accumulator-init guard, and a
        ``pl.when(pl.program_id(2) == nsteps - 1)`` accumulator-store guard
        instead of Helion's default inner ``pltpu.emit_pipeline`` over K.

        Helion's new ``"outer_grid"`` ``pallas_loop_type`` value drives this
        structural restructure when the matmul-pattern eligibility check
        passes (single inner block_id with loop-carried state, non-reduction
        outer pids).  The pin asserts all four sides of the fix:

        * The launcher receives ``_reduction_grid_dims=[2]`` so the lifted
          K axis is marked ``"arbitrary"`` in ``dimension_semantics``.
        * ``_block_spec_info`` for X carries the K grid dim ``(0, 2)`` and
          Y carries ``(2, 1)`` (instead of the pre-G2-H ``(0, None)`` /
          ``(None, 1)`` form where K was inside ``pltpu.emit_pipeline``).
        * No ``pltpu.emit_pipeline(`` call survives in the generated body
          — the K loop is the outer grid now.
        * The kernel body contains ``@pl.when(_outer_pid_2 == 0)`` and
          ``@pl.when(_outer_pid_2 == _k_nsteps_2 - 1)`` guards wrapping
          the accumulator init and store, matching the reference matmul
          exactly.

        Block sizes ``(128, 128, 128)`` keep the shape inside any per-tile
        budgets; ``pallas_loop_type='outer_grid'`` forces the path the
        marker pins.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[128, 128, 128],
            pallas_loop_type="outer_grid",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Launcher-side: K axis is in the outer grid and flagged as
        # ``"arbitrary"`` via ``_reduction_grid_dims=[2]``.  The expanded
        # ``_block_spec_info`` ties X's K dim to grid dim 2 and Y's K dim
        # to grid dim 2 too, while M (0) and N (1) stay on grid dims 0/1.
        self.assertIn("_reduction_grid_dims=[2]", code)
        self.assertIn("((128, 128), (0, 2))", code)  # X(M, K) -> grid (0, 2)
        self.assertIn("((128, 128), (2, 1))", code)  # Y(K, N) -> grid (2, 1)
        self.assertIn("((128, 128), (0, 1))", code)  # out(M, N) -> grid (0, 1)
        # No emit_pipeline call survives — the K loop is the outer grid.
        self.assertNotIn("pltpu.emit_pipeline(", code)
        # Body-side: the lifted K axis pid var and nsteps var anchor the
        # pl.when guards exactly like ``matmul_pallas.py``.
        self.assertIn("_outer_pid_2 = pl.program_id(2)", code)
        self.assertIn("_k_nsteps_2 = pl.num_programs(2)", code)
        self.assertIn("@pl.when(_outer_pid_2 == 0)", code)
        self.assertIn("@pl.when(_outer_pid_2 == _k_nsteps_2 - 1)", code)
        # The accumulator stays in scratch and is += into across K
        # iterations, matching the existing G2-E in-place rewrite.
        self.assertIn("scratch_0[...] += pl.dot(", code)
        # Pre-G2-H ``_pipeline_vmem_strip_indices=`` / ``_pipeline_arg_indices=``
        # kwargs become irrelevant once the K loop is in the outer grid —
        # X / Y use ordinary outer BlockSpecs.
        self.assertNotIn("_pipeline_vmem_strip_indices=", code)
        self.assertNotIn("_pipeline_arg_indices=", code)

    def test_pallas_matmul_outer_grid_falls_back_on_singleton_m(self) -> None:
        """``pallas_loop_type='outer_grid'`` must refuse to lift when an
        outer-grid axis has block size 1 and fall back to ``emit_pipeline``.

        Background: the outer-grid body rewrite assumes the loop-carried
        scratch holds a 2D matmul tile.  When ``bm == 1`` on an M=1 shape
        (e.g. bf16 1×1024×1024), the scratch holds a 1-row broadcast outer
        product and the rewrite emits silently-wrong outputs whenever the
        inner K loop has > 1 iteration (relative diffs up to 4.5e6 vs CPU
        f32 reference; the autotuner's accuracy check skips these configs,
        but forced configs are unprotected).  The eligibility check in
        ``_codegen_outer_grid_or_fallback`` therefore refuses any outer-grid
        axis whose configured block size resolves to 1.

        The pin asserts the fallback both ways: ``pltpu.emit_pipeline(``
        must survive (the K loop stays inside the inner pipeline) and the
        ``@pl.when(_outer_pid_2 == 0)`` / ``@pl.when(_outer_pid_2 == ...)``
        guards that mark the outer-grid form must NOT appear.  The numeric
        check confirms the fallback produces correct outputs.
        """
        torch.manual_seed(0)
        x = torch.randn(1, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[1, 128, 128],
            pallas_loop_type="outer_grid",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Fallback fired: inner pipeline survives, no outer-grid K guards.
        self.assertIn("pltpu.emit_pipeline(", code)
        self.assertNotIn("@pl.when(_outer_pid_2 == 0)", code)
        self.assertNotIn("_reduction_grid_dims=[2]", code)

    def test_pallas_matmul_outer_grid_fires_on_multi_m(self) -> None:
        """``pallas_loop_type='outer_grid'`` still fires on multi-row M.

        Companion to ``test_pallas_matmul_outer_grid_falls_back_on_singleton_m``:
        with ``bm > 1`` the eligibility check should still let the K loop lift
        into the outer ``pl.pallas_call`` grid.  This pins that the M=1 guard
        does not over-refuse and quietly disable the outer-grid path on the
        working shapes G2-H landed on.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[128, 128, 128],
            pallas_loop_type="outer_grid",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Outer-grid path fired: K is in the outer grid, no inner pipeline.
        self.assertIn("_reduction_grid_dims=[2]", code)
        self.assertIn("@pl.when(_outer_pid_2 == 0)", code)
        self.assertNotIn("pltpu.emit_pipeline(", code)

    def test_pallas_matmul_bf16_outer_grid_omits_dead_pids(self) -> None:
        """``pallas_loop_type='outer_grid'`` DCEs unused ``_outer_pid_N`` reads.

        The outer-grid body rewrite emits the K pid (used by the
        ``@pl.when(_outer_pid_K == ...)`` guards) plus an
        ``_outer_pid_N = pl.program_id(N)`` setup for every other outer
        axis (M=0, N=1).  For the matmul body only the K pid is
        referenced, so the M / N setups are dead and the DCE pass in
        ``DeviceFunction.codegen_function_def`` removes them.  Hand-edit
        ablation on the ``matmul_pallas.py`` mirror measured the dead
        reads at +7.4 us / +5.9% on the bf16 1024**3 headline (125.5
        us → 132.9 us), making this the highest-confidence remaining
        ~5% headline lever.

        The K pid setup (``_outer_pid_2 = pl.program_id(2)``) and the
        ``_k_nsteps_2 = pl.num_programs(2)`` companion both MUST
        survive — they anchor the init / store guards.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[128, 128, 128],
            pallas_loop_type="outer_grid",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Dead M / N pids dropped.
        self.assertNotIn("_outer_pid_0 = pl.program_id(0)", code)
        self.assertNotIn("_outer_pid_1 = pl.program_id(1)", code)
        # K pid + nsteps still present and feed the @pl.when guards.
        self.assertIn("_outer_pid_2 = pl.program_id(2)", code)
        self.assertIn("_k_nsteps_2 = pl.num_programs(2)", code)
        self.assertIn("@pl.when(_outer_pid_2 == 0)", code)
        self.assertIn("@pl.when(_outer_pid_2 == _k_nsteps_2 - 1)", code)

    def test_pallas_matmul_bf16_emit_pipeline_omits_dead_pids(self) -> None:
        """``emit_pipeline`` strip path DCEs unused ``_outer_pid_N`` reads.

        At small blocks the strip-eligibility pass keeps both ``x`` and
        ``y`` as VMEM strips, which means the inner BlockSpec lambdas
        emit ``0`` for outer-grid dims instead of reading the M / N
        pids (see ``test_pallas_matmul_bf16_emit_pipeline_outer_vmem_strip``
        for the pid-replacement pin).  At that point both
        ``_outer_pid_0`` and ``_outer_pid_1`` setups are dead and the
        DCE pass removes them.

        Companion negative pin
        ``test_pallas_matmul_bf16_emit_pipeline_keeps_used_pids_on_hbm_ref``
        below covers the HBM-ref branch where the pids ARE referenced
        by the lambdas and therefore must survive.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[128, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Strip path: lambdas emit ``0`` for outer dims; pids are dead.
        self.assertNotIn("_outer_pid_0 = pl.program_id(0)", code)
        self.assertNotIn("_outer_pid_1 = pl.program_id(1)", code)
        # Sanity: this IS the strip path (not HBM-ref).
        self.assertIn("lambda _j: (0, _j)", code)
        self.assertIn("lambda _j: (_j, 0)", code)
        # And ``pltpu.emit_pipeline`` still drives the K loop here
        # (this is the non-outer_grid path).
        self.assertIn("pltpu.emit_pipeline(", code)

    def test_pallas_matmul_bf16_emit_pipeline_keeps_used_pids_on_hbm_ref(
        self,
    ) -> None:
        """``emit_pipeline`` HBM-ref path keeps ``_outer_pid_N`` reads alive.

        When the outer VMEM strip working set exceeds
        ``_OUTER_VMEM_STRIP_BUDGET_BYTES``, the pipelined operands fall
        back to ``pl.BlockSpec(memory_space=pltpu.HBM)`` refs and the
        inner BlockSpec lambdas slice them with ``_outer_pid_N``.  The
        DCE pass must NOT drop those pid setups — they are read by the
        lambdas and the kernel will fail to compile without them.

        Block sizes ``(1024, 1024, 1024)`` on a ``4096³`` shape push
        each strip footprint to ~32 MB (well past the 10 MB budget) so
        every pipelined arg ends up in the HBM-ref branch.
        """
        torch.manual_seed(0)
        x = torch.randn(4096, 4096, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(4096, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_matmul_bf16,
            (x, y),
            block_sizes=[1024, 1024, 1024],
            pallas_loop_type="emit_pipeline",
        )
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        # Pids are read by the BlockSpec lambdas; DCE must leave them.
        self.assertIn("_outer_pid_0 = pl.program_id(0)", code)
        self.assertIn("_outer_pid_1 = pl.program_id(1)", code)
        # Sanity: the lambdas read the pids (HBM-ref form, not strip).
        self.assertIn("lambda _j: (_outer_pid_0, _j)", code)
        self.assertIn("lambda _j: (_j, _outer_pid_1)", code)
        # And the strip kwarg is absent because no operand fits the budget.
        self.assertNotIn("_pipeline_vmem_strip_indices=", code)

    def test_pallas_matmul_bmm_stays_on_dot_general(self) -> None:
        """3D BMM stays on ``lax.dot_general`` (``pl.dot`` is 2D-only).

        ``pl.dot`` rejects non-2D inputs upstream
        (``jax/_src/pallas/primitives.py``), so BMM kernels must keep
        using the 3D ``dimension_numbers`` form. The pin guards against a
        future routing widening that would break 3D paths.
        """
        torch.manual_seed(0)
        a = torch.randn(4, 128, 256, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        b = torch.randn(4, 256, 128, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            pallas_bmm, (a, b), block_sizes=[4, 128, 128, 256]
        )
        expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        self.assertIn("lax.dot_general(", code)
        self.assertIn("preferred_element_type=jnp.float32", code)
        self.assertNotIn("pl.dot(", code)

    def test_pallas_matmul_f32_singleton_n(self) -> None:
        """f32 1024x1024x1: N=1 (matrix x column vector).

        The default ``lax.dot_general`` on TPU silently downcasts f32
        operands to bf16 for the multiplication step. Without
        ``precision=jax.lax.Precision.HIGHEST`` the K=1024 reduction
        accumulates bf16-rounded products and the result drifts ~1e-1
        absolute. The pin asserts both the numeric fix and the marker
        in the generated code so it can't silently regress.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float32)
        torch.manual_seed(1)
        y = torch.randn(1024, 1, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_matmul_f32, (x, y), block_sizes=[128, 1, 128]
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertIn("lax.dot_general(", code)
        self.assertIn("precision=jax.lax.Precision.HIGHEST", code)

    def test_pallas_matmul_f32_singleton_k(self) -> None:
        """f32 1024x1x1024: K=1 (outer product).

        The K-block is forced to 1 so the reduction is a single
        ``lax.dot_general`` call. Even though K=1 there is only one
        multiply per cell, TPU's default precision still rounds the
        intermediate to bf16, leaking error into f32 output.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 1, device=DEVICE, dtype=torch.float32)
        torch.manual_seed(1)
        y = torch.randn(1, 1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_matmul_f32, (x, y), block_sizes=[128, 128, 1]
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertIn("lax.dot_general(", code)
        self.assertIn("precision=jax.lax.Precision.HIGHEST", code)

    def test_pallas_matmul_f32_singleton_m(self) -> None:
        """f32 1x1024x1024: M=1 (vector x matrix)."""
        torch.manual_seed(0)
        x = torch.randn(1, 1024, device=DEVICE, dtype=torch.float32)
        torch.manual_seed(1)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_matmul_f32, (x, y), block_sizes=[1, 128, 128]
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertIn("lax.dot_general(", code)
        self.assertIn("precision=jax.lax.Precision.HIGHEST", code)

    def test_pallas_matmul_f32_scalar_vector(self) -> None:
        """f32 1x1x1024: scalar broadcast through dot_general."""
        torch.manual_seed(0)
        x = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        torch.manual_seed(1)
        y = torch.randn(1, 1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_matmul_f32, (x, y), block_sizes=[1, 128, 1]
        )
        expected = x @ y
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        self.assertIn("lax.dot_general(", code)
        self.assertIn("precision=jax.lax.Precision.HIGHEST", code)

    def test_pallas_autotuner_final_pick_picks_true_best_on_noisy_initial_rank(
        self,
    ) -> None:
        """Final-pick verification re-ranks past noisy initial measurements.

        On TPU pod-wide thermal / scheduler noise sometimes causes the
        bf16 1024^3 autotuner to record a single-call median of ~224 us
        for ``block_sizes=[512, 1024, 512]`` while
        ``block_sizes=[512, 512, 512]`` records ~232 us in the same pass
        and ~190 us when re-measured back-to-back (Deep Replan 2026-05-23,
        plan.md §2.1 (a)). The new
        ``PopulationBasedSearch.run_final_pick_verification`` phase
        guards against that by re-benchmarking the top-K candidates
        ``HELION_AUTOTUNE_FINAL_PICK_PASSES`` extra times and ranking by
        the median of per-pass medians.

        The pin asserts that when the **initial** ``perf`` of the
        ``[512, 1024, 512]`` candidate is noisy-better but its true
        repeated rebenchmark stays slower than ``[512, 512, 512]``, the
        verification picks ``[512, 512, 512]``. The two block-512 configs
        are the ones the bf16 1024^3 autotuner historically ties between
        ([512, 512, 512] and [256, 256, 256]); the [512, 1024, 512]
        config is the one we are explicitly steering away from.
        """
        from unittest.mock import patch

        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.base_search import PopulationMember
        from helion.runtime.config import Config

        # Fake measured times in ms.  "True" means the steady-state cost
        # the kernel actually pays; "noisy initial" is the first
        # rebenchmark median seen during search, which mis-ranks the
        # configs.  ``[512, 1024, 512]`` shows a deceptive 0.220 ms in the
        # initial benchmark even though every repeated probe lands above
        # 0.232 ms; ``[512, 512, 512]`` shows a slow 0.232 ms initial but
        # is consistently 0.190 ms when re-measured.
        scripted_candidates = [
            {
                "block_sizes": [512, 1024, 512],
                "noisy_initial_ms": 0.220,
                "true_passes_ms": [0.232, 0.234, 0.230],
            },
            {
                "block_sizes": [512, 512, 512],
                "noisy_initial_ms": 0.232,
                "true_passes_ms": [0.190, 0.192, 0.188],
            },
            {
                "block_sizes": [256, 256, 256],
                "noisy_initial_ms": 0.234,
                "true_passes_ms": [0.205, 0.210, 0.208],
            },
            {
                "block_sizes": [128, 128, 128],
                "noisy_initial_ms": 0.260,
                "true_passes_ms": [0.300, 0.305, 0.302],
            },
        ]

        members: list[PopulationMember] = []
        for entry in scripted_candidates:
            config = Config(block_sizes=list(entry["block_sizes"]))
            members.append(
                PopulationMember(
                    fn=lambda *a, **kw: None,
                    perfs=[entry["noisy_initial_ms"]],
                    flat_values=list(entry["block_sizes"]),
                    config=config,
                    status="ok",
                    compile_time=0.0,
                )
            )

        # Construct a minimal search instance without going through the
        # full constructor (which requires a compiled kernel + arg tensors).
        # ``log`` must itself be callable plus expose ``.debug`` /
        # ``.warning`` like the real ``AutotuningLogger`` does, so wrap a
        # no-op callable in a small adapter.
        class _NoopLog:
            def __call__(self, *_: object, **__: object) -> None:
                return None

            def debug(self, *_: object, **__: object) -> None:
                return None

            def warning(self, *_: object, **__: object) -> None:
                return None

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.population = members
        search.best_perf_so_far = min(m.perf for m in members)
        search.log = _NoopLog()

        # Lookup of true per-pass timings keyed by config identity.  The
        # pass counter advances per rebenchmark call so each pass sees a
        # different scripted timing.
        true_passes_by_id: dict[int, list[float]] = {
            id(m): list(entry["true_passes_ms"])
            for m, entry in zip(members, scripted_candidates, strict=True)
        }

        def fake_rebenchmark(
            members_to_bench: list[PopulationMember], *, desc: str = ""
        ) -> None:
            for member in members_to_bench:
                # Pop the next scripted timing for this member; reuse the
                # last value if we exhaust the script.
                queue = true_passes_by_id[id(member)]
                timing = queue.pop(0) if queue else member.perfs[-1]
                member.perfs.append(timing)
                if timing < search.best_perf_so_far:
                    search.best_perf_so_far = timing

        with patch.object(search, "rebenchmark", side_effect=fake_rebenchmark):
            initial_best = min(members, key=lambda m: m.perf)
            self.assertEqual(
                list(initial_best.config["block_sizes"]),
                [512, 1024, 512],
                "Precondition: noisy initial perf misranks 1024-bn config as best",
            )
            final_best = search.run_final_pick_verification(
                initial_best, passes=3, top_k=5
            )

        self.assertIn(
            list(final_best.config["block_sizes"]),
            ([512, 512, 512], [256, 256, 256]),
            "Final-pick verification must re-rank away from [512, 1024, 512] "
            "(under-parallelised bn==N) and into the [512, 512, 512] or "
            "[256, 256, 256] family.",
        )

    def test_pallas_matmul_bf16_square_seed_in_initial_population(self) -> None:
        """Compiler seeds the headline-fast config into the initial population.

        Deep Replan 2026-05-23 (plan.md §2.5 row 2) measured the forced
        ``emit_pipeline [512, 512, 512] pb=False`` config on bf16 1024**3
        at 161 us -- the fastest known Helion config -- but the autotuner
        kept mis-ranking it against ``unroll`` family picks because of
        per-call noise on short kernels.

        ``PallasMatmulSquareSeedHeuristic`` plants this config in the
        initial population (via ``ConfigSpec.compiler_seed_configs``,
        the same path used by ``TritonSkinnyGemmHeuristic``).
        Together with the existing final-pick verification phase
        (re-ranking the top-10) this guarantees the autotuner always
        considers the headline-fast config -- the search no longer has
        to discover it by mutation.

        Pin asserts:
          1. The heuristic is eligible for bf16 1024**3.
          2. ``compiler_seed_configs`` includes the
             ``[512, 512, 512] emit_pipeline pb=False`` config.
          3. The skinny shape (M=1) is NOT seeded (the heuristic refuses
             when any static dim is below 512 so the M=1 / N=1 / K=1
             paths -- which want ``unroll [_, _, 1]`` or
             ``fori_loop`` -- aren't biased toward an inappropriate
             family).
        """
        from helion._compiler.autotuner_heuristics.pallas import (
            PallasMatmulSquareSeedHeuristic,
        )

        torch.manual_seed(0)
        x = torch.empty(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.empty(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        bound = pallas_matmul_bf16.bind((x, y))

        self.assertTrue(
            PallasMatmulSquareSeedHeuristic.is_eligible(
                bound.env, bound.host_function.device_ir
            ),
            "Heuristic must fire on bf16 1024**3 (all dims >= 512, 2D matmul).",
        )
        self.assertIn(
            PallasMatmulSquareSeedHeuristic.name,
            bound.config_spec.autotuner_heuristics,
            "Heuristic name must be recorded so the seed is attributable.",
        )
        seed_blocks = (512, 512, 512)
        seeded = [
            (
                tuple(cfg.config.get("block_sizes", ())),
                cfg.config.get("pallas_loop_type"),
                cfg.config.get("pallas_pre_broadcast"),
            )
            for cfg in bound.config_spec.compiler_seed_configs
        ]
        self.assertIn(
            (seed_blocks, "emit_pipeline", False),
            seeded,
            "Compiler seed configs must include the headline-fast "
            "[512, 512, 512] emit_pipeline pb=False entry.",
        )

        skinny_x = torch.empty(1, 1024, device=DEVICE, dtype=torch.bfloat16)
        skinny_y = torch.empty(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        skinny_bound = pallas_matmul_bf16.bind((skinny_x, skinny_y))
        self.assertFalse(
            PallasMatmulSquareSeedHeuristic.is_eligible(
                skinny_bound.env, skinny_bound.host_function.device_ir
            ),
            "Heuristic must refuse skinny shapes (M=1) so the [512, 512, 512] "
            "block tile isn't seeded for shapes that can't use it.",
        )
        skinny_seeded = [
            tuple(cfg.config.get("block_sizes", ()))
            for cfg in skinny_bound.config_spec.compiler_seed_configs
        ]
        self.assertNotIn(
            seed_blocks,
            skinny_seeded,
            "Skinny shape must not receive the [512, 512, 512] seed.",
        )

    def test_pallas_autotuner_compiler_seed_survives_final_pick(self) -> None:
        """Compiler-seeded members are re-considered during final-pick verification.

        The LFBO surrogate-driven search aggressively prunes the
        population between generations: by the time
        ``run_final_pick_verification`` fires, ``self.population`` may
        hold only 2-3 members of the last generation's neighbors plus
        the running best.  Without this pin a compiler-seeded
        ``[512, 512, 512] emit_pipeline pb=False`` config that measured
        merely "average" on its noisy initial bench would be dropped
        from later generations -- the verification phase would never
        see it.

        The fix snapshots compiler-seeded members on
        ``PopulationBasedSearch._compiler_seed_members`` at the end of
        the initial-rebench step and merges them into the verification
        candidate pool, so a backend's hand-picked seed always gets
        re-benched against the search's best even when the search
        moved away from it.
        """
        from unittest.mock import patch

        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.base_search import PopulationMember
        from helion.runtime.config import Config

        # "Last-gen population" = two random survivors of the LFBO
        # pattern search, both around 0.205 ms.  "Compiler seed" = the
        # hand-picked fast config that scored 0.215 ms on its noisy
        # initial bench (so the LFBO surrogate dropped it from later
        # generations) but consistently rebenches at 0.190 ms.
        last_gen = [
            (
                Config(block_sizes=[1024, 256, 1024]),
                0.205,
                [0.207, 0.206, 0.204],
            ),
            (
                Config(block_sizes=[256, 256, 256]),
                0.210,
                [0.212, 0.211, 0.209],
            ),
        ]
        compiler_seed = (
            Config(
                block_sizes=[512, 512, 512],
                pallas_loop_type="emit_pipeline",
                pallas_pre_broadcast=False,
            ),
            0.215,
            [0.190, 0.191, 0.189],
        )

        def make_member(config: Config, noisy_perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda *a, **kw: None,
                perfs=[noisy_perf],
                flat_values=[id(config)],  # opaque -- never read by the test
                config=config,
                status="ok",
                compile_time=0.0,
            )

        last_gen_members = [make_member(cfg, perf) for cfg, perf, _passes in last_gen]
        seed_member = make_member(compiler_seed[0], compiler_seed[1])
        true_passes_by_id: dict[int, list[float]] = {
            id(member): list(passes)
            for member, (_cfg, _noisy, passes) in zip(
                last_gen_members, last_gen, strict=True
            )
        }
        true_passes_by_id[id(seed_member)] = list(compiler_seed[2])

        class _NoopLog:
            def __call__(self, *_: object, **__: object) -> None:
                return None

            def debug(self, *_: object, **__: object) -> None:
                return None

            def warning(self, *_: object, **__: object) -> None:
                return None

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.population = last_gen_members  # seed NOT in last-gen population
        search.best_perf_so_far = min(m.perf for m in last_gen_members)
        search._compiler_seed_members = [seed_member]
        search.log = _NoopLog()

        def fake_rebenchmark(
            members_to_bench: list[PopulationMember], *, desc: str = ""
        ) -> None:
            for member in members_to_bench:
                queue = true_passes_by_id[id(member)]
                timing = queue.pop(0) if queue else member.perfs[-1]
                member.perfs.append(timing)
                if timing < search.best_perf_so_far:
                    search.best_perf_so_far = timing

        with patch.object(search, "rebenchmark", side_effect=fake_rebenchmark):
            initial_best = min(last_gen_members, key=lambda m: m.perf)
            self.assertEqual(
                list(initial_best.config["block_sizes"]),
                [1024, 256, 1024],
                "Precondition: search's running best is one of the last-gen members.",
            )
            final_best = search.run_final_pick_verification(
                initial_best, passes=3, top_k=10
            )

        self.assertEqual(
            list(final_best.config["block_sizes"]),
            [512, 512, 512],
            "Compiler-seeded [512, 512, 512] must be re-benched and "
            "re-rank ahead of the last-gen best once its true 0.190 ms "
            "perf is measured.",
        )

    def test_pallas_launcher_fast_path_hits_on_repeat_invocations(self) -> None:
        """Cached static-shape Pallas launcher elides per-call helper iteration.

        Deep Replan 2026-05-23 (§2.7 in ``plan.md``) decomposed the
        bf16 1024^3 headline gap to per-call Python dispatch overhead.
        On a cache hit, ``default_pallas_launcher`` /
        ``default_pallas_pipeline_launcher`` / ``default_pallas_fori_launcher``
        each branch into a fast path that:

        * Reuses a precomputed ``_LauncherFastPath`` instance carried on
          the cached launcher entry instead of re-running
          ``_pallas_apply_ds_padding`` / ``_pallas_check_dtypes`` /
          the output-only-results loop per call.
        * Increments ``helion.runtime._LAUNCHER_FAST_PATH_HITS`` on each
          cache hit so this pin can assert the elision is exercised.

        The kernel allocates its output via ``torch.empty`` inside the
        kernel body, so ``out`` is an output-only tensor and the fast
        path exercises both the output-only-results loop short-circuit
        and the ds-pad skip (1024-divisible block sizes mean every
        pad_amount is 0).  The test binds + ``compile_config`` directly
        so autotuning doesn't muddy the counter (autotuning would call
        the launcher hundreds of times across configs); only the
        compiled callable's per-call launcher work is observed.

        Regression hazard pinned: if a refactor splits the cache tuple
        differently or removes the counter increment, this test fails.
        Equally, if a refactor accidentally takes the slow path on
        every call (e.g. by storing a 4-tuple instead of a 5-tuple and
        always missing the new ``fast_path`` unpack), the hit count
        drops to 0 and the test fails.
        """
        from helion import runtime as helion_runtime

        torch.manual_seed(0)
        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)

        bound = pallas_matmul_bf16.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        helion_runtime._reset_launcher_fast_path_hits()
        self.assertEqual(helion_runtime._launcher_fast_path_hits(), 0)

        # First call seeds the launcher cache (slow path / no counter
        # bump); subsequent calls take the fast path.
        result = compiled_fn(x, y)
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(
            helion_runtime._launcher_fast_path_hits(),
            0,
            "First call should miss the launcher cache and run the "
            "slow path (no fast-path counter bump).",
        )

        # Repeat invocations: same kernel, same shape, same dtype → cache hit.
        n_repeats = 4
        for _ in range(n_repeats):
            result = compiled_fn(x, y)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

        self.assertEqual(
            helion_runtime._launcher_fast_path_hits(),
            n_repeats,
            f"Repeat calls on a cached static-shape kernel must each hit "
            f"the launcher fast path. Expected {n_repeats} hits; got "
            f"{helion_runtime._launcher_fast_path_hits()}.",
        )

    def test_pallas_jaxcallable_key_cache_hits_on_repeat_invocations(self) -> None:
        """Cached static-shape JaxCallable reuses torch_tpu invocation key.

        Deep Replan 2026-05-23 (§2.7 in ``plan.md``) localized ~50us of
        the per-call Pallas dispatch overhead to torch_tpu's
        ``JaxCallable.__call__``, of which the f-string built by
        ``_get_kernel_invocation_key`` (per-arg shape / dtype walk) is a
        measurable slice on every call.  For ``static_shapes=True``
        kernels the invocation key is constant across calls, so Helion
        installs a ``JaxCallable`` subclass that caches the first
        call's ``(arg_shape_dtype_signature, kernel_key, output_shapes,
        out_tree, input_output_aliases_items)`` and on subsequent calls
        with a matching signature short-circuits to a direct
        ``tpu_torch_pallas.call_custom_kernel`` invocation, bumping
        ``helion.runtime._JAXCALLABLE_KEY_CACHE_HITS`` once per cache
        hit.

        The test asserts the counter increments exactly ``n_repeats``
        times across ``1 + n_repeats`` consecutive calls of the same
        compiled callable: the first call seeds the JaxCallable's
        internal ``output_shapes`` cache and the subclass's
        ``_helion_key_cache``; calls 2..N reuse them.
        """
        from helion import runtime as helion_runtime

        # Define the kernel inside the test so the inner Helion-emitted
        # function (and its ``_pallas_pipeline_cache``) is unique to
        # this test run -- no cross-test pollution from other tests
        # that bind ``pallas_matmul_bf16``.  The launcher cache lives on
        # the *inner* generated function, not on the user-facing
        # decorator object, so clearing attributes off the module-level
        # ``pallas_matmul_bf16`` wouldn't help.

        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_jaxcallable_pin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n],
                device=x.device,
                dtype=torch.promote_types(x.dtype, y.dtype),
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(0)
        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)

        bound = _matmul_jaxcallable_pin.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        helion_runtime._reset_jaxcallable_key_cache_hits()
        self.assertEqual(helion_runtime._jaxcallable_key_cache_hits(), 0)

        # First call seeds the JaxCallable cache (slow path / no counter bump).
        result = compiled_fn(x, y)
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(
            helion_runtime._jaxcallable_key_cache_hits(),
            0,
            "First call must miss the cached invocation key (no counter bump).",
        )

        # Subsequent calls share the same arg shape / dtype signature
        # so the subclass's fast-path short-circuit fires every time.
        n_repeats = 4
        for _ in range(n_repeats):
            result = compiled_fn(x, y)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

        self.assertEqual(
            helion_runtime._jaxcallable_key_cache_hits(),
            n_repeats,
            f"Repeat calls on a cached static-shape kernel must each "
            f"reuse the cached JaxCallable invocation key. Expected "
            f"{n_repeats} hits; got "
            f"{helion_runtime._jaxcallable_key_cache_hits()}.",
        )

    def test_pallas_call_custom_kernel_direct_hits_on_repeat_invocations(
        self,
    ) -> None:
        """Cached static-shape kernel routes through direct ``call_custom_kernel``.

        Deep Replan 5 2026-05-23 (§2.9 in ``plan.md``) measured that
        bypassing the ``JaxCallable`` wrapper on the launcher cache hot
        path saves another ~10us per call by skipping
        ``_HelionStaticJaxCallable.__call__``'s sig comparison, the
        ``list(args)`` allocation, and the JaxCallable method dispatch.
        On a static-shape kernel the launcher lifts a pre-captured
        ``_DirectCallKernel`` off the ``JaxCallable`` subclass after the
        first call and bumps
        ``helion.runtime._CALL_CUSTOM_KERNEL_DIRECT_HITS`` on every
        subsequent invocation that uses the direct-dispatch path.

        The test asserts the counter increments exactly ``n_repeats``
        times across ``1 + n_repeats`` consecutive calls of the same
        compiled callable: the first call seeds the JaxCallable's
        internal ``output_shapes`` cache plus the ``_helion_direct_call``
        snapshot; calls 2..N reuse them and the launcher fast path
        bypasses ``JaxCallable.__call__`` entirely.
        """
        from helion import runtime as helion_runtime

        # Define the kernel inside the test so the inner Helion-emitted
        # function (and its launcher cache) is unique to this test run —
        # no cross-test pollution from other tests that bind
        # ``pallas_matmul_bf16``.

        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_direct_pin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n],
                device=x.device,
                dtype=torch.promote_types(x.dtype, y.dtype),
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(0)
        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)

        bound = _matmul_direct_pin.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        helion_runtime._reset_call_custom_kernel_direct_hits()
        self.assertEqual(helion_runtime._call_custom_kernel_direct_hits(), 0)

        # First call seeds the launcher cache (slow path) and primes the
        # JaxCallable subclass's ``_helion_direct_call`` snapshot; no
        # direct-dispatch counter bump yet.
        result = compiled_fn(x, y)
        expected = (x.float() @ y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(
            helion_runtime._call_custom_kernel_direct_hits(),
            0,
            "First call must seed the cache, not hit the direct-dispatch path.",
        )

        # Subsequent calls share the same arg shape / dtype signature
        # so the launcher fast path lifts the direct-call kernel off the
        # JaxCallable subclass and routes around ``JaxCallable.__call__``
        # entirely.
        n_repeats = 4
        for _ in range(n_repeats):
            result = compiled_fn(x, y)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

        self.assertEqual(
            helion_runtime._call_custom_kernel_direct_hits(),
            n_repeats,
            f"Repeat calls on a cached static-shape kernel must each "
            f"route through ``tpu_torch_pallas.call_custom_kernel`` "
            f"directly. Expected {n_repeats} hits; got "
            f"{helion_runtime._call_custom_kernel_direct_hits()}.",
        )

    def test_pallas_call_custom_kernel_direct_matches_jaxcallable_output(
        self,
    ) -> None:
        """Direct ``call_custom_kernel`` dispatch is bitwise identical to JaxCallable.

        Deep Replan 5 2026-05-23 (§2.9 (d) in ``plan.md``) verified
        bitwise-identical output between a hand-built
        ``tpu_torch_pallas.call_custom_kernel`` invocation and the
        ``JaxCallable`` path on a bf16 matmul.  This pin replays the
        same correctness check inside the test suite: bind a
        static-shape bf16 matmul, run it once (warmup → routes through
        the slow path), then run it 3 more times (each one routes
        through the direct-dispatch path) and compare every output to
        the warmup output bit-for-bit (``equal=True``, not just close).

        Regression hazard pinned: any future refactor that subtly
        diverges the direct-call output from the JaxCallable output —
        e.g. dropping ``out_tree.unflatten``, skipping the alias
        copy-back, or passing the wrong ``donate_argnums`` — fails this
        test.
        """
        from helion import runtime as helion_runtime

        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_direct_correctness(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty(
                [m, n],
                device=x.device,
                dtype=torch.promote_types(x.dtype, y.dtype),
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(0)
        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(1)
        y = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)

        bound = _matmul_direct_correctness.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        helion_runtime._reset_call_custom_kernel_direct_hits()

        # First call: slow path (JaxCallable wrapper).  Saves the
        # reference output to compare against.
        reference = compiled_fn(x, y).clone()
        self.assertEqual(
            helion_runtime._call_custom_kernel_direct_hits(),
            0,
            "First call must be the slow path (JaxCallable wrapper).",
        )

        # Subsequent calls: direct ``call_custom_kernel`` dispatch.
        # Each output must be bitwise identical to the reference.
        for i in range(3):
            result = compiled_fn(x, y)
            self.assertTrue(
                torch.equal(result, reference),
                f"Direct-dispatch call {i + 1} output diverged from "
                f"the JaxCallable-path reference (max_abs_diff="
                f"{(result.float() - reference.float()).abs().max().item()}).",
            )
        self.assertEqual(
            helion_runtime._call_custom_kernel_direct_hits(),
            3,
            "Three repeat calls must each hit the direct-dispatch path.",
        )

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
        _code, result = code_and_output(pallas_bmm, (a, b))
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

    @xfailIfPallas("Non-zero begin K reduction: DMA offset not tile-aligned")
    def test_bmm_nonzero_k_begin(self) -> None:
        """BMM with K reduction starting at non-zero offset, across all loop types."""
        a = torch.randn(4, 128, 384, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4, 384, 128, device=DEVICE, dtype=torch.bfloat16)
        k_start, k_end = 128, 384
        expected = torch.bmm(
            a[:, :, k_start:k_end].float(), b[:, k_start:k_end, :].float()
        ).to(torch.bfloat16)
        for loop_type in ("unroll", "fori_loop", "emit_pipeline"):
            with self.subTest(pallas_loop_type=loop_type):
                _code, result = code_and_output(
                    pallas_bmm_subrange_k,
                    (a, b, k_start, k_end),
                    block_sizes=[4, 128, 128, 256],
                    pallas_loop_type=loop_type,
                )
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

    def _check_scalar_lookup_in_pipeline(self, loop_type: str) -> None:
        torch.manual_seed(0)
        x = torch.randn(256, device=DEVICE, dtype=torch.float32)
        # Run with several distinct bias vectors; each invocation's
        # observable output is the last program's read of biases[-1], so a
        # fresh value of biases[-1] per call exercises the dynamic SMEM
        # load with different runtime values rather than a fixed offset.
        for biases_list in (
            [1.0, 2.0, 3.0, 4.0],
            [-7.5, 11.0, 0.0, 1234.5],
            [100.0, -50.0, 25.0, -12.5],
        ):
            biases = torch.tensor(biases_list, device=DEVICE, dtype=torch.float32)
            out = torch.zeros_like(x)
            _code, result = code_and_output(
                pallas_scalar_lookup_in_pipeline,
                (biases, x, out),
                block_sizes=[64],
                pallas_loop_type=loop_type,
            )
            torch.testing.assert_close(
                result, x + biases[-1].item(), rtol=1e-5, atol=1e-5
            )

    def test_scalar_lookup_with_emit_pipeline(self) -> None:
        """``hl.grid`` outer + scalar lookup ``biases[g]`` + inner pipeline body
        runs end-to-end under ``pallas_loop_type='emit_pipeline'``.

        The scalar load index is per-program runtime, so ``biases`` has to
        live in SMEM — Mosaic rejects a dynamic vector load from a small
        VMEM ref because dim 0 isn't provably aligned to 128 lanes.
        """
        self._check_scalar_lookup_in_pipeline("emit_pipeline")

    def test_scalar_lookup_with_fori_loop(self) -> None:
        """Same kernel as :meth:`test_scalar_lookup_with_emit_pipeline`
        compiled under ``pallas_loop_type='fori_loop'``."""
        self._check_scalar_lookup_in_pipeline("fori_loop")

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

    def test_attention_unroll_fp32(self) -> None:
        """Test attention with unroll (for-loop) inner loop."""
        query = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        key = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        val = torch.randn(1, 4, 32, 64, dtype=torch.float32, device=DEVICE)
        args = (query, key, val)

        _code, result = code_and_output(
            pallas_attention,
            args,
            block_sizes=[1, 32, 32],
            pallas_loop_type="unroll",
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

        # test that we're not manually allocating and donating out tensor HBM,
        # but are instead taking over tensor returned by torch_tpu JaxCallable
        self.assertIn("out = torch.empty_like(q_view, device='meta')", _code)
        self.assertIn("out = _launcher(", _code)

    def test_hl_zeros_outer_arithmetic_emit_pipeline(self) -> None:
        """``hl.zeros`` results must support arithmetic at outer (non-inner-loop) scope.

        Regression test: ``acc = hl.zeros(...); acc += x`` written before an
        inner emit_pipeline / fori_loop must work.  Previously, the Pallas
        codegen for hl.zeros returned a bare VMEM scratch ref, so the outer
        ``acc + x`` emitted ``scratch + x`` and JAX raised
        ``'AbstractRef' object has no attribute '_add'`` at trace time.
        Inner-loop bodies dodged the issue via ``_remap_args_to_scratch``;
        outer scope had no equivalent remap.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, n], dtype=torch.float32)
                # Outer-scope arithmetic on the hl.zeros result with a
                # scalar.  Previously, this emitted ``scratch + 1.0`` and
                # JAX raised the AbstractRef ``_add`` error.
                acc += 1.0
                # Inner emit_pipeline forces the previously-buggy scratch
                # path inside ``hl.zeros`` codegen.
                for tile_k in hl.tile(n):
                    acc += x[tile_m, tile_k].to(torch.float32).sum(dim=-1, keepdim=True)
                out[tile_m, :] = acc.to(x.dtype)
            return out

        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            kernel,
            (x,),
            block_sizes=[32, 128],
            pallas_loop_type="emit_pipeline",
        )
        ref = 1.0 + x.sum(dim=-1, keepdim=True).expand(-1, 128)
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

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

    def test_attention_emit_pipeline_correctness_head_dim_256(self) -> None:
        """Test emit_pipeline attention pre-broadcast with head_dim > PRE_BROADCAST_SIZE."""
        query = torch.randn(2, 2, 128, 256, dtype=torch.float32, device=DEVICE)
        key = torch.randn(2, 2, 128, 256, dtype=torch.float32, device=DEVICE)
        val = torch.randn(2, 2, 128, 256, dtype=torch.float32, device=DEVICE)
        code, result = code_and_output(
            pallas_attention,
            (query, key, val),
            block_sizes=[4, 128, 128],
            pallas_loop_type="emit_pipeline",
            pallas_pre_broadcast=True,
        )
        # m_i and l_i scratches get pre-broadcast trailing dim 128;
        # acc scratch keeps head_dim=256
        self.assertIn(
            "_scratch_shapes=["
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 256), 'jnp.float32', 'vmem')]",
            code,
        )
        self.assertIn("jnp.tile(", code)
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_attention_fori_loop_correctness_head_dim_256(self) -> None:
        """Test fori_loop attention pre-broadcast with head_dim > PRE_BROADCAST_SIZE."""
        query = torch.randn(2, 2, 128, 256, dtype=torch.float32, device=DEVICE)
        key = torch.randn(2, 2, 128, 256, dtype=torch.float32, device=DEVICE)
        val = torch.randn(2, 2, 128, 256, dtype=torch.float32, device=DEVICE)
        args = (query, key, val)
        code, result = code_and_output(
            pallas_attention,
            args,
            block_sizes=[4, 128, 128],
            pallas_loop_type="fori_loop",
            pallas_pre_broadcast=True,
        )
        self.assertIn("jax.lax.fori_loop", code)
        # m_i and l_i scratches get pre-broadcast trailing dim 128;
        # acc scratch keeps head_dim=256; extra entries are DMA buffers/semaphores
        self.assertIn(
            "_scratch_shapes=["
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 128), 'jnp.float32', 'vmem'), "
            "((4, 128, 256), 'jnp.float32', 'vmem'), "
            "((4, 128, 256), 'jnp.float32', 'vmem'), "
            "((), None, 'dma_semaphore'), "
            "((4, 128, 256), 'jnp.float32', 'vmem'), "
            "((), None, 'dma_semaphore')]",
            code,
        )
        self.assertIn("jnp.tile(", code)
        ref = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), val.float().cpu()
        ).to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_pre_broadcast_indirect_consumer(self) -> None:
        """Pre-broadcast tile must propagate through indirect consumers.

        When a pre-broadcast node (2D, trailing dim 128) feeds an intermediate
        op (e.g. running + 1.0, rsqrt) before reaching a wider-dim consumer
        (e.g. acc * scale where acc has head_dim=256), the tile-insertion pass
        must tile the intermediate result, not just direct pre-broadcast nodes.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def outer_chain_scale(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
                scale = torch.rsqrt(running[:, :, None] + 1.0)
                out[tile_b, tile_m, :] = (acc * scale).to(out.dtype)
            return out

        def ref_outer_chain_scale(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # With k=128 and block_k=128, there's 1 tile iteration:
            # running = sum(a, dim=-1), acc = running[:,:,None] (broadcast to 256)
            running = a.sum(-1)
            acc = running[:, :, None].expand(-1, -1, b.shape[-1]).clone()
            scale = torch.rsqrt(running[:, :, None] + 1.0)
            return (acc * scale).to(a.dtype)

        a = torch.rand(4, 64, 128, dtype=torch.float32, device=DEVICE)
        b = torch.rand(4, 64, 256, dtype=torch.float32, device=DEVICE)
        code, result = code_and_output(
            outer_chain_scale,
            (a, b),
            block_sizes=[4, 64, 128],
            pallas_loop_type="fori_loop",
            pallas_pre_broadcast=True,
        )
        ref = ref_outer_chain_scale(a, b)
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

    def test_symnode_index_in_emit_pipeline_body(self) -> None:
        """A SymInt expression derived from an outer tile index
        (e.g. ``tile_h.begin // 2``, as in GQA's
        ``h_idx // num_groups``) must be usable as an index inside an
        inner ``hl.tile`` loop on the Pallas backend.
        """

        @helion.kernel(static_shapes=True)
        def k(x: torch.Tensor) -> torch.Tensor:
            H, N = x.size()
            out = torch.empty_like(x)
            for tile_h in hl.tile(H, block_size=1):
                h_kv = tile_h.begin // 2
                for tile_n in hl.tile(N):
                    out[tile_h.begin, tile_n] = x[h_kv, tile_n]
            return out

        x = torch.randn(4, 8, dtype=torch.float32, device=DEVICE)
        code_and_output(k, (x,))

    def test_tile_index_broadcast_mask(self) -> None:
        """Two 1D ``tile.index`` tensors must broadcast into a 2D mask via
        ``[:, None]`` / ``[None, :]`` indexing — the natural PyTorch idiom
        for causal / sliding-window mask construction.
        """

        @helion.kernel(static_shapes=True)
        def k(x: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N):
                    mask = tile_m.index[:, None] >= tile_n.index[None, :]
                    out[tile_m, tile_n] = torch.where(
                        mask, x[tile_m, tile_n], torch.zeros_like(x[tile_m, tile_n])
                    )
            return out

        x = torch.randn(8, 8, dtype=torch.float32, device=DEVICE)
        _, result = code_and_output(k, (x,))
        idx = torch.arange(8, device=DEVICE)
        ref = torch.where(idx[:, None] >= idx[None, :], x, torch.zeros_like(x))
        torch.testing.assert_close(result, ref)

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

    @xfailIfPallasInterpret("numerical mismatch in JAX interpret mode")
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

    @xfailIfPallasInterpret("numerical mismatch in JAX interpret mode")
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
        # 4 chunks of 64 rows, 128 columns
        x = torch.randn(256, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_chunked_add, (x,), block_sizes=[128])
        expected = x + 1.0
        torch.testing.assert_close(result, expected)
        # tile_k.index + offset uses TileIndexWithOffsetPattern — the
        # pl.multiple_of hint should NOT be applied to offset expressions
        self.assertNotIn("pl.multiple_of(", code)

    @xfailIfPallasInterpret("numerical mismatch in JAX interpret mode")
    def test_tile_index_with_symbolic_offset_emit_pipeline(self) -> None:
        """Same kernel under pallas_loop_type='emit_pipeline'.

        emit_pipeline must emit offset_<bid>/indices_<bid> in the body
        prologue so kernel code that references tile.index sees defined
        symbols.  Without the prologue emission, the body raises
        ``NameError: name 'indices_2' is not defined`` at trace time.
        """
        x = torch.randn(256, 128, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            pallas_chunked_add,
            (x,),
            block_sizes=[128],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(result, x + 1.0)

    def test_tile_index_with_symbolic_offset_fori_loop(self) -> None:
        """Same kernel under pallas_loop_type='fori_loop'.

        fori_loop has the same prologue gap as emit_pipeline: without
        unconditional offset_<bid>/indices_<bid> emission, kernels that
        reference tile.index inside a divisible inner loop raise
        ``NameError: name 'indices_2' is not defined`` at trace time.
        """
        x = torch.randn(256, 128, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            pallas_chunked_add,
            (x,),
            block_sizes=[128],
            pallas_loop_type="fori_loop",
        )
        torch.testing.assert_close(result, x + 1.0)

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

    @xfailIfPallasTpu(
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

    @xfailIfPallasTpu("Pallas TPU not correctly handling tile index with offset")
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

    def test_unroll_loop_multidim_non_divisible(self) -> None:
        """Unroll loop with 2D inner loop where both dims are non-divisible.

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
            pallas_loop_type="unroll",
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
        # Block size 64 < 128 alignment — hint should NOT be applied
        self.assertNotIn("pl.multiple_of(", code)
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

    def test_full_slice_matches_non_power_of_two_factory_dim(self) -> None:
        """Non-pow2 full slices must match concrete factory-created dims."""

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

        x = torch.randn(64, 384, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            per_block_reduction,
            (x,),
            block_sizes=[8, 8],
            pallas_loop_type="fori_loop",
        )
        ref = x.view(8, 8, 384).sum(1)
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

    @xfailIfPallasInterpret(
        "JAX interpret cannot trace dynamic shapes (TypeError: JitTracer ~int32[])"
    )
    def test_emit_pipeline_per_tensor_pipelined_mixed(self) -> None:
        """An emit_pipeline body can mix pipelined and non-pipelined tensors.

        Aligned tensors pass through ``pltpu.emit_pipeline``'s ``pl.Buffered``
        BlockSpecs, while unaligned ones stay on the outer pallas_call
        BlockSpec and are closure-read from the body via ``pl.ds``.
        """
        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        r = torch.randn(64, 1, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_row_scale_mul,
            (x, r),
            block_sizes=[8],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertIn("pl.ds(", code)
        torch.testing.assert_close(result, x * r)

    def test_no_pipeline_outer_inner_shared_dim(self) -> None:
        """Don't pipeline a tensor whose dim is shared between outer and inner tiles.

        Regression test: when a kernel reads a tensor at outer scope using
        an outer block_id (e.g. ``T[tile_m, tile_n]``) and *also* inside an
        inner emit_pipeline / fori_loop using a different inner block_id on
        the same dim (e.g. ``T[tile_m, tile_k]``), the kernel needs outer
        ``pl.ds`` slicing for the shared dim.  Pipelining the tensor turns
        it into an HBM ref, which can't be sliced with ``pl.ds`` -- the
        body then either crashes or generates the wrong offset.  The
        classifier (shared between both inner-loop codegens) must keep
        such tensors on the outer BlockSpec.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = x[tile_m, tile_n].to(torch.float32)  # outer-scope use of n
                # inner loop shares x's n dim with the outer tile via a
                # different block_id -> x's n dim has both tile_n_bid
                # (outer) and tile_k_bid (inner).
                for tile_k in hl.tile(n):
                    acc += x[tile_m, tile_k].to(torch.float32).sum(dim=-1, keepdim=True)
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        expected = x + x.sum(dim=1, keepdim=True)
        for loop_type, loop_marker in (
            ("emit_pipeline", "pltpu.emit_pipeline"),
            ("fori_loop", "jax.lax.fori_loop"),
        ):
            with self.subTest(pallas_loop_type=loop_type):
                code, result = code_and_output(
                    fn,
                    (x,),
                    block_sizes=[32, 128, 128],
                    pallas_loop_type=loop_type,
                )
                self.assertIn(loop_marker, code)
                self.assertNotIn("_pipeline_arg_indices=[0", code)
                torch.testing.assert_close(result, expected)

    def test_no_pipeline_outer_summary_read(self) -> None:
        """Don't pipeline a tensor that's read at outer scope as a per-row
        summary, even when no inner block_id appears alongside an outer/grid
        block_id on any dim of the tensor.

        Outer scope reads ``T[tile_m, :]`` to compute a per-row summary;
        inner loop reads ``T[tile_m, tile_k]`` for per-tile work.  Pipelining
        T would replace its outer BlockSpec with HBM, and the outer-scope
        ``T[tile_m, :]`` load then fails with ``"Loads are only allowed on
        VMEM and SMEM references."``.  Companion to
        ``test_no_pipeline_outer_inner_shared_dim`` -- both exercise the
        outer-scope-access exclusion in ``_classify_pipelined_tensors`` but
        through different access patterns (this one uses ``:`` on the inner
        loop's dim; the other uses an outer-grid block_id on it).
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def fn(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            aux = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                # outer-scope read of T -- a per-row summary
                aux[tile_m] = T[tile_m, :].sum(dim=-1)
                for tile_k in hl.tile(n):
                    # inner-scope read of T -- per-tile elementwise work
                    out[tile_m, tile_k] = T[tile_m, tile_k] * x[tile_m, tile_k]
            return out

        T = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            fn,
            (T, x),
            block_sizes=[128, 128],
            pallas_loop_type="emit_pipeline",
        )
        # T (arg index 0) must NOT be pipelined — its outer-scope load
        # would otherwise hit HBM after the BlockSpec is replaced.
        self.assertIn("pltpu.emit_pipeline", code)
        self.assertNotIn("_pipeline_arg_indices=[0", code)
        torch.testing.assert_close(result, T * x, rtol=1e-3, atol=1e-3)

    def test_fori_loop_per_tensor_dma_mixed(self) -> None:
        """A fori_loop body can mix DMA-aligned and DMA-unaligned tensors.

        Aligned tensors take ``pltpu.make_async_copy`` scratch buffers; the
        unaligned tensor stays in its outer BlockSpec VMEM ref and is read
        via ``pl.ds``.
        """
        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        r = torch.randn(64, 1, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_row_scale_mul,
            (x, r),
            block_sizes=[8],
            pallas_loop_type="fori_loop",
        )
        self.assertIn("pltpu.make_async_copy", code)
        self.assertIn("pl.ds(", code)
        self.assertIn("_pipeline_arg_indices=", code)
        torch.testing.assert_close(result, x * r)

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

    def test_data_dependent_loop_bounds(self) -> None:
        """Data-dependent loop: hl.tile(0, n) where n comes from a tensor."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def data_dependent_sum(
            data: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            B = lengths.size(0)
            out = torch.zeros([B], dtype=data.dtype, device=data.device)
            for seg in hl.grid(B):
                n = lengths[seg]
                acc = hl.zeros([1], dtype=data.dtype)
                for tile in hl.tile(0, n):
                    acc = acc + data[tile].sum(dim=0).unsqueeze(0)
                out[seg] = acc.squeeze(0)
            return out

        N = 256
        B = 4
        data = torch.randn(N, device=DEVICE, dtype=torch.float32)
        lengths = torch.tensor([128, 256, 128, 256], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            data_dependent_sum,
            (data, lengths),
        )
        ref = torch.stack([data[: lengths[i]].sum() for i in range(B)])
        torch.testing.assert_close(result, ref, rtol=1e-4, atol=1e-4)

    @staticmethod
    def _non_zero_tile_begin_kernels() -> tuple[object, object]:
        @helion.kernel(backend="pallas", static_shapes=True)
        def sum_with_constant_offset(
            data: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            B = offsets.size(0) - 1
            out = torch.zeros([B], dtype=data.dtype, device=data.device)
            for seg in hl.grid(B):
                acc = hl.zeros([1], dtype=data.dtype)
                for tile in hl.tile(3, 128, block_size=16):
                    acc = acc + data[tile, :, :].sum(dim=0).sum(dim=0).sum(
                        dim=0
                    ).unsqueeze(0)
                out[seg] = acc.squeeze(0)
            return out

        @helion.kernel(backend="pallas", static_shapes=True)
        def sum_with_dynamic_offset(
            data: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            B = offsets.size(0) - 1
            out = torch.zeros([B], dtype=data.dtype, device=data.device)
            for seg in hl.grid(B):
                start = offsets[seg]
                end = offsets[seg + 1]
                acc = hl.zeros([1], dtype=data.dtype)
                for tile in hl.tile(start, end, block_size=16):
                    acc = acc + data[tile, :, :].sum(dim=0).sum(dim=0).sum(
                        dim=0
                    ).unsqueeze(0)
                out[seg] = acc.squeeze(0)
            return out

        return sum_with_constant_offset, sum_with_dynamic_offset

    def test_non_zero_tile_begin(self) -> None:
        """pl.ds() reads from a non-zero begin can overshoot the tensor boundary.

        Constant-bounds path is pinned to ``unroll``; dynamic-bounds path uses
        ``fori_loop`` via ``set_default``.  The emit_pipeline variant of the
        constant-bounds case is exercised as a separate xfail test below.
        """
        sum_with_constant_offset, sum_with_dynamic_offset = (
            self._non_zero_tile_begin_kernels()
        )
        N, A, B = 128, 8, 256
        data = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        offsets = torch.tensor([3, 128], device=DEVICE, dtype=torch.int32)
        ref = data[3:128].sum().unsqueeze(0)

        _code1, result1 = code_and_output(
            sum_with_constant_offset, (data, offsets), pallas_loop_type="unroll"
        )
        torch.testing.assert_close(result1, ref, rtol=1e-3, atol=1e-3)

        _code2, result2 = code_and_output(sum_with_dynamic_offset, (data, offsets))
        torch.testing.assert_close(result2, ref, rtol=1e-3, atol=1e-3)

    @xfailIfPallas(
        "emit_pipeline BlockSpec index_map drops the tile.begin offset, "
        "so a non-zero start in hl.tile(start, end, ...) reads from offset 0 "
        "instead and produces wrong results."
    )
    def test_non_zero_tile_begin_emit_pipeline(self) -> None:
        """Same kernel as ``test_non_zero_tile_begin`` but pinned to emit_pipeline.

        Documents the known emit_pipeline tile.begin bug.  Will start passing
        once the BlockSpec ``index_map`` is taught to include ``tile.begin``.
        """
        sum_with_constant_offset, _ = self._non_zero_tile_begin_kernels()
        N, A, B = 128, 8, 256
        data = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        offsets = torch.tensor([3, 128], device=DEVICE, dtype=torch.int32)
        ref = data[3:128].sum().unsqueeze(0)

        _code, result = code_and_output(
            sum_with_constant_offset, (data, offsets), pallas_loop_type="emit_pipeline"
        )
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

    @xfailIfPallasInterpret("numerical mismatch in JAX interpret mode")
    def test_dma_buffer_offset_nested_tile(self) -> None:
        """Inner loop reading outer-tiled tensor must use ':' not absolute offset."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def outer_in_inner(
            x: torch.Tensor, y: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            A = hl.specialize(x.size(1))
            B = hl.specialize(x.size(2))
            num_segs = offsets.size(0) - 1
            out = torch.zeros([num_segs, A, B], dtype=x.dtype, device=x.device)
            for seg in hl.grid(num_segs):
                start = offsets[seg]
                end = offsets[seg + 1]
                for tile_i in hl.tile(start, end):
                    for tile_j in hl.tile(start, end):
                        out[seg, :, :] = (
                            out[seg, :, :]
                            + x[tile_i, :, :].sum(dim=0)
                            + y[tile_j, :, :].sum(dim=0)
                        )
            return out

        N, A, B = 128, 8, 256
        x = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        y = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        offsets = torch.tensor([0, 64, 128], device=DEVICE, dtype=torch.int32)

        _code, result = code_and_output(
            outer_in_inner,
            (x, y, offsets),
            block_sizes=[32, 32],
            pallas_loop_type="fori_loop",
        )

        block = 32
        ref = torch.zeros(offsets.size(0) - 1, A, B, device=DEVICE, dtype=x.dtype)
        for seg in range(offsets.size(0) - 1):
            s, e = int(offsets[seg]), int(offsets[seg + 1])
            for i in range(0, e - s, block):
                for j in range(0, e - s, block):
                    ref[seg] += x[s + i : s + i + block].sum(dim=0) + y[
                        s + j : s + j + block
                    ].sum(dim=0)
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

    def test_jagged_sum_3d(self) -> None:
        """3D jagged sum with load-time masking for out-of-bounds data."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def jagged_sum_3d(
            x_data: torch.Tensor, x_offsets: torch.Tensor
        ) -> torch.Tensor:
            num_rows = x_offsets.size(0) - 1
            out = torch.zeros([num_rows], dtype=x_data.dtype, device=x_data.device)
            for seq_index in hl.grid(num_rows):
                start = x_offsets[seq_index]
                end = x_offsets[seq_index + 1]
                row_sums = hl.zeros([1], dtype=x_data.dtype)
                for tile in hl.tile(start, end):
                    vals = x_data[tile, :, :]
                    row_sums = row_sums + vals.sum(dim=0).sum(dim=0).sum(
                        dim=0
                    ).unsqueeze(0)
                out[seq_index] = row_sums.squeeze(0)
            return out

        num_segments, A, B, max_seqlen = 8, 8, 256, 64
        seq_lengths = torch.randint(
            1, max_seqlen + 1, (num_segments,), dtype=torch.int32
        )
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(seq_lengths, dim=0).to(torch.int32),
            ]
        ).to(DEVICE)
        N = int(x_offsets[-1])
        x_data = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            jagged_sum_3d,
            (x_data, x_offsets),
        )
        ref = torch.stack(
            [
                x_data[x_offsets[i] : x_offsets[i + 1], :, :].sum()
                for i in range(num_segments)
            ]
        )
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

    def test_nested_fori_loop_scratch_scoping(self) -> None:
        """Nested hl.tile(start, end) with inner accumulator"""

        @helion.kernel(backend="pallas", static_shapes=True)
        def nested_tile_sum(
            x: torch.Tensor, y: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            A = hl.specialize(x.size(1))
            B = hl.specialize(x.size(2))
            num_segs = offsets.size(0) - 1
            out = torch.zeros([num_segs, A, B], dtype=x.dtype, device=x.device)
            for seg in hl.grid(num_segs):
                start = offsets[seg]
                end = offsets[seg + 1]
                acc = hl.zeros([1, A, B], dtype=x.dtype)
                for tile_i in hl.tile(start, end):
                    inner_acc = hl.zeros([1, A, B], dtype=x.dtype)
                    for tile_j in hl.tile(start, end):
                        inner_acc = inner_acc + (x[tile_i, :, :] * y[tile_j, :, :]).sum(
                            dim=0
                        ).unsqueeze(0)
                    acc = acc + inner_acc
                out[seg, :, :] = acc.squeeze(0)
            return out

        N, A, B = 128, 8, 256
        x = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        y = torch.randn(N, A, B, device=DEVICE, dtype=torch.float32)
        offsets = torch.tensor([0, 64, 128], device=DEVICE, dtype=torch.int32)

        _code, result = code_and_output(
            nested_tile_sum,
            (x, y, offsets),
            block_sizes=[32, 32],
            pallas_loop_type="fori_loop",
        )

        block = 32
        ref = torch.zeros(offsets.size(0) - 1, A, B, device=DEVICE, dtype=x.dtype)
        for seg in range(offsets.size(0) - 1):
            s, e = int(offsets[seg]), int(offsets[seg + 1])
            for i in range(0, e - s, block):
                for j in range(0, e - s, block):
                    ref[seg] += (
                        x[s + i : s + i + block] * y[s + j : s + j + block]
                    ).sum(dim=0)
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)

    def test_nested_tile_matmul_mask_cast(self) -> None:
        """Two nested data-dependent tiles with matmul need float mask expansion."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def jagged_kernel(
            x: torch.Tensor, y: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            num_segs = offsets.size(0) - 1
            out = torch.zeros([num_segs], dtype=x.dtype, device=x.device)
            for seg in hl.grid(num_segs):
                start = offsets[seg]
                end = offsets[seg + 1]
                acc = hl.zeros([1], dtype=x.dtype)
                for tile_i in hl.tile(start, end):
                    for tile_j in hl.tile(start, end):
                        gram = torch.matmul(
                            x[tile_i, :], y[tile_j, :].transpose(-2, -1)
                        )
                        acc = acc + gram.sum(dim=0).sum(dim=0).unsqueeze(0)
                out[seg] = acc.squeeze(0)
            return out

        N, D = 128, 128
        x = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        y = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
        offsets = torch.tensor([0, 64, 128], device=DEVICE, dtype=torch.int32)

        _code, result = code_and_output(
            jagged_kernel,
            (x, y, offsets),
            block_sizes=[32, 32],
            pallas_loop_type="fori_loop",
        )

        ref = torch.zeros(offsets.size(0) - 1, device=DEVICE, dtype=x.dtype)
        for i in range(offsets.size(0) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ref[i] = (x[s:e] @ y[s:e].T).sum()
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)

    def test_if_branch_intermediate_outputs(self) -> None:
        """Branch intermediates must survive in _if output list."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def if_with_intermediate(x: torch.Tensor, flag: float) -> torch.Tensor:
            n, m = x.shape
            block_n = hl.register_block_size(n)
            block_m = hl.register_block_size(m)
            out = torch.empty([n], device=x.device, dtype=torch.float32)
            for tile_n in hl.tile(n, block_size=block_n):
                acc = hl.zeros([tile_n, block_m], dtype=torch.float32)
                for tile_m in hl.tile(m, block_size=block_m):
                    v = x[tile_n, tile_m]
                    if flag == 0.0:
                        doubled = v * 2
                        acc += doubled + 1
                    else:
                        acc += v
                out[tile_n] = torch.sum(acc, dim=1)
            return out

        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)

        # else-branch
        code, result = code_and_output(
            if_with_intermediate,
            (x, 0.5),
            block_sizes=[1, 64],
        )
        self.assertIn("lax.cond", code)
        torch.testing.assert_close(result, torch.sum(x, dim=1))

        # if-branch
        _, result = code_and_output(
            if_with_intermediate,
            (x, 0.0),
            block_sizes=[1, 64],
        )
        torch.testing.assert_close(result, torch.sum(x * 2 + 1, dim=1))

    def test_branch_nonlocal_write(self) -> None:
        """Branch intermediates must survive in _if output list."""

        @helion.kernel(backend="pallas", static_shapes=True, print_output_code=True)
        def fn(x: torch.Tensor, flag: float, coeffs: torch.Tensor) -> torch.Tensor:
            (n,) = x.shape
            block_n = hl.register_block_size(n)
            out = torch.empty([n], device=x.device, dtype=torch.float32)
            for tile_n in hl.tile(n, block_size=block_n):
                coeff_a = coeffs[0]
                coeff_b = coeffs[1]
                if flag == 1.0:
                    coeff_a = coeffs[2]
                    coeff_new = coeffs[3]
                else:
                    coeff_b = coeffs[4]
                    coeff_new = coeffs[5]
                out[tile_n] = x[tile_n] * coeff_a * coeff_b * coeff_new
            return out

        x = torch.ones(64, device=DEVICE, dtype=torch.float32)
        coeffs = torch.arange(6, device=DEVICE, dtype=torch.float32)

        # if-branch
        code, result = code_and_output(
            fn,
            (x, 1.0, coeffs),
            block_sizes=[64],
        )
        torch.testing.assert_close(result, x * coeffs[2] * coeffs[1] * coeffs[3])

        # else-branch
        code, result = code_and_output(
            fn,
            (x, 0.0, coeffs),
            block_sizes=[64],
        )
        torch.testing.assert_close(result, x * coeffs[0] * coeffs[4] * coeffs[5])

    def test_rand_add(self) -> None:
        """Test kernel using hl.rand (RNG ops) passes _rng_seed_buffer correctly.

        Regression test: the Pallas launcher previously inserted _rng_seed_buffer
        at position -1 (before _inplace_indices), which put it between
        _output_indices and _inplace_indices. The fix places _rng_seed_buffer
        before both _output_indices and _inplace_indices so the Pallas runtime
        receives arguments in the correct order.
        """
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(
            pallas_rand_add,
            (x, 42),
            block_sizes=[1024],
        )
        # Verify shape and dtype are correct
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        # The result should differ from x (random values added)
        self.assertFalse(torch.allclose(result, x))
        # All added random values should be in [0, 1), so result >= x and < x + 1
        self.assertTrue(torch.all(result >= x))
        self.assertTrue(torch.all(result < x + 1.0))

    def test_broadcast_mask_size1_first_dim(self) -> None:
        """Mask must not be applied to size-1 broadcast dims (first dim)."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [x.size(0), bias.size(0)],
                device=x.device,
                dtype=x.dtype,
            )
            for tile_m, tile_n in hl.tile(out.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_n]
            return out

        x = torch.randn(1, 10, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(10, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(k, (x, bias), block_sizes=[16, 16])
        expected = x + bias
        torch.testing.assert_close(result, expected)

    def test_broadcast_mask_size1_last_dim(self) -> None:
        """Mask must not be applied to size-1 broadcast dims (last dim)."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
            for tile_m, tile_n in hl.tile(out.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_n]

        x = torch.randn(10, 1, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(10, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(10, 10, device=DEVICE, dtype=torch.float32)
        code, _result = code_and_output(k, (x, bias, out), block_sizes=[16, 16])
        expected = x + bias[None, :]
        torch.testing.assert_close(out, expected)

    def test_broadcast_mask_size1_multiple_dims(self) -> None:
        """Mask must not be applied to size-1 broadcast dims (multiple dims)."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
            for tile_0, tile_1, tile_2 in hl.tile(out.size()):
                tmp0 = x[tile_0, tile_1, tile_2]
                tmp1 = bias[tile_2].unsqueeze(0).unsqueeze(0)
                out[tile_0, tile_1, tile_2] = tmp0 + tmp1.to(torch.float32)

        x = torch.randn(1, 10, 1, device=DEVICE, dtype=torch.float32)
        bias = torch.arange(10, device=DEVICE, dtype=torch.int32)
        out = torch.zeros(1, 10, 10, device=DEVICE, dtype=torch.float32)
        code, _result = code_and_output(k, (x, bias, out), block_sizes=[1, 16, 16])
        expected = x + bias.float()
        torch.testing.assert_close(out, expected)


@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasIndirectGather(TestCase):
    @staticmethod
    def _gather_2d_kernel(static_shapes: bool = True):
        @helion.kernel(backend="pallas", static_shapes=static_shapes)
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

    @parametrize("static_shapes", (True, False))
    def test_gather_fp32_uses_highest_precision(self, static_shapes: bool) -> None:
        gather = self._gather_2d_kernel(static_shapes=static_shapes)
        table = torch.randn(16, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather, (indices, table), block_sizes=[128, 64])
        self.assertIn("one_hot", code)
        self.assertIn("HIGHEST", code)
        if not static_shapes:
            self.assertIn(".shape[0]", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    @parametrize("static_shapes", (True, False))
    def test_gather_bf16_skips_highest(self, static_shapes: bool) -> None:
        gather = self._gather_2d_kernel(static_shapes=static_shapes)
        table = torch.randn(16, 64, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather, (indices, table), block_sizes=[128, 64])
        self.assertIn("one_hot", code)
        self.assertNotIn("HIGHEST", code)
        self.assertNotIn("astype(jnp.float32)", code)
        if not static_shapes:
            self.assertIn(".shape[0]", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    @parametrize("static_shapes", (True, False))
    def test_gather_2d_index_tile(self, static_shapes: bool) -> None:
        """Regression: 2D index tile must contract the last axis, not axis 1."""

        @helion.kernel(backend="pallas", static_shapes=static_shapes)
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
        if not static_shapes:
            self.assertIn(".shape[0]", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    @parametrize("static_shapes", (True, False))
    def test_gather_over_vmem_budget_raises(self, static_shapes: bool) -> None:
        """Table above VMEM budget fails fast with a clear message."""
        gather = self._gather_2d_kernel(static_shapes=static_shapes)
        table = torch.randn(65537, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 65537, (256,), device=DEVICE, dtype=torch.int32)
        if static_shapes:
            with self.assertRaisesRegex(Exception, "exceeds the .* VMEM threshold"):
                code_and_output(gather, (indices, table), block_sizes=[128, 64])
        else:
            # Dynamic shape tables bypass static VMEM check
            pass

    @parametrize("static_shapes", (True, False))
    def test_gather_vmem_budget_uses_block_size(self, static_shapes: bool) -> None:
        """Tiling broadcast dims shrinks the VMEM block.

        Full table is over the threshold but the resident block after tiling
        the broadcast dim fits, so the check must pass.
        """
        gather = self._gather_2d_kernel(static_shapes=static_shapes)
        # Full table = 8192 * 1024 * 4 = 32 MiB (over the 16 MiB limit).
        # Resident VMEM block with BE=256 = 8192 * 256 * 4 = 8 MiB, fits.
        table = torch.randn(8192, 1024, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 8192, (256,), device=DEVICE, dtype=torch.int32)
        code_and_output(gather, (indices, table), block_sizes=[128, 256])

    @parametrize("static_shapes", (True, False))
    def test_gather_int32_table_uses_select_reduce(self, static_shapes: bool) -> None:
        """Gather on int32 tables uses select-reduce instead of dot."""
        gather = self._gather_2d_kernel(static_shapes=static_shapes)
        table = torch.randint(0, 100, (16, 64), device=DEVICE, dtype=torch.int32)
        indices = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather, (indices, table), block_sizes=[128, 64])
        self.assertIn("one_hot", code)
        self.assertIn("dtype=jnp.int32", code)
        self.assertNotIn("dot_general", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    @parametrize("static_shapes", (True, False))
    def test_gather_int32_5d_table_broadcasts_mask(self, static_shapes: bool) -> None:
        """Gather on higher-rank int32 tables broadcasts the select mask."""

        @helion.kernel(backend="pallas", static_shapes=static_shapes)
        def gather(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [
                    indices.size(0),
                    table.size(1),
                    table.size(2),
                    table.size(3),
                    table.size(4),
                ],
                dtype=table.dtype,
                device=table.device,
            )
            for tile_b, tile_i, tile_j, tile_k, tile_l in hl.tile(
                [
                    indices.size(0),
                    table.size(1),
                    table.size(2),
                    table.size(3),
                    table.size(4),
                ]
            ):
                out[tile_b, tile_i, tile_j, tile_k, tile_l] = table[
                    indices[tile_b], tile_i, tile_j, tile_k, tile_l
                ]
            return out

        table = torch.randint(0, 100, (8, 2, 4, 4, 8), device=DEVICE, dtype=torch.int32)
        indices = torch.randint(0, 8, (8,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gather,
            (indices, table),
            block_sizes=[8, 2, 4, 4, 8],
        )
        self.assertEqual(code.count("expand_dims"), 4)
        self.assertNotIn("dot_general", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref)

    @parametrize("static_shapes", (True, False))
    def test_scatter_raises(self, static_shapes: bool) -> None:
        """Indirect store has no Pallas strategy; plan_tiling must raise."""

        @helion.kernel(backend="pallas", static_shapes=static_shapes)
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

    @parametrize("static_shapes", (True, False))
    def test_gather_1d_index_bumps_block_to_tpu_alignment(
        self, static_shapes: bool
    ) -> None:
        """Block size on a 1D int32 index must be bumped to 128."""
        gather = self._gather_2d_kernel(static_shapes=static_shapes)
        table = torch.randn(1024, 256, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 1024, (1024,), device=DEVICE, dtype=torch.int32)
        # If the bump didn't happen, the generated code would slice with
        # `pl.ds(offset_0, 8)`. That string must not appear.
        code, result = code_and_output(gather, (indices, table), block_sizes=[8, 64])
        self.assertNotIn("pl.ds(offset_0, 8)", code)
        ref = table.cpu()[indices.long().cpu()].to(device=DEVICE)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)


instantiate_parametrized_tests(TestPallasIndirectGather)


class TestPallasPrinter(TestCase):
    def test_pallas_texpr_mod(self) -> None:
        """pallas_texpr must handle Mod/PythonMod/FloorDiv (used by bh % heads)."""
        import sympy
        from torch.utils._sympy.functions import FloorDiv
        from torch.utils._sympy.functions import PythonMod

        from helion._compiler.device_function import pallas_texpr

        x, y = sympy.symbols("x y")
        self.assertEqual(pallas_texpr(PythonMod(x, y)), "(x % y)")
        self.assertEqual(pallas_texpr(FloorDiv(x, y)), "(x // y)")
        self.assertEqual(
            pallas_texpr(FloorDiv(x, y) + PythonMod(x, y)),
            "((x // y)) + ((x % y))",
        )


if __name__ == "__main__":
    unittest.main()
