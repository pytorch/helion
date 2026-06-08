from __future__ import annotations

import ast
import math
import os
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
import unittest

from examples.geglu import _geglu_pallas as _geglu_pallas_example
from examples.swiglu import _swiglu_fwd_pallas as _swiglu_fwd_pallas_example
import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfPallasInterpret
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallas
from helion._testing import xfailIfPallasInterpret
from helion._testing import xfailIfPallasTpu
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.base_search import PopulationBasedSearch
    from helion.autotuner.base_search import PopulationMember

# N-D-tiled Pallas geglu/swiglu (#2725), re-wrapped on the pallas backend so the
# example kernels get real correctness coverage under pallas interpret / TPU CI.
_geglu_pallas = helion.kernel(
    _geglu_pallas_example.fn, backend="pallas", static_shapes=True
)
_swiglu_fwd_pallas = helion.kernel(
    _swiglu_fwd_pallas_example.fn, backend="pallas", static_shapes=True
)


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
def pallas_matmul_bf16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16 matmul kernel mirroring the perf harness's helion variant.

    Used by ``test_pallas_matmul_bf16_no_tiling_seed_covers_large_cubes`` to
    exercise the no-tiling ``lax.dot_general`` lowering on bf16 square matmuls.
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
def pallas_scatter_store(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values)
    for tile_m, tile_n in hl.tile(values.size()):
        out[indices[tile_m], tile_n] = values[tile_m, tile_n]
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
def pallas_jagged_segment_add(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Outer grid over jagged segments + an inner ``hl.tile(start, end)`` loop
    whose begin (``offsets[g]``) is an arbitrary runtime offset. A
    block-aligned BlockSpec index can only address starts that are multiples of
    the block size, so the emit_pipeline path must slice the segment with a
    dynamic ``pl.ds`` (``pl.BoundedSlice`` block)."""
    out = torch.empty_like(x)
    for g in hl.grid(offsets.size(0) - 1):
        start = offsets[g]
        end = offsets[g + 1]
        for tile in hl.tile(start, end):
            out[tile, :] = x[tile, :] + 1.0
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


@helion.kernel(backend="pallas", static_shapes=True)
def kernel_output_index_remapping(
    x: torch.Tensor,  # [batch*heads, seq_len, head_dim]
    batch: int,
    heads: int,
) -> torch.Tensor:
    """Reshapes a [batch*heads, seq_len, head_dim] tensor to [batch, heads, seq_len, head_dim].

    Iterates over the combined batch*heads dimension and the seq_len dimension.
    """
    batch_heads, seq_len, head_dim = x.size()
    out = torch.empty([batch, heads, seq_len, head_dim], dtype=x.dtype, device=x.device)
    for bh in hl.grid(batch_heads):
        b = bh // heads
        h = bh % heads
        for tile_m in hl.tile(seq_len):
            out[b, h, tile_m, :] = x[bh, tile_m, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def kernel_tile_index_is_blockwise(
    x: torch.Tensor,
) -> torch.Tensor:
    seq_len = x.size(0)
    out = torch.empty_like(x)
    for tile_m in hl.tile(seq_len):
        out[tile_m.index] = x[tile_m.index] + 1.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def kernel_tile_begin_plus_offset_is_elementwise(
    x: torch.Tensor,
) -> torch.Tensor:
    seq_len = x.size(0)
    out = torch.zeros_like(x)
    for tile_m in hl.tile(seq_len):
        out[tile_m.begin + 5] = x[tile_m.begin + 5] + 1.0
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

    def test_output_index_remapping_in_pipeline(self) -> None:
        total_elements = 8 * 128 * 128
        x = torch.arange(total_elements, device=DEVICE, dtype=torch.bfloat16).view(
            8, 128, 128
        )
        batch = 2
        heads = 4
        code, result = code_and_output(
            kernel_output_index_remapping,
            (x, batch, heads),
            block_sizes=[32],
            pallas_loop_type="emit_pipeline",
        )
        expected = x.reshape(batch, heads, 128, 128)

        with self.subTest(name="correctness"):
            torch.testing.assert_close(result, expected)

        with self.subTest(name="pipeline_emit"):
            self.assertIn("pltpu.emit_pipeline", code)

        with self.subTest(name="shrunken_blockspec"):
            self.assertIn(
                "pl.BlockSpec((1, 1, _BLOCK_SIZE_1, 128), "
                "lambda _j: (offset_0 // heads, offset_0 % heads, _j, 0)",
                code,
            )

        with self.subTest(name="body_vmem_indices"):
            self.assertIn("out_vmem[0, 0, :, :]", code)

    def test_output_index_remapping_in_fori_loop(self) -> None:
        total_elements = 8 * 128 * 128
        x = torch.arange(total_elements, device=DEVICE, dtype=torch.bfloat16).view(
            8, 128, 128
        )
        batch = 2
        heads = 4
        code, result = code_and_output(
            kernel_output_index_remapping,
            (x, batch, heads),
            block_sizes=[32],
            pallas_loop_type="fori_loop",
        )

        with self.subTest(name="correctness"):
            expected = x.reshape(batch, heads, 128, 128)
            torch.testing.assert_close(result, expected)

        with self.subTest(name="fori_loop_emit"):
            self.assertIn("jax.lax.fori_loop", code)

        with self.subTest(name="body_vmem_indices"):
            self.assertIn("out_buf[0, 0, :, :]", code)

        with self.subTest(name="vmem_shape_allocation"):
            self.assertIn("((1, 1, 32, 128), 'jnp.bfloat16', 'vmem')", code)

        with self.subTest(name="hbm_dma_slices"):
            self.assertIn("pl.ds(symnode_0, 1), pl.ds(symnode_1, 1)", code)

    def test_pipeline_kernel_tile_index_is_blockwise(self) -> None:
        x = torch.arange(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            kernel_tile_index_is_blockwise,
            (x,),
            block_sizes=[256],
            pallas_loop_type="emit_pipeline",
        )
        torch.testing.assert_close(result, x + 1.0)
        self.assertNotIn("pltpu.emit_pipeline", code)
        self.assertIn("out[:]", code)

    def test_pipeline_kernel_tile_begin_plus_offset_is_elementwise(self) -> None:
        x = torch.arange(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            kernel_tile_begin_plus_offset_is_elementwise,
            (x,),
            block_sizes=[256],
            pallas_loop_type="emit_pipeline",
        )
        expected = torch.zeros_like(x)
        expected[5::256] = x[5::256] + 1.0
        torch.testing.assert_close(result, expected)
        self.assertNotIn("pltpu.emit_pipeline", code)
        self.assertIn("_smem_arg_indices", code)
        self.assertIn("out[5]", code)

    def test_add_1d(self) -> None:
        args = (torch.randn(1024, device=DEVICE), torch.randn(1024, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=256)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add_large(self) -> None:
        args = (torch.randn(4096, device=DEVICE), torch.randn(4096, device=DEVICE))
        code, result = code_and_output(add_kernel, args, block_size=512)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_geglu_pallas_nd(self) -> None:
        # N-D-tiled GEGLU (#2725): correctness on the pallas backend.
        a = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        b = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(_geglu_pallas, (a, b), block_sizes=[16, 32])
        expected = torch.nn.functional.gelu(a, approximate="tanh") * b
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_swiglu_pallas_nd(self) -> None:
        # N-D-tiled SwiGLU (#2725): correctness on the pallas backend.
        a = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        b = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(_swiglu_fwd_pallas, (a, b), block_sizes=[16, 32])
        expected = torch.nn.functional.silu(a) * b
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

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

    def test_wrapper_gather_before_loop_is_read_only_input(self) -> None:
        """A tensor created by eager wrapper code and only read by Pallas is not output."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def gather_then_tile(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            gathered = x[idx]
            out = torch.empty_like(gathered)
            for tile in hl.tile(out.size()):
                out[tile] = gathered[tile] + 1.0
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        idx = torch.arange(127, -1, -1, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather_then_tile, (x, idx), block_sizes=[128])
        torch.testing.assert_close(result, x[idx] + 1.0)
        self.assertIn("_output_indices=[1]", code)
        self.assertIn("_inplace_indices=[]", code)

    def test_wrapper_gather_and_scatter_around_loop(self) -> None:
        """Eager prologue gather and epilogue scatter compose around Pallas."""

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def gather_tile_scatter(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            gathered = x[idx]
            out = torch.empty_like(gathered)
            for tile in hl.tile(out.size()):
                out[tile] = gathered[tile] + 1.0
            scattered = torch.empty_like(out)
            scattered[idx] = out
            return scattered

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        idx = torch.arange(127, -1, -1, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gather_tile_scatter, (x, idx), block_sizes=[128])
        expected = torch.empty_like(x)
        expected[idx] = x[idx] + 1.0
        torch.testing.assert_close(result, expected)
        self.assertIn("_inplace_indices=[]", code)

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

    def test_bool_view_expand_where(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def pallas_bool_view_expand_where(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                mask = x[tile_m, 0] > 0
                mask_2d = mask.view(tile_m.block_size, 1).expand(
                    tile_m.block_size, tile_n.block_size
                )
                out[tile_m, tile_n] = torch.where(mask_2d, x[tile_m, tile_n], 0.0)
            return out

        x = torch.randn(16, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_bool_view_expand_where,
            (x,),
            block_sizes=[16, 128],
        )

        expected = torch.where(x[:, :1] > 0, x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)
        self.assertIn("astype(jnp.int32)", code)

    def test_indirect_gather_with_tiled_dim(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def pallas_indirect_gather_with_tiled_dim(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty([indices.size(0), values.size(1)], device=values.device)
            for tile_m, tile_n in hl.tile(out.size()):
                out[tile_m, tile_n] = values[indices[tile_m], tile_n]
            return out

        values = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
        indices = torch.randperm(16, device=DEVICE).to(torch.int32)
        code, result = code_and_output(
            pallas_indirect_gather_with_tiled_dim,
            (values, indices),
            block_sizes=[4, 4],
        )

        torch.testing.assert_close(result, values[indices.to(torch.int64), :])
        self.assertIn("values[:,", code)

    def test_scatter_store(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                values = torch.randn(16, 8, device=DEVICE, dtype=dtype)
                indices = torch.randperm(16, device=DEVICE).to(torch.int32)
                code, result = code_and_output(
                    pallas_scatter_store, (values, indices), block_sizes=[4, 4]
                )

                expected = torch.zeros_like(values)
                expected[indices.to(torch.int64)] = values
                torch.testing.assert_close(result, expected)
                self.assertIn("one_hot", code)
                self.assertIn("jnp.triu", code)
                self.assertIn("jnp.eye", code)
                self.assertIn("jnp.swapaxes", code)
                self.assertIn("jnp.ones_like", code)
                self.assertIn("jnp.where", code)
                self.assertIn("dot_general", code)

    def test_scatter_store_duplicate_indices(self) -> None:
        values = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor(
            [0, 1, 1, 2, 4, 4, 4, 8, 8, 9, 10, 10, 12, 13, 14, 14],
            device=DEVICE,
            dtype=torch.int32,
        )
        code, result = code_and_output(
            pallas_scatter_store, (values, indices), block_sizes=[16, 8]
        )

        expected = torch.zeros_like(values)
        expected[indices.to(torch.int64)] = values
        torch.testing.assert_close(result, expected)

    def test_tensor_index_atomic_add_raises(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def atomic_add_tensor_index(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros_like(values)
            for tile in hl.tile(values.size(0)):
                hl.atomic_add(out, [indices[tile]], values[tile])
            return out

        values = torch.randn(16, device=DEVICE, dtype=torch.float32)
        indices = torch.randperm(16, device=DEVICE).to(torch.int32)

        with self.assertRaisesRegex(
            NotImplementedError,
            "tensor-indexed memory op is not supported for op=atomic_add",
        ):
            code_and_output(
                atomic_add_tensor_index,
                (values, indices),
                block_size=16,
            )

    def test_scatter_store_multiple_tensor_indices_raises(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True)
        def scatter_store_multiple_tensor_indices(
            values: torch.Tensor, row_indices: torch.Tensor, col_indices: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros(
                [values.size(0), values.size(0)],
                dtype=values.dtype,
                device=values.device,
            )
            for tile in hl.tile(values.size(0)):
                out[row_indices[tile], col_indices[tile]] = values[tile, tile]
            return out

        values = torch.randn(16, 16, device=DEVICE, dtype=torch.float32)
        row_indices = torch.randperm(16, device=DEVICE).to(torch.int32)
        col_indices = torch.randperm(16, device=DEVICE).to(torch.int32)

        with self.assertRaisesRegex(
            NotImplementedError,
            "multiple indirect dims are not supported",
        ):
            code_and_output(
                scatter_store_multiple_tensor_indices,
                (values, row_indices, col_indices),
                block_size=16,
            )

    def test_inplace_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        expected = x + y
        # Use block_size=1024 so grid=1; with grid>1 the full-array
        # access pattern causes inplace mutations to accumulate.
        code, result = code_and_output(pallas_inplace_add, (x, y), block_size=1024)
        # x should be mutated in place
        torch.testing.assert_close(x, expected)

    def test_shared_output_disjoint_rows(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, autotune_effort="none")
        def pallas_shared_output_disjoint_rows(x: torch.Tensor) -> torch.Tensor:
            for row in hl.grid(2):
                x[row, :] = x[row, :] + (row + 10)
            return x

        x = torch.zeros([2, 128], device=DEVICE, dtype=torch.float32)
        expected = torch.stack(
            [
                torch.full([128], 10.0, device=DEVICE),
                torch.full([128], 11.0, device=DEVICE),
            ]
        )
        code, result = code_and_output(pallas_shared_output_disjoint_rows, (x,))
        torch.testing.assert_close(result, expected)

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

    def test_pallas_launcher_fast_path_hits_on_repeat_invocations(self) -> None:
        """Repeat calls on a cached static-shape kernel take the launcher fast path.

        On the first call the launcher seeds a grid-keyed cache on the inner
        device function; every later call hits ``cache is not None and
        cache[0] == grid`` and reuses the precomputed ``_LauncherFastPath``
        instead of recomputing the per-call dtype check + ds-pad + output-only
        loop.  The launcher branches deterministically on the cache, so a
        populated cache after the first call means the fast path is taken; the
        test asserts that and that the output stays correct.
        """

        # Define the kernel inside the test so its launcher cache -- which lives
        # on the inner generated function, not the decorator object -- is unique
        # to this run, avoiding cross-test pollution.
        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_launcher_fast_path_pin(
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

        bound = _matmul_launcher_fast_path_pin.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        expected = (x.float() @ y.float()).to(torch.bfloat16)

        # First call seeds the launcher cache; subsequent calls take the fast
        # path off it.  Output must stay correct throughout.
        for _ in range(5):
            result = compiled_fn(x, y)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

        # The launcher stores its grid-keyed cache on the inner device-kernel
        # object (``_pallas_cache`` / ``_pallas_pipeline_cache`` /
        # ``_pallas_fori_cache``, depending on which launcher the config
        # selects), reachable via the compiled function's module globals.  A
        # populated cache means repeat calls took the fast-path branch.
        cache_attrs = ("_pallas_cache", "_pallas_pipeline_cache", "_pallas_fori_cache")
        cached = [
            value
            for value in compiled_fn.__globals__.values()
            if any(getattr(value, a, None) is not None for a in cache_attrs)
        ]
        self.assertTrue(
            cached,
            "Repeat calls on a cached static-shape kernel must populate the "
            "launcher fast-path cache on the inner device function.",
        )

    @skipIfPallasInterpret(
        "direct call_custom_kernel dispatch is torch_tpu/TPU-only; the "
        "_DirectCallKernel snapshot is not built in JAX interpret mode"
    )
    def test_pallas_call_custom_kernel_direct_matches_jaxcallable_output(
        self,
    ) -> None:
        """Direct ``call_custom_kernel`` dispatch must produce bitwise-identical
        output to the JaxCallable path on bf16 matmul (pin against silent
        divergence from a refactor)."""

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

        # First call: slow path (JaxCallable wrapper).  Saves the reference.
        reference = compiled_fn(x, y).clone()

        # Subsequent calls: direct ``call_custom_kernel`` dispatch.  Each output
        # must be bitwise identical to the reference.
        for i in range(3):
            result = compiled_fn(x, y)
            self.assertTrue(
                torch.equal(result, reference),
                f"Direct-dispatch call {i + 1} output diverged from the "
                f"JaxCallable-path reference (max_abs_diff="
                f"{(result.float() - reference.float()).abs().max().item()}).",
            )

        # Confirm the direct-call snapshot was actually built (slot 5 of the
        # launcher cache), so the equality above exercised the direct path and
        # is not a trivial slow-path-vs-slow-path comparison.
        cache_attrs = ("_pallas_cache", "_pallas_pipeline_cache", "_pallas_fori_cache")
        caches = [
            getattr(value, a)
            for value in compiled_fn.__globals__.values()
            for a in cache_attrs
            if getattr(value, a, None) is not None
        ]
        self.assertTrue(
            caches and caches[0][5] is not None,
            "Repeat calls must build the _DirectCallKernel snapshot "
            "(direct-dispatch path engaged), not stay on the slow path.",
        )

    @skipIfPallasInterpret(
        "direct call_custom_kernel dispatch is torch_tpu/TPU-only; the "
        "_DirectCallKernel snapshot is not built in JAX interpret mode"
    )
    def test_pallas_direct_call_sig_check_locks_on_static_shapes(self) -> None:
        """Repeat direct-dispatch calls flip ``_DirectCallKernel.sig_locked`` to
        ``True``, eliding the per-call sig check on a static-shape kernel."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_sig_lock_pin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

        bound = _matmul_sig_lock_pin.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        reference = compiled_fn(x, y).clone()
        for _ in range(5):
            result = compiled_fn(x, y)
            self.assertTrue(
                torch.equal(result, reference),
                "Sig-locked direct-dispatch call output diverged "
                "(max_abs_diff="
                f"{(result.float() - reference.float()).abs().max().item()}).",
            )

        # Slot 5 of the launcher cache holds the _DirectCallKernel snapshot.
        cache_attrs = ("_pallas_cache", "_pallas_pipeline_cache", "_pallas_fori_cache")
        caches = [
            getattr(value, a)
            for value in compiled_fn.__globals__.values()
            for a in cache_attrs
            if getattr(value, a, None) is not None
        ]
        self.assertTrue(caches, "Launcher cache was not populated.")
        direct_call = caches[0][5]
        self.assertIsNotNone(direct_call, "Direct-call snapshot was not built.")
        self.assertTrue(
            direct_call.sig_locked,
            "Repeat direct-dispatch calls must flip sig_locked to True.",
        )

    def test_pallas_launcher_caches_output_tensor(self) -> None:
        """Static-shape kernel caches the output ``device='meta'`` placeholder and
        reuses the same object across calls (no per-call re-allocation)."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_output_meta_pin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

        bound = _matmul_output_meta_pin.bind((x, y))
        config = bound.config_spec.default_config()
        compiled_fn = bound.compile_config(config)

        reference = compiled_fn(x, y).clone()
        owners = [
            value
            for value in compiled_fn.__globals__.values()
            if getattr(value, "_helion_output_meta_cache_0", None) is not None
        ]
        self.assertTrue(owners, "Output meta placeholder was not cached.")
        cached_meta = owners[0]._helion_output_meta_cache_0

        # Repeat calls must reuse the same placeholder object and keep the output.
        n_repeats = 10
        for _ in range(n_repeats):
            result = compiled_fn(x, y)
            self.assertTrue(
                torch.equal(result, reference),
                "Output diverged after cached meta-placeholder reuse "
                f"(max_abs_diff="
                f"{(result.float() - reference.float()).abs().max().item()}).",
            )
            self.assertIs(
                owners[0]._helion_output_meta_cache_0,
                cached_meta,
                "Repeat calls must reuse the cached placeholder, not re-allocate.",
            )

        # Bitwise-identical to a fresh-compiled baseline: the cache holds only the
        # zero-storage meta placeholder, not output bytes.
        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_output_meta_baseline(
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

        bound_baseline = _matmul_output_meta_baseline.bind((x, y))
        baseline_fn = bound_baseline.compile_config(
            bound_baseline.config_spec.default_config()
        )
        baseline_result = baseline_fn(x, y)
        self.assertTrue(
            torch.equal(reference, baseline_result),
            "Cached-meta result diverged from fresh-compiled baseline "
            f"(max_abs_diff="
            f"{(reference.float() - baseline_result.float()).abs().max().item()}).",
        )

    def test_pallas_autotuner_final_pick_picks_true_best_on_noisy_initial_rank(
        self,
    ) -> None:
        """Final-pick re-ranks past a noisy initial measurement.

        ``[512, 1024, 512]`` looks fastest on its noisy initial ``perf`` but
        rebenches slower than ``[512, 512, 512]``, so
        ``run_final_pick_verification`` must re-pick ``[512, 512, 512]``.
        """
        from unittest.mock import patch

        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.base_search import PopulationMember
        from helion.runtime.config import Config

        def member(block_sizes: list[int], noisy_ms: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda *a, **kw: None,
                perfs=[noisy_ms],
                flat_values=block_sizes,
                config=Config(block_sizes=block_sizes),
                status="ok",
                compile_time=0.0,
            )

        # noisy_best wins the noisy initial rank (0.220) but rebenches slowest
        # (0.232); true_best looks slower initially (0.232) but is truly fastest
        # (0.180).  true_ms is what the rebenchmark reveals.
        noisy_best = member([512, 1024, 512], 0.220)
        true_best = member([512, 512, 512], 0.232)
        true_ms = {id(noisy_best): 0.232, id(true_best): 0.180}

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.population = [noisy_best, true_best]
        search.best_perf_so_far = min(m.perf for m in search.population)
        search.log = lambda *a, **kw: None  # pyrefly: ignore[bad-assignment]

        def fake_rebenchmark(
            to_bench: list[PopulationMember], *, desc: str = ""
        ) -> None:
            for m in to_bench:
                m.perfs.append(true_ms[id(m)])

        with patch.object(search, "rebenchmark", side_effect=fake_rebenchmark):
            final = search.run_final_pick_verification(noisy_best, top_k=5)
        self.assertEqual(list(final.config["block_sizes"]), [512, 512, 512])

    def test_pallas_matmul_bf16_no_tiling_seed_covers_large_cubes(self) -> None:
        """No-tiling seed fires on each bf16 cube in ``_PALLAS_NO_TILING_DIMS``.

        Per cube N, ``PallasMatmulNoTilingSeedHeuristic`` is eligible and plants
        the ``[N, N, N] unroll pb=True`` compiler seed; a cube outside the set
        (256) is refused so the seed stays scoped to ablation-validated shapes.
        """
        from helion._compiler.autotuner_heuristics.pallas import _PALLAS_NO_TILING_DIMS
        from helion._compiler.autotuner_heuristics.pallas import (
            PallasMatmulNoTilingSeedHeuristic,
        )

        self.assertEqual(sorted(_PALLAS_NO_TILING_DIMS), [1024, 2048, 4096])

        for dim in sorted(_PALLAS_NO_TILING_DIMS):
            x = torch.empty(dim, dim, device=DEVICE, dtype=torch.bfloat16)
            y = torch.empty(dim, dim, device=DEVICE, dtype=torch.bfloat16)
            bound = pallas_matmul_bf16.bind((x, y))

            self.assertTrue(
                PallasMatmulNoTilingSeedHeuristic.is_eligible(
                    bound.env, bound.host_function.device_ir
                ),
                f"heuristic must fire on bf16 {dim}-cube",
            )
            seeded = [
                (
                    tuple(cfg.config.get("block_sizes", ())),
                    cfg.config.get("pallas_loop_type"),
                    cfg.config.get("pallas_pre_broadcast"),
                )
                for cfg in bound.config_spec.compiler_seed_configs
            ]
            self.assertIn(
                ((dim, dim, dim), "unroll", True),
                seeded,
                f"compiler seeds must include the no-tiling entry on bf16 {dim}-cube",
            )

        # 256-cube is outside the set, so the heuristic must refuse it.
        small_x = torch.empty(256, 256, device=DEVICE, dtype=torch.bfloat16)
        small_y = torch.empty(256, 256, device=DEVICE, dtype=torch.bfloat16)
        small_bound = pallas_matmul_bf16.bind((small_x, small_y))
        self.assertFalse(
            PallasMatmulNoTilingSeedHeuristic.is_eligible(
                small_bound.env, small_bound.host_function.device_ir
            ),
            "heuristic must refuse cubes outside _PALLAS_NO_TILING_DIMS",
        )

    def test_pallas_autotuner_compiler_seed_survives_final_pick(self) -> None:
        """Compiler-seeded members are re-considered during final-pick.

        The search prunes a seed that looked average on its noisy initial bench;
        ``capture_compiler_seed_members`` snapshots it and merges it back into the
        final-pick pool so it is re-benched against the search's best.
        """
        from unittest.mock import patch

        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.base_search import PopulationMember
        from helion.runtime.config import Config

        # Last-gen survivors (~0.205 ms) plus a compiler seed that scored a noisy
        # 0.215 ms initially (so the search dropped it) but rebenches at a
        # true-fastest 0.190 ms.  (config, noisy initial ms, true rebench ms):
        last_gen = [
            (Config(block_sizes=[1024, 256, 1024]), 0.205, 0.204),
            (Config(block_sizes=[256, 256, 256]), 0.210, 0.209),
        ]
        compiler_seed = (
            Config(
                block_sizes=[512, 512, 512],
                pallas_loop_type="emit_pipeline",
                pallas_pre_broadcast=False,
            ),
            0.215,
            0.190,
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

        last_gen_members = [make_member(cfg, noisy) for cfg, noisy, _true in last_gen]
        seed_member = make_member(compiler_seed[0], compiler_seed[1])
        true_perf_by_id = {
            id(m): true
            for m, (_cfg, _noisy, true) in zip(last_gen_members, last_gen, strict=True)
        }
        true_perf_by_id[id(seed_member)] = compiler_seed[2]

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.population = last_gen_members  # seed NOT in last-gen population
        search.best_perf_so_far = min(m.perf for m in last_gen_members)
        search._compiler_seed_members = [seed_member]
        search.log = lambda *a, **kw: None  # pyrefly: ignore[bad-assignment]

        def fake_rebenchmark(
            members: list[PopulationMember], *, desc: str = ""
        ) -> None:
            for member in members:
                member.perfs.append(true_perf_by_id[id(member)])

        with patch.object(search, "rebenchmark", side_effect=fake_rebenchmark):
            initial_best = min(last_gen_members, key=lambda m: m.perf)
            self.assertEqual(
                list(initial_best.config["block_sizes"]),
                [1024, 256, 1024],
                "Precondition: search's running best is one of the last-gen members.",
            )
            final_best = search.run_final_pick_verification(initial_best, top_k=10)

        self.assertEqual(
            list(final_best.config["block_sizes"]),
            [512, 512, 512],
            "Compiler-seeded [512, 512, 512] must be re-benched and re-rank "
            "ahead of the last-gen best once its true 0.190 ms perf is measured.",
        )

    def _make_device_micros_search(
        self,
        specs: list[tuple[str, list[int], float, float]],
    ) -> tuple[PopulationBasedSearch, dict[str, PopulationMember]]:
        """Device-µs final-pick scaffold from ``(key, block_sizes, wall_ms,
        device_micros)`` specs, with a fake Pallas backend reporting the scripted
        device µs.
        """
        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.base_search import PopulationMember
        from helion.runtime.config import Config

        def _new_fn() -> Callable[..., object]:
            def _fn() -> None:
                return None

            return _fn

        device_micros_by_fn: dict[int, float] = {}
        members: dict[str, PopulationMember] = {}
        for key, block_sizes, wall_ms, device_micros in specs:
            fn = _new_fn()
            cfg = Config(block_sizes=block_sizes)
            device_micros_by_fn[id(fn)] = device_micros
            members[key] = PopulationMember(
                fn=fn,
                perfs=[wall_ms],
                flat_values=[id(cfg)],
                config=cfg,
                status="ok",
                compile_time=0.0,
            )

        def fake_device_micros_bench(
            fns: list[Callable[..., object]],
            reference_fn: Callable[..., object],
            *,
            desc: str | None = None,
        ) -> list[tuple[float, float]]:
            ref = device_micros_by_fn[id(getattr(reference_fn, "func", reference_fn))]
            vals = [device_micros_by_fn[id(getattr(fn, "func", fn))] for fn in fns]
            return [(v, v - ref) for v in vals]

        class _Backend:
            @staticmethod
            def get_paired_device_micros_bench() -> Callable[
                ..., list[tuple[float, float]]
            ]:
                return fake_device_micros_bench

        class _ConfigSpec:
            backend = _Backend()

        class _Settings:
            autotune_benchmark_fn: Callable[..., list[float]] | None = None
            autotune_progress_bar: bool = False
            static_shapes: bool = True

        class _Kernel:
            class env:
                process_group_name: str | None = None

        class _BenchProvider:
            mutated_arg_indices: tuple[int, ...] = ()

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.population = list(members.values())
        search.best_perf_so_far = min(m.perf for m in search.population)
        search.log = lambda *a, **kw: None  # pyrefly: ignore[bad-assignment]
        search.args = ()
        search.kernel = _Kernel()  # pyrefly: ignore[bad-assignment]
        search.benchmark_provider = _BenchProvider()  # pyrefly: ignore[bad-assignment]
        search.settings = _Settings()  # pyrefly: ignore[bad-assignment]
        search.config_spec = _ConfigSpec()  # pyrefly: ignore[bad-assignment]
        search._compiler_seed_members = []
        return search, members

    def test_pallas_autotuner_final_pick_reranks_by_device_micros(self) -> None:
        """Final-pick ranks by on-device µs, not wall-clock.

        ``fast`` is 10µs on-device but 125µs wall-clock; ``slow`` is 30µs
        on-device but 124µs wall-clock (1µs "faster"). The re-rank must pick
        ``fast`` despite its slower wall-clock.
        """
        from unittest.mock import patch

        search, m = self._make_device_micros_search(
            [
                ("fast", [1024, 1024, 1024], 0.125, 10.0),
                ("slow", [128, 1024, 1024], 0.124, 30.0),
            ]
        )
        with patch.dict(os.environ, {"HELION_AUTOTUNE_PALLAS_RANK_BY": "device_time"}):
            final = search.run_final_pick_verification(m["slow"], top_k=5)
        self.assertEqual(list(final.config["block_sizes"]), [1024, 1024, 1024])

    @skipIfPallasInterpret(
        "device-µs ranking needs a real TPU; CPU-interpret has no /device:TPU events"
    )
    def test_pallas_paired_device_micros_bench_finite_on_large_compute_bound_shape(
        self,
    ) -> None:
        """``paired_device_micros_bench`` stays finite on a 4096³ compute-bound shape.

        Guards the ``count >= _MIN_TRACE_EVENTS`` predicate (vs an exact
        ``== n_calls``): on large shapes the ``stop_trace`` flush drops a few tail
        events, so an exact match would return ``+inf`` and silently route the
        autotuner to its wall-clock fallback. Two structurally-identical jit_fns
        must yield finite device µs and a near-zero paired delta.
        """
        import jax
        import jax.numpy as jnp

        from helion.autotuner.benchmarking import _pallas_device_micros_for_fn
        from helion.autotuner.benchmarking import paired_device_micros_bench

        m = k = n = 4096
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
        y = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16)

        @jax.jit
        def matmul(a: object, b: object) -> object:
            return jax.lax.dot_general(a, b, dimension_numbers=(((1,), (0,)), ((), ())))

        def _device_micros_fn(fn: Callable[[], object]) -> float:
            return _pallas_device_micros_for_fn(fn, n_calls=50, n_warmup=2)

        results = paired_device_micros_bench(
            [lambda: matmul(x, y)],
            lambda: matmul(x, y),
            device_micros_fn=_device_micros_fn,
        )
        median_micros, delta_micros = results[0]
        self.assertTrue(
            math.isfinite(median_micros) and median_micros > 0,
            f"candidate device µs must be finite + positive on 4096³; got {median_micros!r}",
        )
        self.assertTrue(
            math.isfinite(delta_micros),
            f"paired delta must be finite on 4096³; got {delta_micros!r}",
        )
        self.assertLess(
            abs(delta_micros),
            5.0,
            f"identical jit_fns should give a near-zero paired delta; got {delta_micros!r}",
        )

    def test_pallas_matmul_dot_general_lowering_fires_on_no_tiling(self) -> None:
        """No-tiling 2-input matmul emits ``lax.dot_general``, not ``pl.pallas_call``.

        Spies on ``_build_matmul_dot_general_jit_fn``: it runs once on first
        compile (not on cache-hit repeats) and the output matches the
        ``pl.pallas_call`` path.
        """
        from unittest.mock import patch

        from helion import runtime as helion_runtime
        from helion.runtime.config import Config

        @helion.kernel(backend="pallas", static_shapes=True)
        def _matmul_dot_general_pin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

        # Force the no-tiling config (block_sizes match input dims).
        bound = _matmul_dot_general_pin.bind((x, y))
        no_tiling_cfg = Config(block_sizes=[256, 256, 256])

        with patch.object(
            helion_runtime,
            "_build_matmul_dot_general_jit_fn",
            wraps=helion_runtime._build_matmul_dot_general_jit_fn,
        ) as build_spy:
            compiled_fn = bound.compile_config(no_tiling_cfg)
            result_no_tiling = compiled_fn(x, y)
            for _ in range(3):
                compiled_fn(x, y)
        self.assertEqual(
            build_spy.call_count,
            1,
            "Pure matmul + no-tiling config must lower via ``lax.dot_general`` "
            "exactly once (first cache-build); cache hits must not rebuild.",
        )

        # Tiled config must keep the pl.pallas_call path (builder not run), and
        # its output must match the no-tiling dot_general path within bf16 tol.
        bound_ref = _matmul_dot_general_pin.bind((x, y))
        tiled_cfg = Config(block_sizes=[128, 128, 128])
        with patch.object(
            helion_runtime,
            "_build_matmul_dot_general_jit_fn",
            wraps=helion_runtime._build_matmul_dot_general_jit_fn,
        ) as build_spy_tiled:
            compiled_ref = bound_ref.compile_config(tiled_cfg)
            result_tiled = compiled_ref(x, y)
        self.assertEqual(
            build_spy_tiled.call_count,
            0,
            "tiled config (block_size < input_dim) must not run the dot_general builder",
        )
        max_abs_diff = (
            (result_no_tiling.float() - result_tiled.float()).abs().max().item()
        )
        self.assertLess(
            max_abs_diff,
            5e-2,
            f"dot_general output diverged from pallas_call by {max_abs_diff}",
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

    @xfailIfPallasInterpret(
        "dynamic pl.ds / pl.BoundedSlice BlockSpecs are not supported by JAX's "
        "Pallas interpret mode (concrete-shape requirement); runs on real TPU."
    )
    def test_emit_pipeline_data_dependent_begin_uses_dynamic_ds(self) -> None:
        """A data-dependent ``hl.tile(start, end)`` inner loop under
        ``pallas_loop_type='emit_pipeline'`` must address the jagged segment
        with a dynamic ``pl.ds`` slice + ``pl.BoundedSlice`` block, not a
        block-aligned ``// block_size`` index (which only addresses starts that
        are exact multiples of the block size). Regression test for the
        "emit_pipeline fails on unaligned dims" limitation.
        """
        # Arbitrary, deliberately non-block-aligned segment bounds covering
        # [0, 48) so the output is fully written (out = x + 1 everywhere).
        offsets = torch.tensor([0, 10, 31, 48], device=DEVICE, dtype=torch.int32)
        x = torch.randn(48, 128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_jagged_segment_add,
            (x, offsets),
            block_sizes=[16],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        # The fix: dynamic ds slice + BoundedSlice block for the runtime begin.
        self.assertIn("pl.BoundedSlice", code)
        self.assertIn("pl.ds(", code)
        # Load/store asymmetry: the input spec reads a full block (the over-read
        # past ``end`` is zeroed by the mask), while the output spec clamps its
        # extent to min(block, end - offset) so a short final tile writes only
        # its valid rows instead of overrunning into the next segment.
        in_part, _, out_part = code.partition("out_specs=")
        self.assertIn("in_specs=", in_part)
        self.assertNotIn("jnp.minimum", in_part)  # input: full block
        self.assertIn("jnp.minimum", out_part)  # output: clamped extent
        torch.testing.assert_close(result, x + 1.0, rtol=1e-5, atol=1e-5)

    def test_emit_pipeline_static_begin_keeps_block_index(self) -> None:
        """Control: a static (zero-begin) inner loop must NOT switch to the
        dynamic ds/BoundedSlice path, so aligned/static kernels are unchanged.
        """
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
        self.assertNotIn("pl.BoundedSlice", code)
        torch.testing.assert_close(result, args[0] + args[1])

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

    @xfailIfPallasInterpret(
        "pl.program_id captured into emit_pipeline body is not supported in "
        "JAX interpret mode (program_id_p.bind asserts during trace)"
    )
    def test_nested_non_grid_outer_loop_emit_pipeline(self) -> None:
        """Grid (``tile_m``) → non-grid device loop (``tile_n``) wrapping
        an inner emit_pipeline (``tile_k``) whose body reads
        ``w[tile_k, tile_n]`` compiles and produces correct matmul output.
        Mirrors the epilogue structure of ``squeeze_and_excitation_net``.

        ``n`` must exceed the effective ``bs_n`` (128 lanes on TPU) so the
        inner BlockSpec for ``w`` is actually block-sized rather than
        coincidentally equalling the full ``n`` dim.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            n = w.size(1)
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
                    out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        m, k, n = 32, 256, 256
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        w = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            kernel,
            (x, w),
            block_sizes=[16, 128, 128],
            pallas_loop_type="emit_pipeline",
        )
        self.assertIn("pltpu.emit_pipeline", code)
        torch.testing.assert_close(result, x @ w, rtol=1e-2, atol=1e-2)

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
        self.assertIn("torch.empty_like(q_view, device='meta')", _code)
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

    @skipIfPallasInterpret("SMEM preload copy is too expensive in JAX interpret mode")
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

    def test_scalar_access_hl_grid_inplace(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=True, config=helion.Config())
        def fn(x: torch.Tensor) -> torch.Tensor:
            for i in hl.grid(x.size(0)):
                x[i] = x[i] + 1
            return x

        x = torch.arange(128, device=DEVICE, dtype=torch.float32)
        expected = x + 1
        result = fn(x)
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

    @skipIfPallasInterpret(
        "2D SMEM preload copy is too expensive in JAX interpret mode"
    )
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

    @skipIfPallasInterpret(
        "2D SMEM preload copy is too expensive in JAX interpret mode"
    )
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

    def test_pipeline_begin_aligned_skips_pad(self) -> None:
        # A block-aligned inner begin (the outer tile's offset) needs no boundary
        # pad, so _ds_pad_dims must report extra_pad == 0 rather than block_size-1.
        # Locks the pad-skip optimization (would be block_size-1 if it regressed).
        x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        r = torch.randn(64, 1, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            pallas_row_scale_mul,
            (x, r),
            block_sizes=[8],
            pallas_loop_type="fori_loop",
        )
        torch.testing.assert_close(result, x * r)
        match = re.search(r"_ds_pad_dims=(\[[^\]]*\])", code)
        self.assertIsNotNone(match, "expected _ds_pad_dims in the launcher call")
        pad_dims = ast.literal_eval(
            match.group(1)
        )  # [(arg, dim, block_size, extra_pad)]
        self.assertTrue(pad_dims, "expected pl.ds pad dims to be present")
        self.assertTrue(
            all(extra_pad == 0 for *_, extra_pad in pad_dims),
            f"block-aligned begin should skip the pad (extra_pad==0), got {pad_dims}",
        )

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

    @xfailIfPallasInterpret(
        "emit_pipeline now includes tile.begin via a dynamic pl.ds BlockSpec, "
        "but JAX's Pallas interpret mode does not support dynamic pl.ds / "
        "pl.BoundedSlice (concrete-shape requirement). Expected to pass on real "
        "TPU; xfail only under interpret."
    )
    def test_non_zero_tile_begin_emit_pipeline(self) -> None:
        """Same kernel as ``test_non_zero_tile_begin`` but pinned to emit_pipeline.

        The non-zero ``tile.begin`` is now carried by a dynamic ``pl.ds``
        BlockSpec index_map, so this produces correct results on TPU. It still
        xfails under JAX Pallas interpret, which cannot execute dynamic
        ``pl.ds`` / ``pl.BoundedSlice`` BlockSpecs.
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

    def test_inner_tile_alignment_propagates_to_outer(self) -> None:
        """Inner tile alignment min must propagate to the bounding outer tile."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x: torch.Tensor) -> torch.Tensor:
            M = x.size(0)
            out = torch.empty_like(x)
            m_block = hl.register_block_size(M)
            for tile_outer in hl.tile(M, block_size=m_block):
                for tile_inner in hl.tile(tile_outer.begin, tile_outer.end):
                    out[tile_inner, :] = x[tile_inner, :] * 2
            return out

        args = (torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16),)
        spec = k.bind(args).config_spec
        outer_min = spec.block_sizes[0].min_size
        inner_min = spec.block_sizes[1].min_size
        self.assertGreaterEqual(outer_min, inner_min)

    def test_jagged_tile_pins_parent_block_size_to_1(self) -> None:
        """A jagged_tile's parent (items axis) is pinned to block_size=1 on
        Pallas — each program owns exactly one item so the per-item DMA slice
        + chunk_mask emission can use program_id directly as the row index.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            M = x_data.size(1)
            num_rows = x_offsets.size(0) - 1
            out = torch.zeros(
                [num_rows, M], dtype=x_data.dtype, device=x_data.device
            )
            x_flat = x_data.view(-1)
            for tile_b in hl.tile(num_rows):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                for tile_m in hl.tile(M):
                    row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
                    for tile_k in hl.jagged_tile(nnz):
                        base = starts[:, None] + tile_k.index[None, :]
                        flat = base[:, :, None] * M + tile_m.index[None, None, :]
                        row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
                    out[tile_b, tile_m] = row_sums
            return out

        x_offsets = torch.tensor([0, 3, 8, 10, 14], dtype=torch.int32)
        x_data = torch.randn(14, 8, dtype=torch.float32)
        spec = k.bind((x_data, x_offsets)).config_spec
        # tile_b is the items axis (parent of the jagged tile_k). On Pallas
        # it must be pinned to exactly 1.
        parent_spec = spec.block_sizes[0]
        self.assertEqual(parent_spec.min_size, 1)
        self.assertEqual(parent_spec.max_size, 1)

    def test_jagged_tile_parent_pin_survives_alignment_propagation(self) -> None:
        """The pin sticks even when the parent indexes into a tensor that
        would otherwise raise its alignment min via
        ``PallasBackend.adjust_block_size_constraints``.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            M = x_data.size(1)
            num_rows = x_offsets.size(0) - 1
            out = torch.zeros(
                [num_rows, M], dtype=x_data.dtype, device=x_data.device
            )
            x_flat = x_data.view(-1)
            # bfloat16 lane → 256-element alignment requirement on Pallas;
            # without our skip, tile_b's min would be raised above 1.
            for tile_b in hl.tile(num_rows):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                for tile_m in hl.tile(M):
                    row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
                    for tile_k in hl.jagged_tile(nnz):
                        base = starts[:, None] + tile_k.index[None, :]
                        flat = base[:, :, None] * M + tile_m.index[None, None, :]
                        row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
                    out[tile_b, tile_m] = row_sums
            return out

        x_offsets = torch.tensor([0, 3, 8, 10, 14], dtype=torch.int32)
        x_data = torch.randn(14, 8, dtype=torch.bfloat16)
        spec = k.bind((x_data, x_offsets)).config_spec
        self.assertEqual(spec.block_sizes[0].min_size, 1)
        self.assertEqual(spec.block_sizes[0].max_size, 1)

    def test_non_jagged_kernel_does_not_pin_outer_to_1(self) -> None:
        """Regression: the jagged-parent pin must only apply in jagged kernels.
        A regular ``hl.tile(...)``-only kernel should keep its autotuned
        outer block-size range untouched.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2
            return out

        args = (torch.randn([1024], device=DEVICE, dtype=torch.float32),)
        spec = k.bind(args).config_spec
        # The outer tile is the only block in this kernel — its max should
        # be much larger than 1 (the autotuner gets to pick from the full
        # power-of-2 range bounded by the tensor size).
        self.assertGreater(spec.block_sizes[0].max_size, 1)

    def test_jagged_kernel_emits_grid_1_and_fori_loop_wrapper(self) -> None:
        """Pallas jagged kernel must launch as ``grid=(1,)`` and wrap the
        kernel body in ``jax.lax.fori_loop(0, num_items, _kernel_body, None)``
        with ``pid_0`` as the loop fn's iteration parameter — not
        ``pl.program_id(0)``.
        """

        @helion.kernel(backend="pallas", static_shapes=True)
        def k(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            M = x_data.size(1)
            num_rows = x_offsets.size(0) - 1
            out = torch.zeros(
                [num_rows, M], dtype=x_data.dtype, device=x_data.device
            )
            x_flat = x_data.view(-1)
            for tile_b in hl.tile(num_rows):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                for tile_m in hl.tile(M):
                    row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
                    for tile_k in hl.jagged_tile(nnz):
                        base = starts[:, None] + tile_k.index[None, :]
                        flat = base[:, :, None] * M + tile_m.index[None, None, :]
                        row_sums = row_sums + hl.load(x_flat, [flat]).sum(dim=1)
                    out[tile_b, tile_m] = row_sums
            return out

        x_offsets = torch.tensor([0, 3, 8, 10, 14], dtype=torch.int32)
        x_data = torch.randn(14, 8, dtype=torch.float32)
        bound = k.bind((x_data, x_offsets))
        code = bound.to_triton_code(bound.config_spec.default_config())

        # grid collapses to (1,): the single Pallas program iterates items
        # inside via fori_loop.  The launcher receives grid as its second
        # positional arg, so check the call site shape.
        self.assertRegex(code, r"_launcher\(\s*_helion_\w+\s*,\s*\(1,\)")
        # The body is wrapped in a fori_loop whose body fn declares pid_0
        # as the iteration variable.
        self.assertRegex(code, r"def\s+_kernel_body\w*\s*\(\s*pid_0\s*,\s*_\s*\)\s*:")
        self.assertRegex(
            code, r"jax\.lax\.fori_loop\s*\(\s*0\s*,.+_kernel_body\w*\s*,\s*None\s*\)"
        )
        # No ``pl.program_id(0)`` at the top of the body — pid_0 must come
        # from the fori_loop body fn parameter, not the host launcher.
        self.assertNotRegex(code, r"pid_0\s*=\s*pl\.program_id\s*\(\s*0\s*\)")
        # The jagged-flat tensor (x_flat) AND the output tensor (out) must
        # both be marked HBM: x_flat because it's larger than VMEM; out
        # because the whole-tensor VMEM BlockSpec would OOM at realistic
        # output sizes.  Arg order is (x_offsets, x_flat, out) → x_flat at
        # position 1, out at position 2.
        self.assertRegex(code, r"_pipeline_arg_indices=\[\s*1\s*,\s*2\s*\]")
        # x_offsets must live in SMEM so dynamic ``x_offsets[pl.ds(pid_0, 1)]``
        # reads from inside the @pl.loop body don't hit VMEM sublane/lane
        # alignment.  It's at launcher position 0.
        self.assertRegex(code, r"_smem_arg_indices=\[\s*0\s*\]")

    def test_parse_flat_jagged_subscript_canonical(self) -> None:
        """``_parse_flat_jagged_subscript`` recovers
        ``(sublane_bid, sublane_base_fx, lane_bid, M)`` from the canonical
        flat-1D form ``(starts + tile_k.idx) * M + tile_m.idx`` (with
        broadcast wrappers)."""
        import operator
        from types import SimpleNamespace

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx import Graph
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from helion._compiler.pallas.plan_tiling import (
            _parse_flat_jagged_subscript,
        )

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            k_sym = shape_env.create_unbacked_symint()
            m_sym = shape_env.create_unbacked_symint()
            torch._check(k_sym >= 1)
            torch._check(m_sym >= 1)
        bid_of = {id(k_sym): 7, id(m_sym): 8}
        env = SimpleNamespace(
            get_block_id=lambda s: bid_of.get(id(s)),
            is_jagged_tile=lambda bid: bid == 7,
            jagged_tile_parent_ids={7: [3]},
        )

        # Build: add(unsqueeze(mul(unsqueeze(add(starts, k_idx), -1), 64), 0),
        #            unsqueeze(unsqueeze(m_idx, 0), 0))
        g = Graph()
        starts = g.placeholder("starts")
        k_idx = g.placeholder("tile_k_idx")
        k_idx.meta["val"] = k_sym
        m_idx = g.placeholder("tile_m_idx")
        m_idx.meta["val"] = m_sym

        inner = g.call_function(operator.add, args=(starts, k_idx))
        inner_u = g.call_function(
            torch.ops.aten.unsqueeze.default, args=(inner, -1)
        )
        mul = g.call_function(operator.mul, args=(inner_u, 64))
        mul_u = g.call_function(torch.ops.aten.unsqueeze.default, args=(mul, 0))
        m_u1 = g.call_function(torch.ops.aten.unsqueeze.default, args=(m_idx, 0))
        m_u2 = g.call_function(torch.ops.aten.unsqueeze.default, args=(m_u1, 0))
        flat = g.call_function(operator.add, args=(mul_u, m_u2))
        g.output(flat)

        result = _parse_flat_jagged_subscript(flat, env)
        self.assertIsNotNone(result)
        sublane_bid, sublane_base_fx, lane_bid, lane_size = result
        self.assertEqual(sublane_bid, 7)
        self.assertIs(sublane_base_fx, starts)
        self.assertEqual(lane_bid, 8)
        self.assertEqual(lane_size, 64)

    def test_parse_flat_jagged_subscript_non_canonical_returns_none(self) -> None:
        """If the subscript doesn't match the canonical structure (e.g. a
        bare add of two tile indices, missing the ``* M`` factor), the
        parser returns None and the caller falls through to plain
        ``TensorIndexPattern`` (the indirect-gather path)."""
        import operator
        from types import SimpleNamespace

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx import Graph
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from helion._compiler.pallas.plan_tiling import (
            _parse_flat_jagged_subscript,
        )

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            k_sym = shape_env.create_unbacked_symint()
            m_sym = shape_env.create_unbacked_symint()
            torch._check(k_sym >= 1)
            torch._check(m_sym >= 1)
        bid_of = {id(k_sym): 7, id(m_sym): 8}
        env = SimpleNamespace(
            get_block_id=lambda s: bid_of.get(id(s)),
            is_jagged_tile=lambda bid: bid == 7,
            jagged_tile_parent_ids={7: [3]},
        )

        # add(k_idx, m_idx) — no mul, no starts; should not match.
        g = Graph()
        k_idx = g.placeholder("tile_k_idx")
        k_idx.meta["val"] = k_sym
        m_idx = g.placeholder("tile_m_idx")
        m_idx.meta["val"] = m_sym
        flat = g.call_function(operator.add, args=(k_idx, m_idx))
        g.output(flat)

        self.assertIsNone(_parse_flat_jagged_subscript(flat, env))

    def test_tensor_index_pattern_jagged_flat_fields(self) -> None:
        """``TensorIndexPattern`` defaults preserve the indirect-gather emit
        path; the jagged-flat fields opt in to the canonical jagged 2-D DMA
        slice emit (filled by the plan_tiling producer when it parses a
        1-D flat-form jagged subscript)."""
        from helion._compiler.pallas.plan_tiling import TensorIndexPattern

        # Default: non-jagged, the existing gather path consumes this.
        plain = TensorIndexPattern()
        self.assertFalse(plain.is_jagged_flat)
        self.assertIsNone(plain.sublane_bid)
        self.assertIsNone(plain.sublane_base_fx)
        self.assertIsNone(plain.lane_bid)
        self.assertIsNone(plain.lane_size)

        # Jagged-flat: emit reads these to drive the DMA slice; launcher
        # reads ``lane_size`` to reshape x_flat back to 2-D.
        jagged = TensorIndexPattern(
            is_jagged_flat=True,
            sublane_bid=7,
            sublane_base_fx=None,
            lane_bid=12,
            lane_size=64,
        )
        self.assertTrue(jagged.is_jagged_flat)
        self.assertEqual(jagged.sublane_bid, 7)
        self.assertEqual(jagged.lane_bid, 12)
        self.assertEqual(jagged.lane_size, 64)

    def test_get_reduced_block_ids_carried_acc_collapses_loop_bid(self) -> None:
        """When a fori_loop has a carried tensor accumulator whose shape
        lacks one of the loop's iterated block_ids, that block_id is
        considered reduced.  Mirrors the jagged_sum / jagged_mean tile_k
        loop (row_sums [BB, BM] carried across tile_k iters)."""
        import operator
        from types import SimpleNamespace

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx import Graph
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from helion._compiler.pallas.plan_tiling import get_reduced_block_ids
        from helion.language._tracing_ops import _for_loop, _phi

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            k_sym = shape_env.create_unbacked_symint()
            m_sym = shape_env.create_unbacked_symint()
            torch._check(k_sym >= 1)
            torch._check(m_sym >= 1)
            acc_val = torch.empty((m_sym,))  # shape lacks k_sym
        bid_of = {id(k_sym): 7, id(m_sym): 8}
        env = SimpleNamespace(get_block_id=lambda s: bid_of.get(id(s)))

        g = Graph()
        acc = g.placeholder("acc")
        acc.meta["val"] = acc_val
        loop = g.call_function(_for_loop, args=(0, [0], [k_sym], [acc]))
        out_get = g.call_function(operator.getitem, args=(loop, 0))
        g.call_function(_phi, args=(acc, out_get))  # marks acc as carried

        self.assertEqual(get_reduced_block_ids(loop, [7], env), {7})

    def test_get_reduced_block_ids_no_carried_returns_empty(self) -> None:
        """A fori_loop with no FX-phi'd accumulator is not a reduction
        loop; the helper returns ∅ regardless of loop_block_ids."""
        from types import SimpleNamespace

        from torch.fx import Graph

        from helion._compiler.pallas.plan_tiling import get_reduced_block_ids
        from helion.language._tracing_ops import _for_loop

        env = SimpleNamespace(get_block_id=lambda s: None)
        g = Graph()
        loop = g.call_function(_for_loop, args=(0, [0], [16], []))
        self.assertEqual(get_reduced_block_ids(loop, [7], env), set())

    def test_store_is_post_reduction_via_loop_getitem(self) -> None:
        """If the stored value is a ``getitem(_for_loop, idx)`` (i.e. the
        final-carry of an inner reduction loop), promotion fires: the
        loop's reduced block_ids surface in the return set."""
        import operator
        from types import SimpleNamespace

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx import Graph
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from helion._compiler.pallas.plan_tiling import store_is_post_reduction
        from helion.language._tracing_ops import _for_loop, _phi

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            k_sym = shape_env.create_unbacked_symint()
            m_sym = shape_env.create_unbacked_symint()
            torch._check(k_sym >= 1)
            torch._check(m_sym >= 1)
            acc_val = torch.empty((m_sym,))
        bid_of = {id(k_sym): 7, id(m_sym): 8}
        env = SimpleNamespace(get_block_id=lambda s: bid_of.get(id(s)))

        g = Graph()
        acc = g.placeholder("acc")
        acc.meta["val"] = acc_val
        loop = g.call_function(_for_loop, args=(0, [0], [k_sym], [acc]))
        out_get = g.call_function(operator.getitem, args=(loop, 0))
        g.call_function(_phi, args=(acc, out_get))

        # Build a store node whose value (args[2]) is the loop's final-carry.

        def _store_marker(*args: object) -> None:
            return None

        tensor_ph = g.placeholder("tensor")
        store_node = g.call_function(
            _store_marker, args=(tensor_ph, [], out_get, None)
        )

        # graph_id 0 → block_ids [7] (k_sym's bid).
        self.assertEqual(
            store_is_post_reduction(store_node, env, {0: [7]}), {7}
        )

    def test_store_is_post_reduction_inline_value_returns_empty(self) -> None:
        """If the stored value is computed inline (not the final-carry of
        any loop), no promotion fires — even if upstream loop-carry
        values are *consumed* into the inline expression, the per-iter
        write pattern is still disjoint.  Mirrors jagged_softmax /
        jagged_layer_norm's final-pass store of ``block_out``."""
        import operator
        from types import SimpleNamespace

        from torch.fx import Graph

        from helion._compiler.pallas.plan_tiling import store_is_post_reduction

        env = SimpleNamespace(get_block_id=lambda s: None)
        g = Graph()
        x = g.placeholder("x")
        y = g.placeholder("y")
        inline_value = g.call_function(operator.add, args=(x, y))

        def _store_marker(*args: object) -> None:
            return None

        tensor_ph = g.placeholder("tensor")
        store_node = g.call_function(
            _store_marker, args=(tensor_ph, [], inline_value, None)
        )

        self.assertEqual(
            store_is_post_reduction(store_node, env, {}), set()
        )


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

    @parametrize("static_shapes", (True, False))
    def test_gather_mid_dim_3d_float(self, static_shapes: bool) -> None:
        """Float gather on mid dim of a 3D table emits moveaxis."""

        @helion.kernel(backend="pallas", static_shapes=static_shapes)
        def gather(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                [table.size(0), indices.size(0), table.size(2)],
                dtype=table.dtype,
                device=table.device,
            )
            for tile_a, tile_b, tile_c in hl.tile(
                [table.size(0), indices.size(0), table.size(2)]
            ):
                out[tile_a, tile_b, tile_c] = table[tile_a, indices[tile_b], tile_c]
            return out

        table = torch.randn(8, 16, 32, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 16, (64,), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gather, (indices, table), block_sizes=[8, 64, 32]
        )
        self.assertIn("dot_general", code)
        self.assertIn("moveaxis", code)
        torch.testing.assert_close(result, table[:, indices.long(), :])

    @parametrize("static_shapes", (True, False))
    def test_gather_mid_dim_2d_index(self, static_shapes: bool) -> None:
        """Float gather with 2D index on mid dim exercises full moveaxis formula."""

        @helion.kernel(backend="pallas", static_shapes=static_shapes)
        def gather(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            A, _, C = table.size(0), table.size(1), table.size(2)
            B, S = indices.size(0), indices.size(1)
            out = torch.empty([A, B, S, C], dtype=table.dtype, device=table.device)
            for tile_a, tile_b, tile_s, tile_c in hl.tile([A, B, S, C]):
                out[tile_a, tile_b, tile_s, tile_c] = table[
                    tile_a, indices[tile_b, tile_s], tile_c
                ]
            return out

        table = torch.randn(4, 8, 16, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.randint(0, 8, (2, 4), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gather, (indices, table), block_sizes=[4, 2, 4, 16]
        )
        self.assertIn("dot_general", code)
        self.assertIn("moveaxis", code)
        torch.testing.assert_close(result, table[:, indices.long(), :])


instantiate_parametrized_tests(TestPallasIndirectGather)


@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasJaxFn(TestCase):
    """End-to-end tests for the ``Kernel.jax_fn`` pure-JAX export path.

    Covers all three ``pallas_loop_type`` flavours plus a multi-kernel
    composition test, each exercising the kernel inside a ``jax.jit``
    boundary with non-trivial pure-JAX prologue / epilogue around it.
    """

    def _import_jax(self) -> tuple[Any, Any]:
        import jax
        import jax.numpy as jnp

        return jax, jnp

    def test_jax_fn_emit_pipeline(self) -> None:
        """jax_fn drives an emit_pipeline kernel inside ``jax.jit``."""
        jax, jnp = self._import_jax()

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[128, 128], pallas_loop_type="emit_pipeline"
            ),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        jax_kernel = add_kernel.jax_fn

        @jax.jit
        def f(a: Any, b: Any, scale: float) -> Any:
            # prologue (pure jax)
            a = a * scale
            b = jnp.tanh(b)
            # kernel
            c = jax_kernel(a, b)
            # epilogue (pure jax)
            return jnp.sum(c) + jnp.mean(c) * 0.5

        a = jnp.ones((128, 128), dtype=jnp.float32)
        b = jnp.full((128, 128), 0.5, dtype=jnp.float32)
        scale = 2.0

        result = float(f(a, b, scale))

        # Reference: same prologue/epilogue, eager addition
        ref_a = a * scale
        ref_b = jnp.tanh(b)
        ref_c = ref_a + ref_b
        ref = float(jnp.sum(ref_c) + jnp.mean(ref_c) * 0.5)
        self.assertAlmostEqual(result, ref, places=2)

    def test_jax_fn_unroll(self) -> None:
        """jax_fn drives an unroll kernel inside ``jax.jit``."""
        jax, jnp = self._import_jax()

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(block_sizes=[128, 128], pallas_loop_type="unroll"),
        )
        def relu_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = torch.relu(x[tile]) + y[tile]
            return out

        jax_kernel = relu_add_kernel.jax_fn

        @jax.jit
        def f(a: Any, b: Any) -> Any:
            a = a - 0.25
            b = b * b
            c = jax_kernel(a, b)
            return jnp.sum(c * c)

        a = jnp.linspace(-1.0, 1.0, 128 * 128, dtype=jnp.float32).reshape((128, 128))
        b = jnp.full((128, 128), 0.3, dtype=jnp.float32)

        result = float(f(a, b))

        ref_a = a - 0.25
        ref_b = b * b
        ref_c = jnp.maximum(ref_a, 0.0) + ref_b
        ref = float(jnp.sum(ref_c * ref_c))
        self.assertAlmostEqual(result, ref, places=1)

    def test_jax_fn_fori_loop(self) -> None:
        """jax_fn drives a fori_loop kernel inside ``jax.jit``."""
        jax, jnp = self._import_jax()

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(block_sizes=[128, 128], pallas_loop_type="fori_loop"),
        )
        def mul_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                for tile_n in hl.tile(x.size(1)):
                    out[tile_m, tile_n] = x[tile_m, tile_n] * 1.5 + y[tile_m, tile_n]
            return out

        jax_kernel = mul_add_kernel.jax_fn

        @jax.jit
        def f(a: Any, b: Any) -> Any:
            a = a * 0.5 + 1.0
            b = jnp.exp(b * 0.1)
            c = jax_kernel(a, b)
            return jnp.mean(c)

        a = jnp.full((128, 128), 2.0, dtype=jnp.float32)
        b = jnp.zeros((128, 128), dtype=jnp.float32)

        result = float(f(a, b))

        ref_a = a * 0.5 + 1.0
        ref_b = jnp.exp(b * 0.1)
        ref_c = ref_a * 1.5 + ref_b
        ref = float(jnp.mean(ref_c))
        self.assertAlmostEqual(result, ref, places=2)

    def test_jax_fn_multi_kernel_in_one_jit(self) -> None:
        """A single ``jax.jit`` function uses two distinct Helion kernels."""
        jax, jnp = self._import_jax()

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(block_sizes=[128, 128]),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(block_sizes=[128, 128]),
        )
        def mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * y[tile]
            return out

        add_jax = add_kernel.jax_fn
        mul_jax = mul_kernel.jax_fn

        @jax.jit
        def f(a: Any, b: Any, c: Any) -> Any:
            # prologue
            a = a + 1.0
            # first kernel: a + b
            ab = add_jax(a, b)
            # middle pure-jax transform
            ab = jnp.tanh(ab)
            # second kernel: (tanh result) * c
            out = mul_jax(ab, c)
            # epilogue
            return jnp.sum(out)

        a = jnp.full((128, 128), 0.5, dtype=jnp.float32)
        b = jnp.full((128, 128), 0.25, dtype=jnp.float32)
        c = jnp.full((128, 128), 2.0, dtype=jnp.float32)

        result = float(f(a, b, c))

        ref_a = a + 1.0
        ref_ab = jnp.tanh(ref_a + b)
        ref_out = ref_ab * c
        ref = float(jnp.sum(ref_out))
        self.assertAlmostEqual(result, ref, places=2)


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
