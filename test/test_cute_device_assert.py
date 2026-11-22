"""Where CuTe puts an hl.device_assert: ahead of the stores after it, and in
the tile of the generic lowering rather than the fused attention or tcgen05
bodies, which would leave it out or run it before their tile loop (on CPU,
for sm100)."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import TestCase
from helion._testing.cute_aux import as_sm100
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(static_shapes=True, backend="cute")
def checked_scatter(
    idx: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
    for tile in hl.tile(idx.size(0)):
        i = idx[tile, :]
        hl.device_assert((i >= 0) & (i < dst.size(0)), "index out of bounds")
        dst[i.amin(-1)] = src[tile]
    return dst


@helion.kernel(static_shapes=True, backend="cute")
def checked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batch_heads, seq_len, head_dim = q.size()
    out = torch.empty_like(q)
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            qk = hl.dot(q_tile * 0.18033688, k[bh.begin, tile_n, :].transpose(0, 1))
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = hl.dot(
                p.to(v.dtype), v[bh.begin, tile_n, :], acc=acc * alpha[:, None]
            )
            m_i = m_new
        hl.device_assert(l_i > 0, "empty softmax row")
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def checked_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        hl.device_assert(acc == acc, "nan")
        out[tile_m, tile_n] = acc.to(a.dtype)
    return out


class TestCuteDeviceAssert(TestCase):
    def test_before_stores(self):
        # The lane split places the assert by its reads, after the min of a
        # row of 64 indices over two lanes, but keeps it ahead of the scatter
        # it guards.
        args = (
            torch.zeros(1000, 64, dtype=torch.int64),
            torch.zeros(1000),
            torch.zeros(128),
        )
        with as_sm100():
            bound = cpu_bind(checked_scatter, args)
            source = bound.to_code(helion.Config(block_sizes=[1], num_threads=[1, 32]))
        self.assertLess(
            source.index("cute.testing.assert_("), source.index("(dst.iterator")
        )

    def test_fused_attention(self):
        # The fused attention body would drop the assert, also under a config
        # in its envelope: the generic lowering emits it.
        args = tuple(torch.zeros(1, 256, 64, dtype=torch.bfloat16) for _ in range(3))
        with as_sm100():
            bound = cpu_bind(checked_attention, args)
            default = bound.config_spec.default_config()
            flash = helion.Config(
                **{
                    **default.config,
                    "block_sizes": [128, 128],
                    "num_threads": [0, 0, 0],
                }
            )
            for config in (default, flash):
                self.assertRegex(
                    bound.to_code(config),
                    r"cute\.testing\.assert_\(\w+, 'empty softmax row'\)",
                )
        self.assertFalse(bound.config_spec.cute_flash_search_enabled)

    def test_no_tcgen05(self):
        # The tcgen05 body would check the zero accumulator before its tile
        # loop: the generic lowering checks the product after the K loop.  The
        # collective MMA, which refuses the assert, gets no seeds either.
        args = (torch.zeros(1024, 1024, dtype=torch.bfloat16),) * 2
        with as_sm100():
            bound = cpu_bind(checked_matmul, args)
            source = bound.to_code(bound.config_spec.default_config())
            with bound.env:
                seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
        self.assertFalse(bound.config_spec.tcgen05.search_enabled)
        self.assertNotIn("tcgen05", source)
        self.assertRegex(source, r"cute\.testing\.assert_\(\w+, 'nan'\)")
        self.assertEqual(seeds, [])


if __name__ == "__main__":
    unittest.main()
