from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.fake_tensor import create_fake_tensor_mode
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


def test_fake_tensor_mode_supports_cond() -> None:
    with create_fake_tensor_mode(ShapeEnv()):
        x = torch.empty(4)
        pred = torch.empty((), dtype=torch.bool)
        result = torch.ops.higher_order.cond(
            pred, lambda x: x + 1, lambda x: x - 1, (x,)
        )
        assert result.shape == x.shape
        assert result.dtype == x.dtype


@onlyBackends(["triton", "cute"])
@skipIfRefEager("Compiler matmul facts are not collected in ref eager mode")
@pytest.mark.parametrize("queries,keys", [(128, 128), (128, 256), (256, 128)])
def test_bmm_cast_keeps_distinct_tile_axes(queries: int, keys: int) -> None:
    @helion.kernel(static_shapes=True)
    def bmm_chain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batches = q.size(0)
        heads = hl.specialize(q.size(1))
        queries = q.size(2)
        dim = hl.specialize(q.size(3))
        keys = k.size(2)
        out = torch.empty_like(q)
        for batch in hl.grid(batches):
            for tile_q in hl.tile(queries):
                query = q[batch, :, tile_q, :]
                acc = hl.zeros([heads, tile_q, dim], dtype=torch.float32)
                for tile_k in hl.tile(keys):
                    key = k[batch, :, tile_k, :]
                    value = v[batch, :, tile_k, :]
                    scores = torch.bmm(
                        query,
                        key.transpose(-2, -1),
                        torch.float32,
                    )
                    acc = acc + torch.bmm(scores.to(value.dtype), value)
                out[batch, :, tile_q, :] = acc.to(out.dtype)
        return out

    # Binding CPU tensors exercises FakeTensor tracing without launching kernels.
    # The two tile axes and the fixed head dimension all have the representative
    # extent 64; casting the first bmm must retain their distinct identities.
    bound = bmm_chain.bind(
        tuple(
            torch.empty(shape, dtype=torch.float16)
            for shape in ((2, 4, queries, 64), (2, 4, keys, 64), (2, 4, keys, 64))
        )
    )
    qk, pv = bound.config_spec.matmul_facts
    assert qk.m_block_id is not None
    assert qk.n_block_id is not None
    assert qk.m_block_id != qk.n_block_id
    assert pv.m_block_id == qk.m_block_id
    assert pv.k_block_id == qk.n_block_id
    assert (pv.static_m, pv.static_n, pv.static_k) == (queries, 64, keys)
