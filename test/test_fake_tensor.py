from __future__ import annotations

import operator

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.fake_tensor import create_fake_tensor_mode
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl

pytestmark = pytest.mark.cpu_only


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


@helion.kernel(config=helion.Config(block_sizes=[64]), static_shapes=True)
def _tile_groups(x: torch.Tensor, group: int) -> torch.Tensor:
    group = hl.specialize(group)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + tile.begin // group
    return out


@helion.kernel(config=helion.Config(block_sizes=[64]), static_shapes=True)
def _tile_double_groups(x: torch.Tensor, group: int) -> torch.Tensor:
    group = hl.specialize(2 * group)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + tile.begin // group
    return out


@onlyBackends(["triton", "cute"])
@skipIfRefEager("Inspects the compiled device IR, absent in ref eager mode")
@pytest.mark.parametrize(
    ("kernel", "group"), [(_tile_groups, 128), (_tile_double_groups, 64)]
)
def test_profiler_keeps_specialized_scalars_constant(
    kernel: helion.Kernel, group: int
) -> None:
    """``hl.specialize`` folds an int argument into the device code as a
    literal through the shape-env replacement its guard installs (``u0 ->
    128``, or ``u0 -> 64`` for ``hl.specialize(2 * u0)``). Inside a
    torch.profiler context the compile suppresses guards
    (test_misc.py::test_profiler_does_not_concretize_block_vars), which used
    to drop that replacement too: the device graph kept the symbol, and
    pattern matchers expecting the literal (the CuTe flex envelopes'
    ``m_begin // QB`` table index) declined."""
    # Binding CPU tensors traces the kernel without launching it.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        bound = kernel.bind((torch.randn([1024]), group))
    divisors = [
        node.args[1]
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is operator.floordiv
    ]
    assert divisors == [128]
