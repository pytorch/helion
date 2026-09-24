from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_aux_cache import _args
from .test_cute_chained_aux_cache import _code
import helion
from helion._compiler.cute.chained_aux_cache import make_late_auxiliary_cache
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _auxiliary_prior_dot_index(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    q, columns = b.shape[1], v.shape[2]
    out = torch.empty((batch, rows, columns), dtype=a.dtype, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk, qq = hl.arange(reduction), hl.arange(q)
        prior = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        first = hl.dot((a[bi, row, kk] + 0.1).to(a.dtype), b[bi, qq, kk].T)
        result = hl.dot(first.to(a.dtype), v[bi, qq, col])
        index = prior[0, :][col.index].to(torch.int64) % delta.shape[1]
        result += delta[bi, index][None, :].float()
        out[bi, row, col] = result.to(a.dtype)
    return out


def test_auxiliary_cache_prior_dot_index_falls_back() -> None:
    args = _args("cpu", "plain")
    disabled = _code(args, False, _auxiliary_prior_dot_index)
    enabled = _code(args, True, _auxiliary_prior_dot_index)
    assert "OperandSource.TMEM" in disabled
    assert "OperandSource.TMEM" in enabled
    assert "chain_late_aux_0 =" not in enabled


def test_auxiliary_cache_padded_scalar_query_uses_global_fallback() -> None:
    source = _code(_args("cpu", "tail"))
    reads = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.IfExp)
        and "chain_late_aux_0[127]" in ast.unparse(node.body)
    ]
    assert reads
    for read in reads:
        assert ".load()" in ast.unparse(read.orelse)
        assert not eval(
            compile(ast.Expression(read.test), "<aux-cache-domain>", "eval"),
            {},
            {"cutlass": SimpleNamespace(Int32=int, Int64=int)},
        )


@pytest.mark.parametrize("capacity, count", [(511, 1), (512, 1), (767, 1), (768, 2)])
def test_auxiliary_cache_exact_capacity_and_typed_offsets(
    capacity: int, count: int
) -> None:
    def bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs["arena_bytes"] = capacity
        return make_late_auxiliary_cache(*args, **kwargs)

    with patch(
        "helion._compiler.cute.chained_tcgen05.make_late_auxiliary_cache",
        side_effect=bounded,
    ):
        source = _code(_args("cpu", "plain"))
    assert source.count("= cute.make_tensor(cute.recast_ptr(chain_a_workspace") == count
    if capacity == 511:
        # The 512-byte FP32 candidate does not fit; the 256-byte BF16 one does.
        assert "chain_a_workspace + 0, dtype=cutlass.BFloat16" in source
    else:
        assert "chain_a_workspace + 0, dtype=cutlass.Float32" in source
    if count == 2:
        assert "chain_a_workspace + 256, dtype=cutlass.BFloat16" in source
