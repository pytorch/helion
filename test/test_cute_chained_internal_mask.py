from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_intermediate(a, b, gather: hl.constexpr):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, columns in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        computed = a[rows, kk].float() * 0.75
        mask = (kk[None, :] % 3) != 1
        if gather:
            selected = hl.load(computed, [slice(None), k - 1 - kk], extra_mask=mask)
        else:
            selected = hl.load(computed, [slice(None), slice(None)], extra_mask=mask)
        out[rows, columns] = hl.dot(selected.to(a.dtype), b[kk, columns])
    return out


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("gather", [False, True])
def test_internal_load_mask_codegen(schedule: str, gather: bool) -> None:
    with _cpu_codegen():
        bound = _masked_intermediate._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
                gather,
            )
        )
        source = bound.to_code(
            helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
        )
    assert "chain_0_mma" in source
    assert "else cutlass.Float32(0)" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("gather", [False, True])
def test_internal_load_mask_gpu(schedule: str, gather: bool) -> None:
    torch.manual_seed(933)
    a = torch.randn((128, 16), device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn((16, 32), device=DEVICE, dtype=torch.bfloat16)
    bound = _masked_intermediate._bind_isolated((a, b, gather))
    config = helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
    assert "chain_0_mma" in bound.to_code(config)
    actual = bound.compile_config(config)(a, b, gather)
    computed = a.float() * 0.75
    if gather:
        computed = computed.flip(1)
    computed[:, torch.arange(16, device=DEVICE) % 3 == 1] = 0
    expected = computed.to(a.dtype).float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.002)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _data_indexed_intermediate(a, b, positions, mode: hl.constexpr):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, columns in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        computed = torch.exp(a[rows, kk].float())
        if mode == "host":
            indices = positions[kk]
        elif mode == "dot":
            jj = hl.arange(k)
            product = hl.dot(a[rows, kk], b[kk, jj])
            indices = product[0, :].to(torch.int32) % k
        elif mode == "sum":
            indices = computed.sum(0).to(torch.int32) % k
        else:
            indices = hl.cumsum(a[0, kk].float(), 0).to(torch.int32) % k
        selected = hl.load(computed, [slice(None), indices])
        out[rows, columns] = hl.dot(selected.to(a.dtype), b[kk, columns])
    return out


@pytest.mark.parametrize("mode", ["host", "dot", "sum", "scan"])
def test_data_dependent_domains_are_not_proven_by_substituting_zero(mode: str) -> None:
    with _cpu_codegen():
        bound = _data_indexed_intermediate._bind_isolated(
            (
                torch.empty((128, 32), dtype=torch.bfloat16),
                torch.empty((32, 32), dtype=torch.bfloat16),
                torch.empty(32, dtype=torch.int32),
                mode,
            )
        )
        with pytest.raises(exc.BackendUnsupported, match="data-dependent"):
            bound.to_code(
                helion.Config(cute_chained_mma_schedule="tcgen05_tmem", num_warps=4)
            )
