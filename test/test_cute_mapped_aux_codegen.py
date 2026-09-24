from __future__ import annotations

import ast

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_batched_aux_tma import _config
from test.test_cute_batched_aux_tma import _cpu_codegen

import helion
from helion import exc
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mapped_batched_aux(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    residual: torch.Tensor,
    diagonal: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    group_size = hl.specialize(group_size)
    batches, rows, inner = lhs.shape
    columns = rhs.shape[-1]
    groups = columns // group_size
    for bi, mi, ni in hl.tile([batches, rows, columns], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(inner):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        work = bi.index[:, None, None]
        row = mi.index[None, :, None]
        column = ni.index[None, None, :]
        group = column // group_size
        base = (work * groups + group) * rows
        weight = hl.load(weights, [base + row])
        last = hl.load(weights, [base + rows - 1])
        scale = torch.exp(weight - last * 0.5).to(out.dtype).float()
        d = hl.load(diagonal, [group]).float()
        x = residual[bi, mi, ni].float()
        bias = (x * d).to(out.dtype).float()
        out[bi, mi, ni] = (acc * scale + bias).to(out.dtype)
    return out


def _inputs(dtype: torch.dtype, group_size: int = 64) -> tuple[object, ...]:
    batches, rows, columns, inner = 3, 128, 768, 64
    groups = columns // group_size
    return (
        torch.empty((batches, rows, inner), dtype=dtype),
        torch.empty((batches, inner, columns), dtype=dtype),
        torch.empty((batches, rows, columns), dtype=dtype),
        torch.empty((groups,), dtype=dtype),
        torch.empty((batches * groups * rows,), dtype=torch.float32),
        torch.empty((batches, rows, columns), dtype=dtype),
        group_size,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [1, 3, 64])
def test_mapped_batched_aux_uses_one_staged_ring(dtype, group_size) -> None:
    with _cpu_codegen():
        code = _mapped_batched_aux._bind_isolated(_inputs(dtype, group_size)).to_code(
            _config(tcgen05_consumer_regs=128, tcgen05_aux_role_local_scheduler=True)
        )
    assert code.count("'kind': 'tcgen05_aux_tma'") == 1
    assert code.count("'kind': 'tcgen05_aux_direct'") == 3
    assert "tcgen05_mapped_coord" in code
    assert "cutlass.range_constexpr" in code
    assert "'c_dtype': 'torch.float32'" in code
    assert "'c_idx':" in code and "'d_idx':" in code
    assert "tcgen05_aux_gmem_tile_1" not in code
    ast.parse(code)


def test_mapped_aux_rejects_unsupported_simt_transport() -> None:
    with _cpu_codegen(), pytest.raises(exc.BackendUnsupported, match="mapped"):
        _mapped_batched_aux._bind_isolated(_inputs(torch.bfloat16)).to_code(
            _config(tcgen05_aux_load_mode="simt")
        )
