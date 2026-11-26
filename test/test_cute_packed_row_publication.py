from __future__ import annotations

import ast

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import default_cute_mma_support
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _biased_norm(
    x: torch.Tensor,
    bias: torch.Tensor,
    _cute_publication_pack: torch.Tensor,
    eps: float,
    early_round: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    for j in hl.tile(x.size(1)):
        y[0, j] = x[0, j] + bias[j]
    hl.barrier()
    for row in hl.tile(x.size(0), block_size=1):
        f = y[row, :].float()
        inv = torch.rsqrt((f * f).mean(dim=-1) + eps)
        normalized = f * inv[:, None]
        if early_round:
            z[row, :] = normalized.to(y.dtype) * _cute_publication_pack[None, :]
        else:
            z[row, :] = normalized * _cute_publication_pack[None, :]
    return y, z


def test_paired_normalization_preserves_rounding_and_reserves_helper() -> None:
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        row = torch.empty((1, 768), dtype=torch.bfloat16)
        bias = torch.empty((768,), dtype=torch.bfloat16)
        scale = torch.empty_like(bias)
    c = helion.Config(
        block_sizes=[128],
        reduction_loops=[1],
        pid_type="persistent_interleaved",
        num_threads=[32, 1],
        cute_single_row_publication=True,
        cute_parallel_row_sum=True,
        cute_packed_row_publication=True,
    )
    with cpu_render_target(
        fake_device_sm_count=148, mma_support=default_cute_mma_support()
    ):
        eligible = cpu_bind(_biased_norm, (row, bias, scale, 0.125, True))
        text = eligible.to_code(c)
        f = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
        assert "_cute_publication_pack" not in {a.arg for a in f.args.args}
        assert any(
            isinstance(n, ast.Call) and ast.unparse(n.func) == "_cute_publication_pack"
            for n in ast.walk(f)
        )
        assert "cutlass.range_constexpr(12)" in text
        without_rounding = cpu_bind(_biased_norm, (row, bias, scale, 0.125, False))
        with pytest.raises(exc.InvalidConfig, match="packed row publication"):
            without_rounding.to_code(c)
