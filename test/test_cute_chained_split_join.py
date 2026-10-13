from __future__ import annotations

import operator
from typing import cast

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.chained_split_join import supports
import helion.language as hl
from helion.language import view_ops


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _quadrants(a: torch.Tensor, b: torch.Tensor, fill: hl.constexpr) -> torch.Tensor:
    output = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    for batch in hl.tile(a.size(0), block_size=1):
        first = hl.dot(a[batch.id, :, :], b[batch.id, :, :])
        blocks = first.reshape(2, 16, 2, 16).permute(1, 3, 0, 2)
        left, right = hl.split(blocks)
        upper_left, lower_left = hl.split(left)
        upper_right, lower_right = hl.split(right)
        second = hl.dot(upper_left.to(a.dtype), lower_right.to(a.dtype))
        left_result = hl.join(second, lower_left)
        right_result = hl.join(
            torch.full_like(upper_right, cast("float", fill)), lower_right
        )
        result = hl.join(left_result, right_result)
        output[batch.id, :, :] = result.permute(2, 0, 3, 1).reshape(32, 32)
    return output


def _config() -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fill", [0.0, -1.25])
def test_nested_views_codegen(dtype: torch.dtype, fill: float) -> None:
    args = (torch.empty((2, 32, 32), dtype=dtype),) * 2 + (fill,)
    with _cpu_codegen():
        source = _quadrants._bind_isolated(args).to_code(_config())
    assert "chain_0_mma =" in source and "chain_1_mma =" in source
    assert "chain_0_c" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fill", [0.0, -1.25])
def test_nested_views_gpu(dtype: torch.dtype, fill: float) -> None:
    torch.manual_seed(81)
    a, b = (
        torch.randn((2, 32, 32), device="cuda", dtype=dtype) * 0.125 for _ in range(2)
    )
    first = a.float() @ b.float()
    expected = first.clone()
    expected[:, :16, :16] = (
        first[:, :16, :16].to(dtype).float() @ first[:, 16:, 16:].to(dtype).float()
    )
    expected[:, :16, 16:] = fill
    args = (a, b, fill)
    compiled = _quadrants._bind_isolated(args).compile_config(_config())
    torch.testing.assert_close(compiled(*args), expected, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("change", ["shape", "dtype", "index", "tuple", "broadcast"])
def test_malformed_views_reject(change: str) -> None:
    graph = Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty((16, 16, 2))
    split = graph.call_function(view_ops.split, (source,))
    split.meta["val"] = (torch.empty((16, 16)), torch.empty((16, 16)))
    item = graph.call_function(operator.getitem, (split, 0))
    item.meta["val"] = torch.empty((16, 16))
    assert supports(split) and supports(item)
    if change == "shape":
        source.meta["val"] = torch.empty((16, 16, 3))
    elif change == "dtype":
        item.meta["val"] = torch.empty((16, 16), dtype=torch.float16)
    elif change == "index":
        item.args = (split, -1)
    elif change == "tuple":
        split.target = operator.add
    else:
        right = graph.placeholder("right")
        right.meta["val"] = torch.empty((16,))
        joined = graph.call_function(view_ops.join, (item, right))
        joined.meta["val"] = torch.empty((16, 16, 2))
        assert not supports(joined)
        return
    assert not supports(item)
