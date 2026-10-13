from __future__ import annotations

import ast

import pytest
import torch

from test._cute_aux import _cpu_codegen

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    limit: int,
    hole: hl.constexpr,
) -> torch.Tensor:
    batches, _, reduction = a.shape
    rows = 64
    output = torch.empty(
        (batches, rows, v.size(2)), device=a.device, dtype=torch.float32
    )
    for bi in hl.tile(batches, block_size=1):
        row, inner = hl.arange(rows), hl.arange(reduction)
        mask = (row < limit) & (row != hole)
        left = hl.load(a, [bi.id, row, inner], extra_mask=mask[:, None])
        right = hl.load(b, [bi.id, row, inner], extra_mask=mask[:, None])
        product = hl.dot(left, right.T)
        output[bi.id, :, :] = hl.dot(product.to(a.dtype), v[bi.id, :, :])
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_loop(
    a: torch.Tensor, initial: torch.Tensor, limit: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batches, steps, rows, _ = a.shape
    output = torch.empty(
        (batches, steps, rows, 32), device=a.device, dtype=torch.float32
    )
    final = torch.empty_like(initial)
    for bi in hl.tile(batches, block_size=1):
        state = initial[bi.id, :, :]
        for step in hl.grid(steps):
            row, column = hl.arange(rows), hl.arange(rows)
            value = hl.load(
                a, [bi.id, step, row, column], extra_mask=(row + step < limit)[:, None]
            )
            first = hl.dot(value, state.to(a.dtype))
            state = hl.dot(value.T, first.to(a.dtype), acc=state * 0.125)
            output[bi.id, step, :, :] = state
        final[bi.id, :, :] = state
    return output, final


def _config() -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule="cp_async")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,hole", [(64, -1), (53, -1), (64, 31)])
def test_masked_copy_codegen_retains_scalar_fallback(
    dtype: torch.dtype, rows: int, hole: int
) -> None:
    args = (
        torch.empty((2, rows, 64), dtype=dtype),
        torch.empty((2, rows, 64), dtype=dtype),
        torch.empty((2, 64, 32), dtype=dtype),
        64,
        hole,
    )
    with _cpu_codegen():
        source = _masked_chain._bind_isolated(args).to_code(_config())
    ast.parse(source)
    assert source.count("_masked_async_copied = cutlass.Boolean(False)") == 2
    assert source.count("if not chain_0_") >= 2
    assert "_masked_async_address" in source
    assert "_masked_async_source" in source
    assert ".shape[1] ==" in source and ".layout.stride[1] ==" in source
    assert "cp_async_wait_group(0)" in source


def test_changing_iteration_copy_codegen() -> None:
    args = (
        torch.empty((2, 3, 64, 64), dtype=torch.bfloat16),
        torch.empty((2, 64, 32)),
        64,
    )
    with _cpu_codegen():
        source = _masked_loop._bind_isolated(args).to_code(_config())
    assert source.count("_masked_async_copied = cutlass.Boolean(False)") == 2
    loop = source.index("for chain_loop_index")
    assert loop < source.index("_masked_async_copied = cutlass.Boolean(False)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,offset,hole", [(64, 0, -1), (53, 0, -1), (64, 1, -1), (64, 0, 31)]
)
def test_masked_async_chain_gpu(
    dtype: torch.dtype, rows: int, offset: int, hole: int
) -> None:
    def value(shape: tuple[int, ...]) -> torch.Tensor:
        count = 1
        for extent in shape:
            count *= extent
        storage = torch.randn(count + offset, device="cuda", dtype=dtype) * 0.04
        return storage[offset:].view(shape)

    a, b, v = value((2, rows, 64)), value((2, rows, 64)), value((2, 64, 32))
    frozen = [item.clone() for item in (a, b, v)]
    function = _masked_chain._bind_isolated((a, b, v, 64, hole)).compile_config(
        _config()
    )
    padded = []
    for item in (a, b):
        full = torch.zeros((2, 64, 64), dtype=torch.float64, device="cuda")
        full[:, :rows] = item.double()
        if hole >= 0:
            full[:, hole] = 0
        padded.append(full)
    expected = (padded[0] @ padded[1].transpose(-1, -2)).to(dtype).double() @ v.double()
    for _ in range(2):
        torch.testing.assert_close(
            function(a, b, v, 64, hole), expected.float(), atol=0.002, rtol=0.01
        )
    for item, before in zip((a, b, v), frozen, strict=True):
        torch.testing.assert_close(item, before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_masked_async_changing_loop_gpu(dtype: torch.dtype) -> None:
    a = torch.randn((2, 3, 64, 64), device="cuda", dtype=dtype) * 0.025
    initial = torch.randn((2, 64, 32), device="cuda") * 0.1
    function = _masked_loop._bind_isolated((a, initial, 64)).compile_config(_config())
    for shift in (0.0, 0.005):
        inputs = a + shift
        frozen_inputs = inputs.clone()
        frozen_initial = initial.clone()
        state = initial.double()
        expected = []
        for step in range(3):
            value = inputs[:, step].double().clone()
            value[:, 64 - step :] = 0
            first = (value @ state.to(dtype).double()).to(dtype).double()
            state = value.transpose(-1, -2) @ first + state.float().double() * 0.125
            state = state.float().double()
            expected.append(state)
        output, final = function(inputs, initial, 64)
        torch.testing.assert_close(
            output, torch.stack(expected, dim=1).float(), atol=0.002, rtol=0.01
        )
        torch.testing.assert_close(final, state.float(), atol=0.002, rtol=0.01)
        torch.testing.assert_close(inputs, frozen_inputs, atol=0, rtol=0)
        torch.testing.assert_close(initial, frozen_initial, atol=0, rtol=0)
