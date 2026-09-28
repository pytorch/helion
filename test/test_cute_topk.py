"""CuTe top-k lowering, selection networks, tuning, and correctness tests."""

from __future__ import annotations

import ast
import itertools
import random
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


# Basic top-k tests.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_topk(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    rows = x.size(0)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=largest, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


def _check_topk(
    x: torch.Tensor,
    original: torch.Tensor,
    output: tuple[torch.Tensor, torch.Tensor],
    k: int,
    largest: bool,
) -> None:
    values, indices = output
    expected = torch.topk(original, k, dim=-1, largest=largest, sorted=True).values
    assert values.shape == indices.shape == (x.size(0), k)
    assert values.dtype == x.dtype
    assert indices.dtype == torch.int64
    assert values.device == indices.device == x.device
    torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
    assert bool(((indices >= 0) & (indices < x.size(1))).all())
    torch.testing.assert_close(
        values, original.gather(1, indices), rtol=0, atol=0, equal_nan=True
    )
    ordered_indices = indices.sort(dim=-1).values
    assert bool((ordered_indices[:, 1:] != ordered_indices[:, :-1]).all())
    # Compare bit patterns to preserve NaN payloads and signed zero in the input.
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


def _inputs(width: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((17, width), generator=generator, dtype=dtype)
    x[0].zero_()
    x[1] = (torch.arange(width) % 5 - 2).to(dtype)
    x[2] = -x[2].abs() - 1
    x[3].fill_(float("inf"))
    x[4].fill_(float("-inf"))
    x[5, 0::3] = float("nan")
    x[5, 1::3] = float("inf")
    x[5, 2::3] = float("-inf")
    x[6].fill_(float("nan"))
    x[7].zero_()
    x[7, 1::2] = -0.0
    # The final row is outside the two complete eight-row blocks.
    x[-1] = torch.arange(width - 1, -1, -1).to(dtype)
    return x.to(DEVICE)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "dtype,width,k,largest,lanes",
    [
        (torch.bfloat16, 16, 1, True, 1),
        (torch.float16, 16, 3, False, 4),
        (torch.bfloat16, 65, 3, True, 4),
        (torch.float16, 65, 32, False, 16),
        (torch.bfloat16, 1024, 32, True, 16),
        (torch.float16, 1024, 32, False, 32),
        (torch.bfloat16, 65, 1, False, 1),
        (torch.float16, 65, 3, True, 32),
        (torch.bfloat16, 16, 3, False, 16),
        (torch.float16, 1024, 1, True, 4),
    ],
)
def test_direct_topk(
    dtype: torch.dtype, width: int, k: int, largest: bool, lanes: int
) -> None:
    x = _inputs(width, dtype)
    original = x.clone()
    code, output = code_and_output(
        _row_topk,
        (x, k, largest),
        block_sizes=[8],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
    )
    assert "_cute_local_topk" in code
    assert "sort_rank" not in code
    _check_topk(x, original, output, k, largest)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _topk_and_copy(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = x.size(0)
    values = torch.empty((rows, 3), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, 3), dtype=torch.int64, device=x.device)
    copied = torch.empty_like(x)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], 3, dim=-1, largest=True, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
        copied[row, :] = x[row, :]
    return values, indices, copied


@onlyBackends(["cute"])
def test_direct_topk_preserves_other_stores() -> None:
    """The whole-region matcher must reject a region with another output store."""
    x = torch.arange(48, dtype=torch.bfloat16, device=DEVICE).reshape(3, 16)
    original = x.clone()
    code, (values, indices, copied) = code_and_output(
        _topk_and_copy, (x,), block_sizes=[8]
    )
    assert "_cute_local_topk" not in code
    _check_topk(x, original, (values, indices), 3, True)
    torch.testing.assert_close(copied, original, rtol=0, atol=0)


# Cached launcher alignment transitions.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alignment_relaunch_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    rows = x.size(0)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_same_topk_launcher_aligned_shifted_aligned() -> None:
    rows, width, k = 17, 64, 3
    storage = torch.randn(rows * width + 1, dtype=torch.bfloat16, device="cuda")
    bits = storage.view(torch.int16)
    bits[:8] = torch.tensor(
        [0x7FC1, -46, -32768, 0, -128, 0x7F80, 0x3F80, -16512],
        dtype=torch.int16,
        device="cuda",
    )
    original_storage = storage.clone()
    aligned = storage[:-1].view(rows, width)
    shifted = storage[1:].view(rows, width)
    assert aligned.shape == shifted.shape and aligned.stride() == shifted.stride()
    assert aligned.data_ptr() % 16 == 0 and shifted.data_ptr() % 16 == 2
    bound = _alignment_relaunch_topk.bind((aligned, k))
    config = bound.config_spec.default_config()
    config.config.update(
        cute_topk_lanes_per_row=16, cute_topk_rows_per_block=8, cute_topk_vector_width=8
    )
    # Reuse this exact generated host function to exercise launcher caches even
    # if Kernel.bind independently specializes input alignment.
    compiled = bound.compile_config(config)
    for tensor in (aligned, shifted, aligned):
        original = tensor.clone()
        values, indices = compiled(tensor, k)
        expected = torch.topk(original, k, dim=-1, sorted=True).values
        torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(
            values.view(torch.int16), original.gather(1, indices).view(torch.int16)
        )
        assert torch.equal(
            storage.view(torch.int16), original_storage.view(torch.int16)
        )


# Geometry top-k tests.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _extra_row_topk(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    rows = x.size(0)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _extra_out_topk(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    largest: hl.constexpr,
) -> None:
    k = hl.specialize(k)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx


def _layout_input(
    rows: int, width: int, dtype: torch.dtype, padding: int, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    stride = width + padding
    generator = torch.Generator().manual_seed(20260924)
    storage = torch.randn(offset + rows * stride, generator=generator, dtype=dtype)
    cpu_view = torch.as_strided(storage, (rows, width), (stride, 1), offset)
    # Include distinct positive/negative NaN payloads and both zero signs.
    words = (
        (0x7FC1, 0xFFD2, 0x8000, 0x0000, 0xFF80, 0x7F80, 0x3F80, 0xBF80)
        if dtype == torch.bfloat16
        else (0x7E01, 0xFE22, 0x8000, 0x0000, 0xFC00, 0x7C00, 0x3C00, 0xBC00)
    )
    bits = torch.tensor(
        [word if word < 0x8000 else word - 0x10000 for word in words],
        dtype=torch.int16,
    )
    cpu_view[0] = bits.view(dtype).repeat((width + 7) // 8)[:width]
    if rows > 1:
        cpu_view[1].zero_()
        cpu_view[1, 1::2] = -0.0
    storage = storage.to("cuda")
    return torch.as_strided(storage, (rows, width), (stride, 1), offset), storage


def _assert_topk_output(
    original: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    largest: bool = True,
    *,
    index_dtype: torch.dtype = torch.int64,
) -> None:
    assert values.shape == indices.shape == (original.size(0), k)
    assert values.dtype == original.dtype and indices.dtype == index_dtype
    assert bool(((indices >= 0) & (indices < original.size(1))).all())
    expected_values = torch.topk(
        original, k, dim=-1, largest=largest, sorted=True
    ).values
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0, equal_nan=True)
    gathered = original.gather(1, indices.to(torch.int64))
    assert torch.equal(values.view(torch.int16), gathered.view(torch.int16))
    sorted_indices = indices.sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "rows,width,k,dtype,lanes,block_rows,vector,padding,offset,output_vector",
    [
        pytest.param(3, 1, 1, torch.bfloat16, 1, 1, 1, 0, 0, 1, id="singleton-min-cta"),
        pytest.param(
            17, 65, 65, torch.float16, 8, 4, 2, 4, 1, 1, id="all-elements-odd-stride"
        ),
        pytest.param(
            35, 65, 3, torch.bfloat16, 32, 32, 4, 2, 1, 1, id="max-cta-odd-stride"
        ),
        pytest.param(9, 33, 17, torch.float16, 2, 2, 4, 0, 1, 1, id="vector4-offset"),
        pytest.param(
            7, 31, 1, torch.bfloat16, 8, 1, 1, 5, 0, 1, id="unit-vector-row-stride"
        ),
        pytest.param(
            17, 64, 3, torch.bfloat16, 16, 8, 2, 0, 0, 1, id="vector2-aligned"
        ),
        pytest.param(
            129, 64, 32, torch.bfloat16, 1, 128, 8, 0, 0, 1, id="one-lane-128-rows-tail"
        ),
        pytest.param(
            65, 128, 32, torch.float16, 2, 64, 8, 0, 0, 1, id="two-lanes-64-rows-tail"
        ),
        pytest.param(5, 64, 32, torch.bfloat16, 4, 4, 8, 0, 0, 2, id="output2-bf16"),
        pytest.param(5, 64, 6, torch.float16, 4, 4, 8, 0, 0, 2, id="output2-fp16-tail"),
        pytest.param(5, 64, 32, torch.bfloat16, 4, 4, 8, 0, 0, 4, id="output4-bf16"),
        pytest.param(
            5, 64, 12, torch.float16, 4, 4, 8, 0, 0, 4, id="output4-fp16-tail"
        ),
        pytest.param(5, 64, 32, torch.bfloat16, 4, 4, 8, 0, 0, 8, id="output8-bf16"),
        pytest.param(
            5, 64, 24, torch.float16, 4, 4, 8, 0, 0, 8, id="output8-fp16-tail"
        ),
        pytest.param(
            5, 64, 17, torch.float16, 4, 4, 8, 2, 1, 8, id="output8-odd-k-offset"
        ),
    ],
)
def test_direct_topk_layout_and_geometry(
    rows: int,
    width: int,
    k: int,
    dtype: torch.dtype,
    lanes: int,
    block_rows: int,
    vector: int,
    padding: int,
    offset: int,
    output_vector: int,
) -> None:
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, True),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=block_rows,
        cute_topk_vector_width=vector,
        cute_topk_output_vector_width=output_vector,
    )
    assert "_cute_local_topk" in code
    assert ("cute.autovec_copy" in code) == (
        output_vector > 1 and k % output_vector == 0
    )
    _assert_topk_output(original, values, indices, k)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("value_mode", ["gather", "decode"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "selection_layout,key_dtype,k,lanes",
    [
        ("replicated", "int32", 32, 4),
        ("distributed", "int32", 32, 4),
        ("distributed", "float32_bits", 6, 2),
        ("distributed", "float32_bits", 8, 32),
    ],
)
def test_topk_output_alignment_changes_on_same_bound_kernel(
    dtype: torch.dtype,
    value_mode: str,
    index_dtype: torch.dtype,
    selection_layout: str,
    key_dtype: str,
    k: int,
    lanes: int,
) -> None:
    rows, width = 5, 64
    x, storage = _layout_input(rows, width, dtype, 0, 0)
    original_storage = storage.clone()
    value_storage = torch.empty(rows * k + 3, dtype=dtype, device="cuda")
    index_storage = torch.empty(rows * k + 3, dtype=index_dtype, device="cuda")
    values = value_storage[: rows * k].view(rows, k)
    indices = index_storage[: rows * k].view(rows, k)
    bound = _extra_out_topk._bind_isolated((x, values, indices, k, True))
    bound.set_config(
        helion.Config(
            block_sizes=[1],
            cute_topk_lanes_per_row=lanes,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=8,
            cute_topk_value_mode=value_mode,
            cute_topk_key_dtype=key_dtype,
            cute_topk_rank_mode="ordinal",
            cute_topk_selection_layout=selection_layout,
        )
    )
    for value_offset, index_offset in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (0, 0)):
        value_storage.fill_(7)
        index_storage.fill_(-7)
        values = value_storage[value_offset : value_offset + rows * k].view(rows, k)
        indices = index_storage[index_offset : index_offset + rows * k].view(rows, k)
        bound(x, values, indices, k, True)
        _assert_topk_output(x, values, indices, k, index_dtype=index_dtype)
        assert bool((value_storage[:value_offset] == 7).all())
        assert bool((value_storage[value_offset + rows * k :] == 7).all())
        assert bool((index_storage[:index_offset] == -7).all())
        assert bool((index_storage[index_offset + rows * k :] == -7).all())
        assert torch.equal(
            storage.view(torch.int16), original_storage.view(torch.int16)
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,output_vector,key_dtype,rank_mode,value_mode,layout,padding,offset",
    [
        (1, 1, 1, 1, "int32", "signed", "gather", "replicated", 0, 0),
        (64, 32, 4, 2, "float32_bits", "ordinal", "decode", "replicated", 0, 0),
        (65, 65, 8, 8, "float32", "signed", "decode", "distributed", 2, 1),
        (64, 6, 2, 2, "int32", "ordinal", "decode", "distributed", 0, 0),
    ],
)
def test_topk_index_output_dtype_preserves_selected_bits(
    index_dtype: torch.dtype,
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    output_vector: int,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
    layout: str,
    padding: int,
    offset: int,
) -> None:
    rows = 5
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original_storage = storage.clone()
    values = torch.full((rows, k), 7, dtype=dtype, device="cuda")
    indices = torch.full((rows, k), -7, dtype=index_dtype, device="cuda")
    code, _output = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_selection_layout=layout,
    )
    helper = "_cute_distributed_topk" if layout == "distributed" else "_cute_local_topk"
    # Singleton top-k simplifies to a value copy and index zero.
    if width > 1:
        assert helper in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_topk_narrow_indices_preserve_wide_address_math(
    index_dtype: torch.dtype,
) -> None:
    x, storage = _layout_input(1, 8, torch.bfloat16, 0, 0)
    x = torch.as_strided(x, (1, 8), (2**35, 1))
    original_storage = storage.clone()
    values = torch.empty((1, 8), dtype=x.dtype, device="cuda")
    indices = torch.empty((1, 8), dtype=index_dtype, device="cuda")
    code, _output = code_and_output(
        _extra_out_topk,
        (x, values, indices, 8, True),
        block_sizes=[1],
        cute_topk_lanes_per_row=1,
        cute_topk_rows_per_block=1,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=8,
    )
    assert "topk_row * cutlass.Int64(34359738368)" in code
    _assert_topk_output(x, values, indices, 8, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize(
    "width,k,output_vector,padding,offset",
    [(64, 64, 8, 0, 0), (65, 65, 1, 2, 1), (64, 17, 4, 2, 1)],
)
def test_topk_decode_preserves_selected_bits(
    dtype: torch.dtype,
    largest: bool,
    rank_mode: str,
    width: int,
    k: int,
    output_vector: int,
    padding: int,
    offset: int,
) -> None:
    x, storage = _layout_input(5, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=4,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode="decode",
        cute_topk_rank_mode=rank_mode,
    )
    assert "topk_value_decodable" in code
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize(
    "key_dtype,width,k,output_vector,padding,offset,value_mode",
    [
        ("float32", 512, 32, 8, 0, 0, "decode"),
        ("float32", 65, 65, 1, 2, 1, "gather"),
        ("float32", 1024, 32, 4, 0, 1, "decode"),
        ("float32_bits", 1024, 32, 8, 0, 0, "decode"),
        ("float32_bits", 65, 65, 1, 2, 1, "gather"),
        ("float32_bits", 1, 1, 1, 0, 0, "decode"),
    ],
)
def test_topk_float_keys_preserve_selected_bits(
    dtype: torch.dtype,
    largest: bool,
    rank_mode: str,
    key_dtype: str,
    width: int,
    k: int,
    output_vector: int,
    padding: int,
    offset: int,
    value_mode: str,
) -> None:
    x, storage = _layout_input(5, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=4,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
    )
    assert "_cute_local_topk" in code
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,block_rows,padding,offset,key_dtype,rank_mode,value_mode",
    [
        (1, 1, 1, 1, 0, 0, "int32", "ordinal", "decode"),
        (65, 33, 1, 8, 2, 1, "int32", "ordinal", "decode"),
        (65, 6, 2, 8, 2, 1, "int32", "ordinal", "decode"),
        (128, 32, 2, 64, 0, 0, "float32_bits", "ordinal", "decode"),
        (128, 32, 4, 32, 0, 0, "float32", "signed", "decode"),
        (128, 64, 8, 16, 0, 0, "int32", "ordinal", "gather"),
        (128, 24, 8, 16, 2, 1, "float32_bits", "ordinal", "decode"),
        (128, 12, 8, 16, 0, 0, "int32", "signed", "decode"),
        (256, 128, 16, 8, 0, 0, "float32_bits", "ordinal", "decode"),
        (128, 64, 32, 4, 0, 0, "float32", "signed", "decode"),
        (65, 65, 4, 32, 2, 1, "float32", "signed", "gather"),
        (64, 7, 8, 4, 0, 0, "int32", "ordinal", "decode"),
        (128, 17, 16, 4, 0, 0, "float32_bits", "ordinal", "decode"),
        (65, 31, 32, 1, 2, 1, "float32", "signed", "decode"),
        (64, 3, 32, 1, 0, 0, "int32", "ordinal", "gather"),
    ],
)
def test_topk_distributed_selection(
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    block_rows: int,
    padding: int,
    offset: int,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
) -> None:
    x, storage = _layout_input(block_rows + 1, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=block_rows,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=8,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_selection_layout="distributed",
    )
    assert "_cute_distributed_topk" in code
    vector = min(8, lanes, max(1, (1 << (k - 1).bit_length()) // lanes))
    assert ("cute.autovec_copy" in code) == (vector > 1 and k % vector == 0)
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,input_vector,key_dtype,rank_mode,value_mode",
    [
        (8, 8, 8, 1, "int32", "ordinal", "decode"),
        (13, 9, 16, 1, "float32_bits", "signed", "decode"),
        (64, 32, 32, 2, "float32", "ordinal", "gather"),
        (128, 32, 16, 8, "int32", "ordinal", "decode"),
        (128, 32, 32, 4, "float32_bits", "ordinal", "decode"),
        (65, 33, 32, 4, "float32", "signed", "decode"),
    ],
)
def test_topk_distributed_growing_selection(
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    input_vector: int,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
) -> None:
    # Misaligned, strided rows and a partial CTA exercise the smaller fragment
    # with both real values and padding through the growing subgroup stages.
    x, storage = _layout_input(5, width, dtype, 2, 1)
    original = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=input_vector,
        cute_topk_output_vector_width=8,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_value_mode=value_mode,
        cute_topk_selection_layout="distributed",
    )
    assert "_cute_distributed_topk" in code
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,input_vector,key_dtype,network",
    [
        (256, 8, 16, 8, "int32", "batcher"),
        (256, 8, 32, 8, "float32", "compact"),
        (512, 8, 32, 8, "float32_bits", "compact_pruned"),
        (128, 3, 16, 8, "int32", "compact_pruned"),
        (33, 1, 32, 4, "float32", "batcher"),
        (3, 3, 32, 1, "float32_bits", "compact"),
    ],
)
def test_topk_distributed_more_lanes_than_outputs(
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    input_vector: int,
    key_dtype: str,
    network: str,
) -> None:
    x, storage = _layout_input(5, width, dtype, 2, 1)
    original = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=input_vector,
        cute_topk_output_vector_width=8,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout="distributed",
    )
    assert "_cute_distributed_topk" in code
    assert "cute.autovec_copy" not in code
    assert f"topk_output_col < cutlass.Int32({k})" in code
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


@pytest.fixture
def topk_spec(monkeypatch: pytest.MonkeyPatch) -> ConfigSpec:
    # Keep configuration validation independent of CUDA device discovery.
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1)
    spec.enable_cute_topk_search()
    return spec


@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_topk_lanes_per_row", True),
        ("cute_topk_rows_per_block", False),
        ("cute_topk_vector_width", True),
        ("cute_topk_lanes_per_row", 0),
        ("cute_topk_rows_per_block", 3),
        ("cute_topk_vector_width", 16),
        ("cute_topk_vector_width", 8.0),
        ("cute_topk_output_vector_width", True),
        ("cute_topk_output_vector_width", 16),
        ("cute_topk_value_mode", True),
        ("cute_topk_value_mode", 1),
        ("cute_topk_value_mode", "unknown"),
        ("cute_topk_key_dtype", True),
        ("cute_topk_key_dtype", 1),
        ("cute_topk_key_dtype", "unknown"),
        ("cute_topk_rank_mode", True),
        ("cute_topk_rank_mode", 1),
        ("cute_topk_rank_mode", "unknown"),
        ("cute_topk_selection_layout", True),
        ("cute_topk_selection_layout", 1),
        ("cute_topk_selection_layout", "unknown"),
    ],
)
def test_topk_config_rejects_noninteger_or_unsupported_choices(
    topk_spec: ConfigSpec, key: str, value: object
) -> None:
    config = helion.Config.from_dict({key: value})
    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        topk_spec.normalize(config)


def test_topk_config_is_scoped_to_matched_cute_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    cute_spec = ConfigSpec(
        backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1
    )
    with pytest.raises(exc.InvalidConfig, match="compatible top-k root"):
        cute_spec.normalize({"cute_topk_lanes_per_row": 16})
    triton_spec = ConfigSpec(backend=TritonBackend(), num_sm=1)
    with pytest.raises(exc.InvalidConfig, match="Unsupported config keys"):
        triton_spec.normalize({"cute_topk_lanes_per_row": 16})


@pytest.mark.parametrize("lanes,rows", [(2, 4), (1, 128), (2, 64), (8, 128), (16, 64)])
def test_topk_geometry_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, lanes: int, rows: int
) -> None:
    config = helion.Config(
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=rows,
        cute_topk_vector_width=1,
    )
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


@pytest.mark.parametrize("vector", [1, 2, 4, 8])
def test_topk_output_vector_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, vector: int
) -> None:
    config = helion.Config(cute_topk_output_vector_width=vector)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


@pytest.mark.parametrize("mode", ["gather", "decode"])
def test_topk_value_mode_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, mode: str
) -> None:
    assert topk_spec.default_config()["cute_topk_value_mode"] == "gather"
    config = helion.Config(cute_topk_value_mode=mode)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_value_mode_normalizes_to_gather(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_value_mode=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_value_mode"] == "gather"


@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
def test_topk_key_dtype_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, key_dtype: str
) -> None:
    assert topk_spec.default_config()["cute_topk_key_dtype"] == "int32"
    config = helion.Config(cute_topk_key_dtype=key_dtype)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_key_dtype_normalizes_to_int32(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_key_dtype=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_key_dtype"] == "int32"


@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
def test_topk_rank_mode_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, rank_mode: str
) -> None:
    assert topk_spec.default_config()["cute_topk_rank_mode"] == "signed"
    config = helion.Config(cute_topk_rank_mode=rank_mode)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_rank_mode_normalizes_to_signed(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_rank_mode=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_rank_mode"] == "signed"


@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_selection_layout_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, layout: str
) -> None:
    assert topk_spec.default_config()["cute_topk_selection_layout"] == "replicated"
    config = helion.Config(cute_topk_selection_layout=layout)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_selection_layout_normalizes_to_replicated(
    topk_spec: ConfigSpec,
) -> None:
    config = helion.Config(cute_topk_selection_layout=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_selection_layout"] == "replicated"


@pytest.mark.parametrize("lanes,rows", [(16, 128), (32, 64), (32, 128)])
def test_topk_config_limits_threads_per_block(
    topk_spec: ConfigSpec, lanes: int, rows: int
) -> None:
    config = helion.Config(
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=rows,
    )
    with pytest.raises(exc.InvalidConfig, match="must not exceed 1024 threads"):
        topk_spec.normalize(config)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_lanes_per_row"] == lanes
    assert config["cute_topk_rows_per_block"] == 1024 // lanes
    topk_spec.normalize(config)


# Safety top-k tests.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _allocating_topk(
    x: torch.Tensor,
    k: int,
    largest: hl.constexpr,
    index_dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=index_dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transformed_indices_topk(
    x: torch.Tensor,
    k: int,
    largest: hl.constexpr,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=index_dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx + 1
    return values, indices


def _code(
    rows: int,
    cols: int,
    stride: int,
    k: int,
    output_vector: int = 1,
    *,
    dtype: torch.dtype = torch.bfloat16,
    largest: bool = True,
    value_mode: str = "gather",
    key_dtype: str = "int32",
    rank_mode: str = "signed",
    selection_layout: str = "replicated",
    lanes: int = 16,
    input_vector: int = 8,
    sort_network: str = "batcher",
    key_encoder: str = "dsl",
    defer_value_gathers: bool = False,
    merge_schedule: str = "sequential",
    index_dtype: torch.dtype = torch.int64,
    kernel: helion.Kernel[Any] = _allocating_topk,
) -> str:
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        with FakeTensorMode():
            x = torch.empty_strided(
                (rows, cols), (stride, 1), dtype=dtype, device="cpu"
            )
        bound = kernel._bind_isolated((x, k, largest, index_dtype))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_topk_lanes_per_row=lanes,
            cute_topk_rows_per_block=8,
            cute_topk_vector_width=input_vector,
            cute_topk_output_vector_width=output_vector,
            cute_topk_value_mode=value_mode,
            cute_topk_key_dtype=key_dtype,
            cute_topk_rank_mode=rank_mode,
            cute_topk_selection_layout=selection_layout,
        )
        return bound.to_triton_code(config)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vector", [1, 2, 4, 8])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_index_output_dtype_codegen(
    index_dtype: torch.dtype, vector: int, layout: str
) -> None:
    code = _code(
        5,
        128,
        128,
        64,
        vector,
        index_dtype=index_dtype,
        selection_layout=layout,
        lanes=8,
    )
    dtype = "cutlass.Int32" if index_dtype == torch.int32 else "cutlass.Int64"
    element_bytes = 4 if index_dtype == torch.int32 else 8
    assert f"indices[topk_row, topk_output_col] = {dtype}(topk_selected_index)" in code
    if vector > 1:
        assert f"topk_index_fragment = cute.make_rmem_tensor({vector}, {dtype})" in code
        assert (
            f"indices.iterator.alignment >= {min(16, element_bytes * vector)}" in code
        )
    assert "topk_row = cutlass.Int32(" in code


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows,stride", [(1, 2**35), (268435457, 0)])
def test_topk_output_dtype_does_not_narrow_addresses(
    index_dtype: torch.dtype, rows: int, stride: int
) -> None:
    code = _code(rows, 8, stride, 8, 2, index_dtype=index_dtype)
    assert "cutlass.Int64(cute.arch.block_idx()[0])" in code
    assert f"topk_row * cutlass.Int64({stride})" in code
    assert "topk_output_offset = topk_row * cutlass.Int64(8)" in code


@pytest.mark.parametrize("index_dtype", [torch.int16, torch.float32])
def test_topk_matcher_rejects_other_index_store_dtypes(
    index_dtype: torch.dtype,
) -> None:
    with pytest.raises(exc.InvalidConfig, match="requires a compatible top-k root"):
        _code(5, 64, 64, 32, index_dtype=index_dtype)


def test_topk_matcher_rejects_arithmetic_before_index_narrowing() -> None:
    with pytest.raises(exc.InvalidConfig, match="requires a compatible top-k root"):
        _code(5, 64, 64, 32, index_dtype=torch.int32, kernel=_transformed_indices_topk)


@pytest.mark.parametrize(
    "rows,cols,stride,k,wide",
    [
        (17, 8, 8, 3, False),
        (1, 8, 2**35, 3, True),
        (65536, 8, 32768, 3, False),
        (65537, 8, 32768, 3, True),
        (268435457, 8, 0, 8, True),
    ],
)
def test_topk_wide_address_codegen(
    rows: int, cols: int, stride: int, k: int, wide: bool
) -> None:
    code = _code(rows, cols, stride, k)
    dtype = "cutlass.Int64" if wide else "cutlass.Int32"
    assert f"{dtype}(cute.arch.block_idx()[0])" in code
    assert f"* {dtype}({stride})" in code
    assert "_cute_local_topk_" in code


@pytest.mark.parametrize(
    "rows,cols,stride,k,vector,wide",
    [
        (17, 64, 64, 32, 2, False),
        (17, 64, 64, 32, 4, False),
        (17, 64, 64, 32, 8, False),
        (1, 8, 2**35, 8, 8, True),
        (268435457, 8, 0, 8, 8, True),
    ],
)
def test_topk_output_vector_address_codegen(
    rows: int, cols: int, stride: int, k: int, vector: int, wide: bool
) -> None:
    code = _code(rows, cols, stride, k, vector)
    dtype = "cutlass.Int64" if wide else "cutlass.Int32"
    assert f"topk_row * {dtype}({k}) + {dtype}(topk_output_col)" in code
    assert f"values.iterator.alignment >= {2 * vector}" in code
    assert "indices.iterator.alignment >= 16" in code
    assert code.count("cute.autovec_copy") == 2
    # Preserve the dynamic offset's divisibility through pointer addition.
    # The address arithmetic must retain its selected width before the hint.
    offset = f"topk_row * {dtype}({k}) + {dtype}(topk_output_col)"
    assumption = f"cute.assume(topk_output_offset, divby={vector})"
    assert code.index(offset) < code.index(assumption)
    assert code.index(assumption) < code.index("values.iterator + topk_output_offset")
    assert code.index(assumption) < code.index("indices.iterator + topk_output_offset")
    assert k % vector == 0
    assert f"topk_lane * cutlass.Int32({vector})" in code


def test_topk_odd_output_stride_uses_scalar_stores() -> None:
    code = _code(17, 64, 64, 17, 8)
    assert "_cute_local_topk" in code
    assert "cute.autovec_copy" not in code
    assert "cute.assume(topk_output_offset" not in code


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("key_encoder", ["dsl"])
def test_topk_decode_all_16bit_inputs(
    dtype: torch.dtype, largest: bool, rank_mode: str, key_encoder: str
) -> None:
    code = _code(
        1,
        64,
        64,
        32,
        dtype=dtype,
        largest=largest,
        value_mode="decode",
        rank_mode=rank_mode,
        key_encoder=key_encoder,
    )
    tree = ast.parse(code)
    encoder = None
    if key_encoder == "asm":
        packed_assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "topk_packed"
        )
        assert isinstance(packed_assignment.value, ast.Call)
        encoder_call = packed_assignment.value
        assert ast.unparse(encoder_call.func).startswith("_cute_encode_ordered_topk_")
        assert [ast.literal_eval(arg) for arg in encoder_call.args[2:]] == [
            6,
            largest,
            rank_mode,
            0x7F80 if dtype == torch.bfloat16 else 0x7C00,
        ]
    else:
        encode_body = next(
            node.body
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "topk_magnitude"
                for stmt in node.body
            )
        )
        encode_start = next(
            i
            for i, stmt in enumerate(encode_body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "topk_magnitude"
        )
        encode_end = next(
            i
            for i, stmt in enumerate(encode_body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "topk_packed"
        )
        encoder = compile(
            ast.Module(
                body=encode_body[encode_start : encode_end + 1], type_ignores=[]
            ),
            "<topk-encode>",
            "exec",
        )
    names = {
        "topk_value_rank",
        "topk_value_sign",
        "topk_value_magnitude",
        "topk_value_bits",
        "topk_value_decodable",
    }
    assignments: list[ast.stmt] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert len(assignments) == (3 if rank_mode == "ordinal" else 5) + (not largest)
    decoder = compile(
        ast.Module(body=assignments, type_ignores=[]), "<topk-decode>", "exec"
    )
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    namespace = {"cutlass": SimpleNamespace(Int32=int)}
    ranks = []
    for word in range(1 << 16):
        magnitude = word & 0x7FFF
        rank = magnitude if word < 32768 else -magnitude - (rank_mode == "ordinal")
        if magnitude > infinity:
            rank = 32767
        if not largest:
            rank = -rank
        packed = (rank << 6) | (63 - (word & 63))
        if encoder is not None:
            encoded = {
                "topk_bits": word if word < 32768 else word - 65536,
                "topk_col": word & 63,
            }
            exec(encoder, namespace, encoded)
            assert encoded["topk_packed"] == packed
        assert packed & 63 == 63 - (word & 63)
        ranks.append(packed >> 6)
        local = {"topk_local": [packed], "topk_j": 0}
        exec(decoder, namespace, local)
        decodable = magnitude <= infinity and (rank_mode == "ordinal" or magnitude != 0)
        assert local["topk_value_decodable"] == decodable
        if decodable:
            # Uint16 truncates the recovered signed representation.
            bits = local["topk_value_bits"]
            assert isinstance(bits, int)
            assert bits & 65535 == word
    # These bounds justify both Float32 key guards for either ranking mode.
    assert min(ranks) >= -32767 and max(ranks) <= 32767
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = words.view(dtype).float()
    finite_or_inf = ~torch.isnan(values)
    order = torch.tensor(ranks)[finite_or_inf].argsort(descending=True)
    ordered_values = values[finite_or_inf][order]
    assert bool(
        (
            ordered_values[:-1] >= ordered_values[1:]
            if largest
            else ordered_values[:-1] <= ordered_values[1:]
        ).all()
    )
    if rank_mode == "ordinal":
        assert ranks[0] == 0
        assert ranks[32768] == (-1 if largest else 1)


def test_topk_decode_keeps_wide_vector_output_addresses() -> None:
    code = _code(268435457, 8, 0, 8, 8, value_mode="decode")
    assert "topk_row * cutlass.Int64(8) + cutlass.Int64(topk_output_col)" in code
    assert code.count("topk_value_decodable =") == 2
    assert "values.iterator.alignment >= 16" in code


@pytest.mark.parametrize(
    "cols,key_dtype,selected_dtype",
    [
        (512, "int32", "Int32"),
        (512, "float32", "Float32"),
        (513, "float32", "Int32"),
        (1024, "float32", "Int32"),
        (1, "float32_bits", "Float32"),
        (1024, "float32_bits", "Float32"),
        (16384, "float32_bits", "Float32"),
        (16385, "float32_bits", "Int32"),
        (32768, "float32_bits", "Int32"),
    ],
)
def test_topk_float_key_codegen_precision_guard(
    cols: int, key_dtype: str, selected_dtype: str
) -> None:
    code = _code(
        5, cols, cols, min(cols, 32), 8, key_dtype=key_dtype, value_mode="decode"
    )
    tree = ast.parse(code)
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "topk_keys"
    )
    assert ast.unparse(assignment.value).endswith(f"cutlass.{selected_dtype})")
    if selected_dtype == "Int32":
        assert "topk_keys.fill(cutlass.Int32(-2147483648))" in code
        assert "topk_keys[topk_i] = topk_packed" in code
    elif key_dtype == "float32":
        assert "topk_keys.fill(cutlass.Float32(-2147483648))" in code
        assert "topk_keys[topk_i] = cutlass.Float32(topk_packed)" in code
        assert code.count("cutlass.Int32(topk_selected[topk_output])") == 2
    else:
        assert "topk_keys.fill(cutlass.Float32(0.0))" in code
        assert (
            "(topk_packed + cutlass.Int32(1073741824)).bitcast(cutlass.Float32)" in code
        )
        assert (
            "topk_selected[topk_output].bitcast(cutlass.Int32) - cutlass.Int32(1073741824)"
            in code
        )


def test_topk_numeric_float_key_exactness_boundary() -> None:
    # Every integer in [-2**24, 2**24] has an exact Float32 representation.
    # Check both rank signs and payload endpoints at the widest enabled key.
    ranks = torch.arange(-32767, 32768, dtype=torch.int32)
    payloads = torch.tensor([0, 1, 510, 511], dtype=torch.int32)
    keys = ((ranks[:, None] << 9) | payloads).flatten()
    floats = keys.to(torch.float32)
    assert bool((keys.abs() <= 2**24).all())
    assert torch.equal(floats.to(torch.int32), keys)
    assert bool((floats[1:] > floats[:-1]).all())
    padding = torch.tensor(-2147483648, dtype=torch.int32)
    assert padding.float().to(torch.int32) == padding
    assert bool((floats > padding.float()).all())
    # At ten payload bits a representable rank can lose its tie-breaking bit.
    too_wide = torch.tensor((32767 << 10) | 1, dtype=torch.int32)
    assert too_wide.float().to(torch.int32) != too_wide


def test_topk_biased_float_key_exactness_boundary() -> None:
    # All valid biased keys are positive normal finite floats. Their IEEE
    # representation therefore preserves integer order without conversions.
    ranks = torch.arange(-32767, 32768, dtype=torch.int32)
    payloads = torch.tensor([0, 1, 16382, 16383], dtype=torch.int32)
    keys = ((ranks[:, None] << 14) | payloads).flatten()
    floats = (keys + 0x40000000).view(torch.float32)
    assert bool(torch.isfinite(floats).all())
    assert bool((floats >= torch.finfo(torch.float32).tiny).all())
    assert bool((floats[1:] > floats[:-1]).all())
    assert bool((floats > 0.0).all())  # Padding cannot displace any real key.
    assert torch.equal(floats.view(torch.int32) - 0x40000000, keys)
    # One more index bit admits NaN encodings and must use the Int32 fallback.
    too_wide = torch.tensor(((32767 << 15) | 32767) + 0x40000000, dtype=torch.int32)
    assert bool(torch.isnan(too_wide.view(torch.float32)))


def _simulate_distributed_topk(inputs: list[list[int]], k: int) -> list[int]:
    """Simulate cyclic subgroup exchanges, independently of CuTe execution."""
    lanes = len(inputs)
    selected = [sorted(values, reverse=True)[:k] for values in inputs]
    for stage in range(1, lanes.bit_length()):
        half = 1 << (stage - 1)
        group = 2 * half
        previous_size = len(selected[0])
        grow = previous_size * group <= k
        merge_group = group
        if previous_size == 1 and group > k:
            merged = [
                [max(selected[lane][0], selected[lane ^ (group - 1)][0])]
                for lane in range(lanes)
            ]
            halves = [
                [
                    merged[lane ^ (k - 1)] if lane & half else merged[lane]
                    for lane in range(lanes)
                ]
            ]
            merge_group = k
        elif grow and previous_size == 1:
            halves = [
                [
                    selected[lane ^ (half - 1)] if lane & half else selected[lane]
                    for lane in range(lanes)
                ]
            ]
        else:
            halves = []
            for compare in (max, min) if grow else (max,):
                merged = []
                for lane in range(lanes):
                    peer_lane = lane ^ (group - 1)
                    fragment = []
                    for index in range(previous_size // 2):
                        own_index = (
                            previous_size - 2 - 2 * index if lane & half else 2 * index
                        )
                        peer_index = (
                            previous_size - 1 - 2 * index
                            if peer_lane & half
                            else 2 * index + 1
                        )
                        fragment.append(
                            compare(
                                selected[lane][own_index],
                                selected[peer_lane][peer_index],
                            )
                        )
                    merged.append(fragment)
                # Reverse only the upper half to establish cyclic rank ownership.
                halves.append(
                    [
                        merged[lane ^ (half - 1)] if lane & half else merged[lane]
                        for lane in range(lanes)
                    ]
                )
        for merged in halves:
            distance = len(merged[0]) * merge_group // 2
            while distance:
                previous = [fragment[:] for fragment in merged]
                for lane in range(lanes):
                    for index in range(len(merged[0])):
                        rank = index * merge_group + lane % merge_group
                        peer = (
                            previous[lane][index ^ (distance // merge_group)]
                            if distance >= merge_group
                            else previous[lane ^ distance][index]
                        )
                        compare = min if rank & distance else max
                        merged[lane][index] = compare(previous[lane][index], peer)
                distance //= 2
        selected = [
            list(itertools.chain.from_iterable(part[lane] for part in halves))
            for lane in range(lanes)
        ]
    return [selected[rank % lanes][rank // lanes] for rank in range(k)]


@pytest.mark.parametrize("k,lanes", [(2, 2), (4, 4), (8, 4)])
def test_distributed_topk_zero_one_network(k: int, lanes: int) -> None:
    # After local sorting, every binary fragment is characterized by its
    # number of ones; this exhausts all binary inputs without permutations.
    for counts in itertools.product(range(k + 1), repeat=lanes):
        inputs = [[1] * count + [0] * (k - count) for count in counts]
        expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
        assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
def test_distributed_topk_random_keys_and_padding(lanes: int) -> None:
    generator = random.Random(20260925)
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        if k < lanes:
            continue
        for _trial in range(16):
            inputs = [
                [
                    generator.choice(
                        (-2147483648, 0, 1, generator.randrange(-100000, 100000))
                    )
                    for _index in range(2 * k)
                ]
                for _lane in range(lanes)
            ]
            expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
            assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize(
    "size,lanes,k", [(1, 8, 8), (2, 4, 8), (2, 8, 8), (4, 4, 8), (4, 4, 16)]
)
def test_distributed_topk_growing_zero_one_network(
    size: int, lanes: int, k: int
) -> None:
    for counts in itertools.product(range(size + 1), repeat=lanes):
        inputs = [[1] * count + [0] * (size - count) for count in counts]
        expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
        assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_distributed_topk_growing_random_and_padding(lanes: int) -> None:
    generator = random.Random(20260926)
    for size in (1, 2, 4, 8, 16, 32, 64):
        for k in (2, 4, 8, 16, 32, 64, 128):
            if not lanes <= k <= size * lanes or size >= k:
                continue
            for _trial in range(16):
                inputs = [
                    [
                        generator.choice(
                            (-2147483648, -1, 0, 1, generator.randrange(-10000, 10000))
                        )
                        for _index in range(size)
                    ]
                    for _lane in range(lanes)
                ]
                expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[
                    :k
                ]
                assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("size,lanes,k", [(1, 8, 1), (1, 8, 2), (2, 8, 4), (4, 4, 2)])
def test_distributed_topk_wide_zero_one_network(size: int, lanes: int, k: int) -> None:
    for counts in itertools.product(range(size + 1), repeat=lanes):
        inputs = [[1] * count + [0] * (size - count) for count in counts]
        expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
        assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_distributed_topk_wider_than_k_random(lanes: int) -> None:
    generator = random.Random(20260928)
    for k in (1, 2, 4, 8, 16):
        if k >= lanes:
            continue
        for size in (1, 2, 4, 8, 16, 32, 64):
            for _trial in range(32):
                inputs = [
                    [
                        generator.choice(
                            (-2147483648, -1, 0, 1, generator.randrange(-10000, 10000))
                        )
                        for _index in range(size)
                    ]
                    for _lane in range(lanes)
                ]
                expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[
                    :k
                ]
                assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize(
    "n,k,lanes,vector,fragment",
    [
        (128, 32, 16, 8, 8),
        (128, 32, 32, 8, 8),
        (128, 32, 32, 4, 4),
        (8, 8, 8, 1, 1),
        (13, 9, 16, 1, 1),
        (64, 32, 32, 2, 2),
    ],
)
def test_distributed_fragment_does_not_pad_every_lane_to_k(
    n: int, k: int, lanes: int, vector: int, fragment: int
) -> None:
    code = _code(
        9, n, n, k, selection_layout="distributed", lanes=lanes, input_vector=vector
    )
    assert "_cute_distributed_topk" in code
    assert f"topk_keys = cute.make_rmem_tensor({fragment}, cutlass.Int32)" in code


@pytest.mark.parametrize(
    "k,lanes",
    [(3, 8), (3, 4), (6, 2), (33, 1), (32, 16), (1, 32), (8, 32)],
)
@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
def test_topk_distributed_codegen_guard(k: int, lanes: int, key_dtype: str) -> None:
    code = _code(
        5,
        64,
        64,
        k,
        8,
        lanes=lanes,
        key_dtype=key_dtype,
        selection_layout="distributed",
        value_mode="decode",
        rank_mode="ordinal",
    )
    assert "_cute_distributed_topk" in code
    assert "topk_selected_key =" in code
    assert "topk_local = cute.make_rmem_tensor" not in code
    vector = min(8, lanes, max(1, (1 << (k - 1).bit_length()) // lanes))
    assert ("cute.autovec_copy" in code) == (vector > 1 and k % vector == 0)
    assert f"topk_j * {min(lanes, 1 << (k - 1).bit_length())}" in code
    if k != 1 << (k - 1).bit_length() or k < lanes:
        assert f"topk_output_col < cutlass.Int32({k})" in code


@pytest.mark.parametrize("rows,stride", [(1, 2**35), (268435457, 0)])
def test_topk_distributed_keeps_wide_addresses(rows: int, stride: int) -> None:
    code = _code(
        rows,
        8,
        stride,
        8,
        8,
        selection_layout="distributed",
        lanes=4,
        value_mode="decode",
        rank_mode="ordinal",
    )
    assert "_cute_distributed_topk" in code
    assert "cutlass.Int64(cute.arch.block_idx()[0])" in code
    assert f"* cutlass.Int64({stride})" in code
    assert "values[topk_row, topk_output_col]" in code
    assert "indices[topk_row, topk_output_col]" in code
    assert "topk_row * cutlass.Int64(8) + cutlass.Int64(topk_output_col)" in code
    assert "cute.assume(topk_output_offset, divby=2)" in code


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("requested_vector", [1, 2, 4, 8])
def test_distributed_output_transpose_rank_mapping(
    lanes: int, requested_vector: int
) -> None:
    # Label every register with its original global rank. This verifies the
    # permutation for any key values, including duplicates and padding.
    for local_size in (1, 2, 4, 8, 16, 32, 64):
        vector = min(requested_vector, lanes, local_size)
        ranks = [
            [index * lanes + lane for index in range(local_size)]
            for lane in range(lanes)
        ]
        for block in range(local_size // vector):
            for stage in range(vector.bit_length() - 1):
                bit = 1 << stage
                previous = [registers[:] for registers in ranks]
                for lane in range(lanes):
                    for register in range(vector):
                        if register & bit:
                            continue
                        low = block * vector + register
                        high = low + bit
                        ranks[lane][low] = (
                            previous[lane ^ bit][high]
                            if lane & bit
                            else previous[lane][low]
                        )
                        ranks[lane][high] = (
                            previous[lane][high]
                            if lane & bit
                            else previous[lane ^ bit][low]
                        )
        for lane in range(lanes):
            output_lane = (lane % vector) * (lanes // vector) + lane // vector
            expected = [
                block * lanes * vector + output_lane * vector + element
                for block in range(local_size // vector)
                for element in range(vector)
            ]
            assert ranks[lane] == expected


@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
@pytest.mark.parametrize(
    "k,lanes,requested,effective",
    [
        (32, 1, 8, 1),
        (32, 32, 8, 1),
        (32, 2, 8, 2),
        (32, 4, 8, 4),
        (64, 8, 8, 8),
        (24, 8, 8, 4),
        (12, 8, 8, 2),
        (6, 2, 8, 2),
        (3, 2, 8, 1),
        (32, 4, 1, 1),
        (64, 8, 2, 2),
        (64, 8, 4, 4),
    ],
)
def test_distributed_vector_codegen(
    k: int, lanes: int, requested: int, effective: int, key_dtype: str
) -> None:
    code = _code(
        9,
        128,
        130,
        k,
        requested,
        lanes=lanes,
        selection_layout="distributed",
        key_dtype=key_dtype,
        value_mode="decode",
        rank_mode="ordinal",
    )
    assert ("_cute_transpose_topk_output" in code) == (effective > 1)
    assert ("cute.autovec_copy" in code) == (effective > 1)
    if effective == 1:
        return
    assert f"values.iterator.alignment >= {2 * effective}" in code
    assert "indices.iterator.alignment >= 16" in code
    assert f"cute.assume(topk_output_offset, divby={effective})" in code
    assert f"topk_output_lane * cutlass.Int32({effective})" in code
    if k != 1 << (k - 1).bit_length():
        assert f"topk_output_col < cutlass.Int32({k})" in code
    # Scalar ABI fallback must retain the original cyclic fragment, while
    # Float32 keys are only converted back after the transpose.
    assert "topk_selected[topk_j]" in code
    vector_key = f"topk_output_keys[topk_j * {effective} + topk_v]"
    if key_dtype == "float32":
        assert f"cutlass.Int32({vector_key})" in code
    elif key_dtype == "float32_bits":
        assert f"{vector_key}.bitcast(cutlass.Int32)" in code
    else:
        assert f"topk_selected_key = {vector_key}" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _out_topk(x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> None:
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], x.size(1), dim=-1, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx


@pytest.mark.parametrize("alias_kind", ["separate", "view", "dlpack"])
@pytest.mark.parametrize("alias_target", ["values", "indices"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_matcher_uses_runtime_span_alias_proof(
    alias_kind: str, alias_target: str, index_dtype: torch.dtype
) -> None:
    x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    indices = torch.empty_like(x, dtype=index_dtype)
    if alias_target == "indices" and alias_kind != "separate":
        x = indices.view(torch.bfloat16).flatten()[: x.numel()].view_as(x)
    values = x.clone()
    if alias_target == "values" and alias_kind != "separate":
        values = x.view_as(x) if alias_kind == "view" else torch.from_dlpack(x)
    elif alias_target == "indices" and alias_kind == "dlpack":
        indices = torch.from_dlpack(indices)
    aliased = values if alias_target == "values" else indices
    if alias_kind == "dlpack":
        assert x.data_ptr() == aliased.data_ptr()
        assert x.untyped_storage()._cdata != aliased.untyped_storage()._cdata
        with FakeTensorMode() as mode:
            fx, fa = mode.from_tensor(x), mode.from_tensor(aliased)
            assert fx.untyped_storage()._cdata != fa.untyped_storage()._cdata
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        bound = _out_topk._bind_isolated((x, values, indices))
        code = bound.to_triton_code(bound.config_spec.default_config())
        assert ("_cute_local_topk" in code) == (alias_kind == "separate")
