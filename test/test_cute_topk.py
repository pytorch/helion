"""CuTe top-k lowering, selection networks, tuning, and correctness tests."""

from __future__ import annotations

import ast
import copy
import dataclasses
import gc
import itertools
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteTopKHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.memory_ops import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
from helion._compiler.cute.topk import match_topk_root
from helion._compiler.cute.topk import topk_tensors_are_proven_disjoint
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion.runtime.cute.sorting_networks import COMPACT_SORT_LAYERS
from helion.runtime.cute.topk import _balanced_chunk_program
from helion.runtime.cute.topk import _odd_even_sort_network
from helion.runtime.cute.topk import _pruned_sort_network
from helion.runtime.cute.topk import _sort_network
from helion.runtime.cute.topk import _use_pruned_sort_network

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
    storage = torch.randn(rows * width + 1, dtype=torch.bfloat16, device=DEVICE)
    bits = storage.view(torch.int16)
    bits[:8] = torch.tensor(
        [0x7FC1, -46, -32768, 0, -128, 0x7F80, 0x3F80, -16512],
        dtype=torch.int16,
        device=DEVICE,
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
    value_storage = torch.empty(rows * k + 3, dtype=dtype, device=DEVICE)
    index_storage = torch.empty(rows * k + 3, dtype=index_dtype, device=DEVICE)
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
        (128, 24, 8, 8, "float32_native", "ordinal", "decode", "distributed", 2, 1),
        (128, 8, 32, 8, "float32_native", "signed", "gather", "distributed", 0, 0),
        (64, 6, 2, 2, "int32", "ordinal", "decode", "distributed", 0, 0),
        (128, 32, 8, 4, "float32_native", "ordinal", "decode", "replicated", 0, 0),
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
    values = torch.full((rows, k), 7, dtype=dtype, device=DEVICE)
    indices = torch.full((rows, k), -7, dtype=index_dtype, device=DEVICE)
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
        cute_topk_sort_network="compact_pruned",
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
    values = torch.empty((1, 8), dtype=x.dtype, device=DEVICE)
    indices = torch.empty((1, 8), dtype=index_dtype, device=DEVICE)
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
        cute_topk_sort_network=network,
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


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("network", ["compact", "compact_pruned"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,layout,key_dtype,rank_mode,value_mode",
    [
        (64, 32, 1, "replicated", "float32_bits", "ordinal", "decode"),
        (128, 32, 2, "replicated", "float32", "signed", "decode"),
        (128, 32, 16, "distributed", "int32", "ordinal", "decode"),
        (65, 33, 2, "replicated", "int32", "signed", "gather"),
        (512, 8, 2, "replicated", "float32_bits", "ordinal", "decode"),
    ],
)
def test_topk_compact_selection_networks(
    network: str,
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    layout: str,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
) -> None:
    x, storage = _layout_input(5, width, dtype, 2, 1)
    original = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_value_mode=value_mode,
        cute_topk_selection_layout=layout,
        cute_topk_sort_network=network,
    )
    assert repr(network) in code
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


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
        ("cute_topk_sort_network", True),
        ("cute_topk_sort_network", 1),
        ("cute_topk_sort_network", "unknown"),
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


@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
def test_topk_sort_network_config_roundtrip(
    topk_spec: ConfigSpec, network: str
) -> None:
    assert topk_spec.default_config()["cute_topk_sort_network"] == "batcher"
    config = helion.Config(cute_topk_sort_network=network)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config
    assert helion.Config.from_json(config.to_json()) == config


def test_topk_invalid_sort_network_normalizes_to_batcher(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_sort_network=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_sort_network"] == "batcher"


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


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("selection_layout", ["replicated", "distributed"])
@pytest.mark.parametrize("sort_network", ["batcher", "compact_pruned"])
@pytest.mark.parametrize(
    "width,k,lanes,output_vector,padding,offset,value_mode",
    [
        (64, 64, 4, 8, 0, 0, "decode"),
        (65, 65, 4, 1, 2, 1, "gather"),
        (128, 32, 16, 4, 0, 0, "decode"),
        (128, 8, 32, 4, 0, 0, "decode"),
    ],
)
def test_topk_native_float_preserves_all_bits(
    dtype: torch.dtype,
    largest: bool,
    selection_layout: str,
    sort_network: str,
    width: int,
    lanes: int,
    k: int,
    output_vector: int,
    padding: int,
    offset: int,
    value_mode: str,
) -> None:
    rows = (65536 + width - 1) // width + 2
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    data = words.view(dtype).repeat(2)[: (rows - 2) * width].reshape(rows - 2, width)
    x[2:].copy_(data)
    original = x.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype="float32_native",
        cute_topk_selection_layout=selection_layout,
        cute_topk_sort_network=sort_network,
    )
    assert "topk_native_key" in code
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


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
                (rows, cols), (stride, 1), dtype=dtype, device=torch.device("cpu")
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
            cute_topk_sort_network=sort_network,
            cute_topk_key_encoder=key_encoder,
            cute_topk_defer_value_gathers=defer_value_gathers,
            cute_topk_merge_schedule=merge_schedule,
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
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
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


@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_sort_network_reaches_codegen(network: str, layout: str) -> None:
    code = _code(
        9, 128, 128, 32, selection_layout=layout, lanes=2, sort_network=network
    )
    name = "local_topk" if layout == "replicated" else "distributed_topk"
    assert f"_cute_{name}_" in code
    assert f"(topk_keys, 32, 2, '{network}', 'sequential')" in code


def test_topk_cache_hash_includes_compact_tables() -> None:
    from helion._compiler.cute import topk_codegen

    original_getsource = topk_codegen.inspect.getsource

    def changed_tables(obj: Any) -> str:
        source = original_getsource(obj)
        if obj is topk_codegen.runtime_sorting_networks:
            source += "\n# test table change\n"
        return source

    before = _code(9, 128, 128, 32, sort_network="compact_pruned")
    with patch.object(topk_codegen.inspect, "getsource", side_effect=changed_tables):
        after = _code(9, 128, 128, 32, sort_network="compact_pruned")
    assert before != after
    assert "_cute_local_topk_" in before and "_cute_local_topk_" in after


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


@pytest.mark.parametrize(
    "dtype,width", [(torch.bfloat16, 16384), (torch.float16, 8192)]
)
def test_native_float_key_range_and_roundtrip(dtype: torch.dtype, width: int) -> None:
    words = torch.arange(65536, dtype=torch.int32)
    original = words.to(torch.int16).view(dtype)
    magnitude = words & 32767
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    bits = original.float().view(torch.int32)
    bits = (
        bits | 16384
        if dtype == torch.bfloat16
        else torch.where(magnitude == 0, bits | 32768, bits)
    )
    bits = torch.where(
        magnitude == infinity, 0x7F7F8000 | ((words & 32768) << 16), bits
    )
    bits = torch.where(magnitude > infinity, 0x7F7FC000, bits)
    indices = words % width
    encoded_index = torch.where(bits < 0, indices, width - 1 - indices)
    keys_bits = bits | encoded_index
    keys = keys_bits.view(torch.float32)
    assert bool(torch.isfinite(keys).all())
    assert not bool((keys == 0).any())
    recovered_indices = (keys_bits ^ ((keys_bits >> 31) ^ -1)) & (width - 1)
    assert torch.equal(recovered_indices, indices)
    clean = keys_bits & ~(width - 1)
    recovered_values = clean.view(torch.float32).to(dtype)
    finite = magnitude < infinity
    assert torch.equal(
        recovered_values[finite].view(torch.int16), original[finite].view(torch.int16)
    )
    order = keys.argsort()
    values = original.float()[order]
    nonnan = ~values.isnan()
    assert bool((values[nonnan][:-1] <= values[nonnan][1:]).all())
    assert bool(values[-int((~nonnan).sum()) :].isnan().all())


@pytest.mark.parametrize("index_bits", range(15))
@pytest.mark.parametrize("largest", [False, True])
def test_native_bfloat_bias_exhaustive(index_bits: int, largest: bool) -> None:
    # All payload widths cover every N in [1, 16384]. Checking both endpoints
    # bounds every intervening index because the payload is monotone and lies
    # strictly below the reserved bit. Also check a middle index explicitly.
    mask = (1 << index_bits) - 1
    words = torch.arange(65536, dtype=torch.int32)
    original = words.to(torch.int16).view(torch.bfloat16)
    magnitude = words & 32767
    finite = magnitude < 0x7F80
    bits = original.float().view(torch.int32) | 0x4000
    bits = torch.where(magnitude == 0x7F80, 0x7F7F8000 | ((words & 32768) << 16), bits)
    bits = torch.where(magnitude > 0x7F80, 0x7F7FC000, bits)
    indices = torch.tensor([0, mask // 2, mask], dtype=torch.int32)[:, None]
    payload = torch.where(bits[None, :] < 0, indices, mask - indices)
    key_bits = bits[None, :] | payload
    keys = key_bits.view(torch.float32)
    if not largest:
        keys = -keys
    assert bool(torch.isfinite(keys).all())
    assert not bool((keys == 0).any())

    # Recover indices and bits by the emitted decoder, including undoing the
    # smallest-first sign reversal before interpreting the native payload.
    output_bits = (keys if largest else -keys).view(torch.int32)
    recovered_indices = (output_bits ^ ((output_bits >> 31) ^ -1)) & mask
    assert torch.equal(recovered_indices, indices.expand_as(recovered_indices))
    clean = output_bits & ~mask
    decoded = clean.view(torch.float32).to(torch.bfloat16)
    assert torch.equal(
        decoded[:, finite].view(torch.int16),
        original[finite].view(torch.int16)[None, :].expand_as(decoded[:, finite]),
    )
    # The existing guard must gather exactly infinities and NaNs, preserving
    # their original sign/payload. Quarter-ULP finite keys must stay below it.
    decodable = (output_bits & 0x7FFF8000) < 0x7F7F8000
    assert torch.equal(decodable, finite[None, :].expand_as(decodable))
    assert bool(((output_bits[:, finite] & 0x7FFFFFFF) <= 0x7F7F7FFF).all())
    gathered = original.view(torch.int16)[None, :].expand_as(output_bits)
    selected = torch.where(decodable, decoded.view(torch.int16), gathered)
    assert torch.equal(selected, gathered)

    # An integer ordinal oracle independently orders original BF16 words;
    # signed zeros can be ordered either way and NaNs share the largest rank.
    signed = words.to(torch.int16).to(torch.int32)
    ordinal = signed ^ ((signed >> 31) & 32767)
    ordinal = torch.where(magnitude > 0x7F80, 32767, ordinal)
    if not largest:
        ordinal = -ordinal
    order = ordinal.argsort()
    distinct = ordinal[order][1:] != ordinal[order][:-1]
    minimum = keys.amin(dim=0)[order]
    maximum = keys.amax(dim=0)[order]
    # Even opposite index endpoints cannot exchange differently ranked values.
    assert bool((maximum[:-1][distinct] < minimum[1:][distinct]).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
def test_native_float_bias_codegen(dtype: torch.dtype, largest: bool) -> None:
    code = _code(
        3,
        128,
        128,
        32,
        4,
        dtype=dtype,
        largest=largest,
        key_dtype="float32_native",
        value_mode="decode",
    )
    input_type = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    assert f"cutlass.Uint16(topk_bits).bitcast({input_type})" in code
    assert ("if topk_magnitude == 0:" in code) == (dtype == torch.float16)
    expected_bias = 16384 if dtype == torch.bfloat16 else 32768
    assert f"topk_native_bits | cutlass.Int32({expected_bias})" in code
    assert ("topk_native_key = -topk_native_key" in code) == (not largest)


@pytest.mark.parametrize(
    "dtype,width,native",
    [
        (torch.bfloat16, 16384, True),
        (torch.bfloat16, 16385, False),
        (torch.float16, 8192, True),
        (torch.float16, 8193, False),
    ],
)
def test_native_float_width_guard(dtype: torch.dtype, width: int, native: bool) -> None:
    code = _code(
        3,
        width,
        width,
        32,
        4,
        dtype=dtype,
        key_dtype="float32_native",
        value_mode="decode",
    )
    assert ("topk_native_key" in code) == native


# Networks top-k tests.


def _evaluate(values: np.ndarray, size: int, k: int, network: str) -> np.ndarray:
    actual = values.copy()
    program = (
        _pruned_sort_network(size, k)
        if network == "compact_pruned"
        else tuple(
            (left, right, True, True) for left, right in _sort_network(size, network)
        )
    )
    for left, right, keep_left, keep_right in program:
        a, b = actual[:, left].copy(), actual[:, right].copy()
        if keep_left:
            actual[:, left] = np.maximum(a, b)
        if keep_right:
            actual[:, right] = np.minimum(a, b)
    return actual[:, :k]


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
def test_selection_network_exhaustive_binary(size: int, network: str) -> None:
    values = (
        (np.arange(1 << size, dtype=np.uint32)[:, None] >> np.arange(size)) & 1
    ).astype(np.uint8)
    expected = np.sort(values, axis=1)[:, ::-1]
    # Zero-one verification proves data-independent comparator networks for
    # every ordered input domain. Check every supported prefix length too.
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(
            _evaluate(values, size, k, network), expected[:, :k]
        )


@pytest.mark.parametrize("size", [32, 64, 128, 256])
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
@pytest.mark.parametrize("floating", [False, True])
def test_selection_network_random_duplicates_and_padding(
    size: int, network: str, floating: bool
) -> None:
    generator = np.random.default_rng(20260926)
    values = generator.integers(-(1 << 22), 1 << 22, size=(128, size), dtype=np.int32)
    values[0] = 0
    values[1] = np.iinfo(np.int32).min
    values[2] = np.arange(size) % 4
    for row in range(3, len(values)):
        values[row, row % size :] = np.iinfo(np.int32).min
        generator.shuffle(values[row])
    if floating:
        values = values.astype(np.float32)
    expected = np.sort(values, axis=1)[:, ::-1]
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(
            _evaluate(values, size, k, network), expected[:, :k]
        )


def test_compact_network_bounds_and_operation_count() -> None:
    for size, layers in COMPACT_SORT_LAYERS.items():
        for layer in layers:
            wires = [wire for pair in layer for wire in pair]
            assert len(wires) == len(set(wires))
            assert all(0 <= left < right < size for left, right in layer)
    assert len(_sort_network(32, "compact")) == 185
    assert len(_sort_network(64, "compact")) == 521
    # For the complete local 64->32 operation: two sorts plus a bitonic
    # top-k merge versus the pruned whole-fragment comparator program.
    merge_operations = 32 * 6
    assert 4 * len(_sort_network(32, "batcher")) + merge_operations == 956
    assert 4 * len(_sort_network(32, "compact")) + merge_operations == 932
    assert (
        sum(left + right for _, _, left, right in _pruned_sort_network(64, 32)) == 870
    )


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_network_dispatch_and_power_two_fallback(size: int) -> None:
    assert not _use_pruned_sort_network(size, "batcher")
    assert not _use_pruned_sort_network(size, "compact")
    assert _use_pruned_sort_network(size, "compact_pruned") == (
        size in COMPACT_SORT_LAYERS
    )
    if size not in COMPACT_SORT_LAYERS:
        assert _sort_network(size, "compact") == _odd_even_sort_network(size)
        assert _sort_network(size, "compact_pruned") == _odd_even_sort_network(size)


@pytest.mark.parametrize("size,k", [(0, 1), (3, 1), (8, 0), (8, 3), (8, 16)])
def test_pruned_network_rejects_invalid_sizes(size: int, k: int) -> None:
    with pytest.raises(AssertionError):
        _pruned_sort_network(size, k)


# Ordered asm top-k tests.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("index_bits", range(16))
def test_ordered_key_asm_exhaustive(
    dtype: torch.dtype, rank_mode: str, largest: bool, index_bits: int
) -> None:
    words = torch.arange(65536, dtype=torch.int32)
    signed = words.to(torch.int16).to(torch.int32)
    magnitude = signed & 32767
    sign = signed >> 31
    mask = (1 << index_bits) - 1
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    # PTX arithmetic with signed32 wrapping and the same narrow input.
    rank = (
        signed ^ (sign & 32767) if rank_mode == "ordinal" else (magnitude ^ sign) - sign
    )
    rank = torch.where(magnitude > infinity, 32767, rank)
    if not largest:
        rank = -rank
    columns = torch.tensor([0, mask // 2, mask], dtype=torch.int32)[:, None]
    packed = (rank[None, :] << index_bits) | (mask - columns)

    # Independent sign/magnitude oracle: ordinal negative values include -1.
    expected_rank = torch.where(
        words < 32768, magnitude, -magnitude - int(rank_mode == "ordinal")
    )
    expected_rank = torch.where(magnitude > infinity, 32767, expected_rank)
    if not largest:
        expected_rank = -expected_rank
    expected = (expected_rank[None, :] << index_bits) | (mask - columns)
    assert torch.equal(packed, expected)
    assert torch.equal(packed >> index_bits, rank[None, :].expand_as(packed))
    assert torch.equal(mask - (packed & mask), columns.expand_as(packed))
    assert bool((packed > torch.iinfo(torch.int32).min).all())
    assert int(rank.min()) >= -32767 and int(rank.max()) <= 32767
    # The existing Float32 conversions remain exact inside their guards.
    if index_bits <= 9:
        assert torch.equal(packed.float().to(torch.int32), packed)
    if index_bits <= 14:
        biased = packed + 0x40000000
        assert bool(torch.isfinite(biased.view(torch.float32)).all())
        assert bool((biased.view(torch.float32) > 0).all())
        assert torch.equal(
            biased.view(torch.float32).view(torch.int32) - 0x40000000, packed
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "width,key_dtype,ordered",
    [
        (128, "int32", True),
        (512, "float32", True),
        (1024, "float32", True),
        (16384, "float32_bits", True),
        (32768, "float32_bits", True),
        (128, "float32_native", False),
        (32768, "float32_native", True),
    ],
)
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
def test_ordered_key_asm_codegen_scope(
    dtype: torch.dtype, width: int, key_dtype: str, ordered: bool, key_encoder: str
) -> None:
    code = _code(
        3,
        width,
        width + 1,
        8,
        dtype=dtype,
        key_dtype=key_dtype,
        key_encoder=key_encoder,
    )
    ordered = ordered and key_encoder == "asm"
    assert ("_cute_encode_ordered_topk_" in code) == ordered
    if ordered:
        assert "topk_bits = topk_input_bits[topk_row, topk_col]" in code
        tree = ast.parse(code)
        assert not any(
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "topk_sign"
            for node in ast.walk(tree)
        )


# Packed rare top-k tests.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_bits", range(16))
@pytest.mark.parametrize("largest", [False, True])
def test_packed_rare_exhaustive(
    dtype: torch.dtype, index_bits: int, largest: bool
) -> None:
    words = ((torch.arange(65536, dtype=torch.int32) * 32771) & 65535).reshape(-1, 16)
    magnitude = words & 32767
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    original_rank = torch.where(words < 32768, magnitude, -1 - magnitude)
    original_rank = torch.where(magnitude > infinity, 32767, original_rank)
    mask = (1 << index_bits) - 1
    for column in (0, mask // 2, mask):
        ranked = original_rank if largest else -original_rank
        packed = (ranked << index_bits) | (mask - column)
        for key_dtype in ("int32", "float32", "float32_bits"):
            if key_dtype == "float32" and index_bits <= 9:
                selected = packed.float().to(torch.int32)
            elif key_dtype == "float32_bits" and index_bits <= 14:
                selected = (packed + 0x40000000).view(torch.float32).view(
                    torch.int32
                ) - 0x40000000
            else:
                selected = packed
            rank = selected >> index_bits
            if not largest:
                rank = -rank
            assert torch.equal(rank, original_rank)
            branch = rank.amax(dim=-1, keepdim=True) == 32767
            exceptional = magnitude > infinity
            assert torch.equal(branch, exceptional.any(dim=-1, keepdim=True))
            ordered_rank = selected.sort(dim=-1, descending=True).values >> index_bits
            if not largest:
                ordered_rank = -ordered_rank
            endpoint = ordered_rank[:, :1] if largest else ordered_rank[:, -1:]
            assert torch.equal(endpoint, rank.amax(dim=-1, keepdim=True))
            assert torch.equal(endpoint == 32767, branch)
            direct = (rank ^ ((rank >> 31) & 32767)).to(torch.int16)
            repaired = torch.where(
                branch & (rank == 32767), words.to(torch.int16), direct
            )
            assert torch.equal(repaired, words.to(torch.int16))
            assert bool((mask - (selected & mask) == column).all())


@pytest.mark.parametrize(
    "key,default,valid,invalid",
    [
        ("cute_topk_key_encoder", "dsl", "asm", True),
        ("cute_topk_key_encoder", "dsl", "dsl", "unknown"),
        ("cute_topk_defer_value_gathers", False, True, 1),
        ("cute_topk_defer_value_gathers", False, False, "true"),
    ],
)
def test_packed_rare_config(
    key: str,
    default: object,
    valid: object,
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1)
    with pytest.raises(exc.InvalidConfig, match="compatible top-k root"):
        spec.normalize({key: valid})
    spec.enable_cute_topk_search()
    assert spec.default_config()[key] == default
    config = helion.Config()
    config.config[key] = valid
    spec.normalize(config)
    generation = ConfigGeneration(spec)
    assert generation.unflatten(generation.flatten(config)) == config
    invalid_config = helion.Config()
    invalid_config.config[key] = invalid
    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        spec.normalize(invalid_config)
    spec.normalize(invalid_config, _fix_invalid=True)
    assert invalid_config[key] == default


@pytest.mark.parametrize(
    "width,k,lanes,vector,layout,key_dtype,rank_mode,value_mode,active",
    [
        (128, 32, 2, 4, "replicated", "int32", "ordinal", "decode", True),
        (128, 24, 2, 4, "replicated", "float32_bits", "ordinal", "decode", True),
        (128, 32, 4, 4, "distributed", "float32", "ordinal", "decode", True),
        (128, 12, 2, 4, "replicated", "int32", "ordinal", "decode", False),
        (64, 6, 2, 2, "distributed", "int32", "ordinal", "decode", False),
        (128, 32, 2, 1, "replicated", "int32", "ordinal", "decode", False),
        (128, 32, 2, 4, "replicated", "float32_native", "ordinal", "decode", False),
        (128, 32, 2, 4, "replicated", "int32", "ordinal", "gather", False),
        (128, 32, 2, 4, "replicated", "int32", "signed", "decode", False),
        (32768, 32, 2, 4, "replicated", "float32_native", "ordinal", "decode", True),
    ],
)
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
@pytest.mark.parametrize("largest", [False, True])
def test_packed_rare_codegen_guard(
    width: int,
    k: int,
    lanes: int,
    vector: int,
    layout: str,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
    active: bool,
    key_encoder: str,
    largest: bool,
) -> None:
    code = _code(
        5,
        width,
        width + 2,
        k,
        vector,
        largest=largest,
        lanes=lanes,
        key_dtype=key_dtype,
        rank_mode=rank_mode,
        value_mode=value_mode,
        selection_layout=layout,
        key_encoder=key_encoder,
        defer_value_gathers=True,
    )
    assert ("topk_output_max_rank" in code) == active
    if not active:
        return
    branch = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "topk_output_max_rank == cutlass.Int32(32767)"
    )
    text = ast.unparse(branch)
    assert (
        "transpose" not in text and "shuffle" not in text and "autovec_copy" not in text
    )
    assert "x[topk_row, topk_selected_index]" in text
    assert ("topk_value_rank = -topk_value_rank" in text) == (not largest)
    assert code.index("topk_selected =") < code.index("if topk_output_max_rank")
    if layout == "distributed":
        assert code.index("topk_output_keys = _cute_transpose") < code.index(
            "if topk_valid_row:"
        )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_packed_rare_wide_addresses(index_dtype: torch.dtype) -> None:
    code = _code(
        1,
        128,
        2**35,
        32,
        4,
        lanes=2,
        index_dtype=index_dtype,
        rank_mode="ordinal",
        value_mode="decode",
        defer_value_gathers=True,
    )
    assert "topk_output_max_rank" in code
    assert "topk_row = cutlass.Int64(" in code
    assert "topk_output_offset = topk_row * cutlass.Int64(32)" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
@pytest.mark.parametrize(
    "width,k,lanes,vector,layout,key_dtype,offset",
    [
        (128, 32, 2, 4, "replicated", "int32", 0),
        (128, 128, 4, 8, "distributed", "float32_bits", 0),
        (65, 8, 2, 2, "distributed", "float32", 0),
        (64, 6, 2, 2, "distributed", "int32", 1),
    ],
)
def test_packed_rare_exact_bits(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    key_encoder: str,
    width: int,
    k: int,
    lanes: int,
    vector: int,
    layout: str,
    key_dtype: str,
    offset: int,
) -> None:
    rows = 514 if k == 128 else 5
    x, storage = _layout_input(rows, width, dtype, 2 if offset else 0, offset)
    if k == 128:
        x[2:].copy_(
            torch.arange(65536, dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
            .reshape(-1, width)
        )
    else:
        x[2].fill_(1)
        x[3].fill_(float("inf"))
        x[3, 1::2] = -float("inf")
        nan_words = [0x7FC1, -46] if dtype == torch.bfloat16 else [0x7E01, -478]
        x[4].copy_(
            torch.tensor(nan_words, dtype=torch.int16)
            .view(dtype)
            .repeat((width + 1) // 2)[:width]
        )
    original = storage.clone()
    value_storage = torch.full((rows * k + offset + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full(
        (rows * k + offset + 2,), -7, dtype=index_dtype, device=DEVICE
    )
    values = value_storage[offset : offset + rows * k].view(rows, k)
    indices = index_storage[offset : offset + rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=vector,
        cute_topk_selection_layout=layout,
        cute_topk_value_mode="decode",
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode="ordinal",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder=key_encoder,
        cute_topk_defer_value_gathers=True,
    )
    assert ("topk_output_max_rank" in code) == (k != 6)
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((value_storage[:offset] == 7).all())
    assert bool((value_storage[offset + rows * k :] == 7).all())
    assert bool((index_storage[:offset] == -7).all())
    assert bool((index_storage[offset + rows * k :] == -7).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
def test_packed_rare_alignment_transition(
    index_dtype: torch.dtype, layout: str, key_encoder: str
) -> None:
    rows, width, k = 5, 128, 32
    x, storage = _layout_input(rows, width, torch.bfloat16, 0, 0)
    original = storage.clone()
    value_storage = torch.empty(rows * k + 2, dtype=torch.bfloat16, device=DEVICE)
    index_storage = torch.empty(rows * k + 2, dtype=index_dtype, device=DEVICE)
    values = value_storage[: rows * k].view(rows, k)
    indices = index_storage[: rows * k].view(rows, k)
    bound = _extra_out_topk._bind_isolated((x, values, indices, k, True))
    bound.set_config(
        helion.Config(
            block_sizes=[1],
            cute_topk_lanes_per_row=4,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=4,
            cute_topk_selection_layout=layout,
            cute_topk_key_dtype="int32",
            cute_topk_rank_mode="ordinal",
            cute_topk_key_encoder=key_encoder,
            cute_topk_value_mode="decode",
            cute_topk_defer_value_gathers=True,
        )
    )
    for value_offset, index_offset in ((0, 0), (0, 1), (1, 0), (1, 1), (0, 0)):
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
        assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


# Endpoints top-k tests.


@pytest.mark.parametrize(
    "k,lanes,requested,layout,last",
    [
        (24, 2, 4, "replicated", 11),
        (32, 4, 4, "distributed", 7),
        (8, 2, 8, "distributed", 3),
    ],
)
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
@pytest.mark.parametrize("encoder", ["dsl", "asm"])
def test_packed_endpoint_codegen(
    k: int,
    lanes: int,
    requested: int,
    layout: str,
    last: int,
    largest: bool,
    key_dtype: str,
    encoder: str,
) -> None:
    code = _code(
        5,
        128,
        128,
        k,
        requested,
        lanes=lanes,
        largest=largest,
        key_dtype=key_dtype,
        key_encoder=encoder,
        rank_mode="ordinal",
        value_mode="decode",
        selection_layout=layout,
        defer_value_gathers=True,
    )
    assignments = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "topk_output_max_rank"
    ]
    assert len(assignments) == 1
    expression = assignments[0].value
    reads = [
        node
        for node in ast.walk(expression)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "topk_output_keys"
    ]
    assert len(reads) == 1
    assert ast.literal_eval(reads[0].slice) == (0 if largest else last)
    assert isinstance(expression, ast.UnaryOp) == (not largest)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
        for node in ast.walk(expression)
    )
    assert "max(topk_output_max_rank" not in code
    if layout == "distributed" and key_dtype == "float32_bits":
        assert ".bitcast(cutlass.Int32) - cutlass.Int32(1073741824)" in ast.unparse(
            expression
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("encoder", ["dsl", "asm"])
@pytest.mark.parametrize("offset", [0, 1])
def test_packed_endpoint_non_power_two_k(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    encoder: str,
    offset: int,
) -> None:
    rows, width, k = 5, 128, 24
    x, storage = _layout_input(rows, width, dtype, 2 if offset else 0, offset)
    nan_word = -46 if dtype == torch.bfloat16 else -478
    nan = torch.tensor([nan_word], dtype=torch.int16, device=DEVICE).view(dtype)[0]
    # Only one output-owning lane needs a NaN repair. For smallest top-k,
    # exactly k-1 finite inputs put the selected NaN at the last valid rank.
    if largest:
        x[2].copy_(torch.arange(width, dtype=dtype, device=DEVICE))
        x[2, -1] = nan
    else:
        x[2].copy_(nan.expand(width))
        x[2, : k - 1].copy_(torch.arange(k - 1, dtype=dtype, device=DEVICE))
    x[3].fill_(float("inf"))
    x[3, 1::2] = -float("inf")
    original = storage.clone()
    values_storage = torch.full((rows * k + offset + 2,), 7, dtype=dtype, device=DEVICE)
    indices_storage = torch.full(
        (rows * k + offset + 2,), -7, dtype=index_dtype, device=DEVICE
    )
    values = values_storage[offset : offset + rows * k].view(rows, k)
    indices = indices_storage[offset : offset + rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout="replicated",
        cute_topk_key_encoder=encoder,
        cute_topk_defer_value_gathers=True,
    )
    assert "topk_output_keys[0]" in code if largest else "topk_output_keys[11]" in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((values_storage[:offset] == 7).all())
    assert bool((values_storage[offset + rows * k :] == 7).all())
    assert bool((indices_storage[:offset] == -7).all())
    assert bool((indices_storage[offset + rows * k :] == -7).all())


# Alias cache top-k tests.


@pytest.fixture
def cpu_codegen() -> Iterator[None]:
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
        yield


def _new_kernel() -> helion.Kernel[Any]:
    return helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")(
        _extra_out_topk.fn
    )


def _config(encoder: str) -> helion.Config:
    return helion.Config(
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_value_mode="decode",
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_selection_layout="replicated",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder=encoder,
        cute_topk_defer_value_gathers=True,
    )


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("retain_input", [False, True])
def test_cached_topk_recompiles_after_construction_tensors_expire(
    retain_input: bool,
) -> None:
    kernel = _new_kernel()

    def first_binding() -> tuple[Any, torch.Tensor | None]:
        x = torch.empty((5, 128), dtype=torch.bfloat16)
        values = torch.empty((5, 32), dtype=x.dtype)
        indices = torch.empty((5, 32), dtype=torch.int32)
        bound = kernel.bind((x, values, indices, 32, True))
        assert "topk_output_max_rank" in bound.to_triton_code(_config("dsl"))
        return bound, x if retain_input else None

    bound, retained = first_binding()
    gc.collect()
    assert sum(
        ref() is not None for ref in bound._runtime_tensor_refs_by_name.values()
    ) == int(retain_input)
    assert bound.env.runtime_arg_values_by_name == {}
    x = torch.empty((5, 128), dtype=torch.bfloat16)
    values = torch.empty((5, 32), dtype=x.dtype)
    indices = torch.empty((5, 32), dtype=torch.int32)
    assert kernel.bind((x, values, indices, 32, True)) is bound
    code = bound.to_triton_code(_config("asm"))
    assert "topk_output_max_rank" in code
    assert "_cute_encode_ordered_topk" in code
    assert (retained is not None) == retain_input


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("alias_kind", ["view", "dlpack"])
@pytest.mark.parametrize("width", [1, 32])
def test_cached_topk_alias_binding_stays_rejected_after_release(
    alias_kind: str, width: int
) -> None:
    kernel = _new_kernel()

    def bind_pair() -> tuple[Any, Any]:
        x = torch.empty((5, width), dtype=torch.bfloat16)
        values = torch.empty_like(x)
        indices = torch.empty((5, width), dtype=torch.int32)
        disjoint = kernel.bind((x, values, indices, width, True))
        alias = x.view_as(x) if alias_kind == "view" else torch.from_dlpack(x)
        overlapping = kernel.bind((x, alias, indices, width, True))
        assert overlapping is not disjoint
        return disjoint, overlapping

    disjoint, overlapping = bind_pair()
    gc.collect()
    for bound, expected in ((disjoint, True), (overlapping, False)):
        assert all(ref() is None for ref in bound._runtime_tensor_refs_by_name.values())
        assert bound.config_spec.cute_topk_search_enabled == expected
        assert (
            "cute_topk_lanes_per_row" in bound.config_spec.default_config()
        ) == expected
        code = bound.to_triton_code(bound.config_spec.default_config())
        assert ("_cute_local_topk" in code) == expected


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
def test_singleton_topk_search_and_recompile_after_input_release(
    dtype: torch.dtype, index_dtype: torch.dtype, largest: bool
) -> None:
    kernel = _new_kernel()
    config = helion.Config(
        block_sizes=[1],
        cute_topk_lanes_per_row=1,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=1,
    )

    def first_binding() -> Any:
        x = torch.empty((5, 1), dtype=dtype)
        values = torch.empty_like(x)
        indices = torch.empty((5, 1), dtype=index_dtype)
        bound = kernel.bind((x, values, indices, 1, largest))
        assert bound.config_spec.cute_topk_search_enabled
        assert "_cute_local_topk" in bound.to_triton_code(config)
        return bound

    bound = first_binding()
    gc.collect()
    assert all(ref() is None for ref in bound._runtime_tensor_refs_by_name.values())
    assert bound.env.runtime_arg_values_by_name == {}
    x = torch.empty((5, 1), dtype=dtype)
    values = torch.empty_like(x)
    indices = torch.empty((5, 1), dtype=index_dtype)
    assert kernel.bind((x, values, indices, 1, largest)) is bound
    config.config["cute_topk_vector_width"] = 4
    assert "_cute_local_topk" in bound.to_triton_code(config)


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "missing",
        "identity",
        "sources",
        "properties",
        "length",
        "nonbool",
        "live_alias",
        "partial_live_alias",
    ],
)
def test_cached_topk_alias_fact_requires_matching_descriptor(damage: str) -> None:
    x = torch.empty((5, 32), dtype=torch.bfloat16)
    values = torch.empty_like(x)
    indices = torch.empty((5, 32), dtype=torch.int32)
    bound = _new_kernel().bind((x, values, indices, 32, True))
    env = bound.env
    key = _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    descriptor = env.runtime_input_specializations[key]
    if damage == "missing":
        env.bound_runtime_input_specialization_results.clear()
    elif damage in ("identity", "sources", "properties"):
        changes: dict[str, Any] = {
            "identity": {"classifier_identity": "other"},
            "sources": {"sources": descriptor.sources[::-1]},
            "properties": {"reusable_tensor_properties": frozenset()},
        }[damage]
        env.runtime_input_specializations[key] = dataclasses.replace(
            descriptor, **changes
        )
    elif damage == "length":
        env.bound_runtime_input_specialization_results[key] = (True,)
    elif damage == "nonbool":
        env.bound_runtime_input_specialization_results[key] = (1, True, True)
    live_args: dict[str, object] = {}
    if damage in ("live_alias", "partial_live_alias"):
        live_args = {"x": x, "values": torch.from_dlpack(x)}
        if damage == "live_alias":
            live_args["indices"] = indices
    host = bound.host_function
    assert host is not None
    with env, host, env.use_runtime_arg_values(live_args):
        candidate = match_topk_root(
            host.device_ir.graphs,
            noncanonical_block_ids=host.device_ir.noncanonical_task_origin_block_ids,
        )
        assert candidate is not None
        assert topk_tensors_are_proven_disjoint(candidate, env) == (damage == "none")


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "damage",
    ["none", "alias", "missing_argument", "identity", "bound_false", "bound_none"],
)
def test_search_alias_proof_requires_registered_live_facts(damage: str) -> None:
    x = torch.empty((5, 32), dtype=torch.bfloat16)
    values = torch.empty_like(x)
    indices = torch.empty((5, 32), dtype=torch.int32)
    bound = _new_kernel().bind((x, values, indices, 32, True))
    env = bound.env
    key = _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    env.bound_runtime_input_specialization_results.clear()
    runtime_args: dict[str, object] = {"x": x, "values": values, "indices": indices}
    if damage == "alias":
        runtime_args["values"] = torch.from_dlpack(x)
    elif damage == "missing_argument":
        runtime_args.pop("values")
    elif damage == "identity":
        env.runtime_input_specializations[key] = dataclasses.replace(
            env.runtime_input_specializations[key], classifier_identity="other"
        )
    elif damage == "bound_false":
        env.bound_runtime_input_specialization_results[key] = (False, False, False)
    elif damage == "bound_none":
        env.bound_runtime_input_specialization_results[key] = None
    host = bound.host_function
    assert host is not None
    with env, host, env.use_runtime_arg_values(runtime_args):
        candidate = match_topk_root(
            host.device_ir.graphs,
            noncanonical_block_ids=host.device_ir.noncanonical_task_origin_block_ids,
        )
        assert candidate is not None
        assert not topk_tensors_are_proven_disjoint(candidate, env)
        assert topk_tensors_are_proven_disjoint(candidate, env, allow_unbound=True) == (
            damage == "none"
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
def test_cached_topk_recompiles_exact_values_after_input_release(
    dtype: torch.dtype, index_dtype: torch.dtype, largest: bool
) -> None:
    kernel = _new_kernel()

    def run(encoder: str) -> Any:
        x, storage = _layout_input(5, 128, dtype, 0, 0)
        values = torch.empty((5, 32), dtype=dtype, device=DEVICE)
        indices = torch.empty((5, 32), dtype=index_dtype, device=DEVICE)
        args = (x, values, indices, 32, largest)
        bound = kernel.bind(args)
        code, _ = code_and_output(kernel, args, **_config(encoder).config)
        assert "topk_output_max_rank" in code
        _assert_topk_output(x, values, indices, 32, largest, index_dtype=index_dtype)
        assert torch.equal(x.view(torch.int16), storage.view(5, 128).view(torch.int16))
        return bound

    first = run("dsl")
    gc.collect()
    assert all(ref() is None for ref in first._runtime_tensor_refs_by_name.values())
    assert run("asm") is first


# Balanced top-k tests.


def _merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.maximum(left, right[:, ::-1])
    distance = result.shape[1] // 2
    while distance:
        for begin in range(0, result.shape[1], 2 * distance):
            for offset in range(distance):
                low, high = begin + offset, begin + offset + distance
                a, b = result[:, low].copy(), result[:, high].copy()
                result[:, low], result[:, high] = np.maximum(a, b), np.minimum(a, b)
        distance //= 2
    return result


def _select(values: np.ndarray, k: int, network: str) -> np.ndarray:
    partials: dict[int, np.ndarray] = {}
    for left, right in _balanced_chunk_program(values.shape[1] // k):
        if left == right:
            partials[left] = _evaluate(
                values[:, left * k : (left + 1) * k], k, k, network
            )
        else:
            partials[left] = _merge(partials[left], partials.pop(right))
    assert len(partials) == 1
    return partials[0]


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("network", ["batcher", "compact"])
def test_balanced_exhaustive_binary(size: int, network: str) -> None:
    values = (
        (np.arange(1 << size, dtype=np.uint32)[:, None] >> np.arange(size)) & 1
    ).astype(np.uint8)
    expected = np.sort(values, axis=1)[:, ::-1]
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(_select(values, k, network), expected[:, :k])


@pytest.mark.parametrize("size", [32, 64, 128, 256])
@pytest.mark.parametrize("network", ["batcher", "compact"])
@pytest.mark.parametrize("floating", [False, True])
def test_balanced_random_duplicates_padding(
    size: int, network: str, floating: bool
) -> None:
    generator = np.random.default_rng(20260926)
    values = generator.integers(-64, 64, size=(128, size), dtype=np.int32)
    values[0] = 0
    for row in range(1, len(values)):
        values[row, row % size :] = np.iinfo(np.int32).min
        generator.shuffle(values[row])
    if floating:
        values = values.astype(np.float32)
        values[values == np.float32(np.iinfo(np.int32).min)] = -np.inf
    expected = np.sort(values, axis=1)[:, ::-1]
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(_select(values, k, network), expected[:, :k])


@pytest.mark.parametrize("chunks", [1, 2, 4, 8, 16, 32])
def test_balanced_schedule_equal_adjacent_groups(chunks: int) -> None:
    groups: dict[int, int] = {}
    merges = 0
    for left, right in _balanced_chunk_program(chunks):
        if left == right:
            assert left not in groups
            groups[left] = 1
        else:
            assert groups[left] == groups[right] == right - left
            groups[left] += groups.pop(right)
            merges += 1
    assert groups == {0: chunks}
    assert merges == chunks - 1


def _operation_count_and_depth(
    size: int, k: int, network: str, balanced: bool
) -> tuple[int, int]:
    depths: dict[int, list[int]] = {}
    operations = 0
    chunks = size // k
    program = (
        _balanced_chunk_program(chunks)
        if balanced
        else tuple(
            item
            for chunk in range(chunks)
            for item in (
                ((chunk, chunk),) if chunk == 0 else ((chunk, chunk), (0, chunk))
            )
        )
    )
    for left, right in program:
        if left == right:
            depth = [0] * k
            for a, b in _sort_network(k, network):
                depth[a] = depth[b] = max(depth[a], depth[b]) + 1
                operations += 2
            depths[left] = depth
        else:
            depth = [
                max(a, b) + 1
                for a, b in zip(depths[left], reversed(depths.pop(right)), strict=True)
            ]
            operations += k
            distance = k // 2
            while distance:
                for begin in range(0, k, distance * 2):
                    for offset in range(distance):
                        a, b = begin + offset, begin + offset + distance
                        depth[a] = depth[b] = max(depth[a], depth[b]) + 1
                        operations += 2
                distance //= 2
            depths[left] = depth
    return operations, max(depths[0])


@pytest.mark.parametrize(
    "size,k,operations,sequential,balanced",
    [
        (32, 8, 248, 18, 14),
        (64, 8, 528, 34, 18),
        (128, 8, 1088, 66, 22),
        (32, 32, 382, 15, 15),
        (128, 32, 2104, 33, 27),
    ],
)
def test_balanced_same_operations_shorter_depth(
    size: int, k: int, operations: int, sequential: int, balanced: int
) -> None:
    assert _operation_count_and_depth(size, k, "batcher", False) == (
        operations,
        sequential,
    )
    assert _operation_count_and_depth(size, k, "batcher", True) == (
        operations,
        balanced,
    )


def test_balanced_config_roundtrip_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1)
    with pytest.raises(exc.InvalidConfig):
        spec.normalize(helion.Config(cute_topk_merge_schedule="balanced"))
    spec.enable_cute_topk_search()
    assert spec.default_config()["cute_topk_merge_schedule"] == "sequential"
    config = helion.Config(cute_topk_merge_schedule="balanced")
    spec.normalize(config)
    generation = ConfigGeneration(spec)
    assert generation.unflatten(generation.flatten(config)) == config
    for invalid in (True, 1, "tree"):
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(helion.Config(cute_topk_merge_schedule=invalid))


@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
def test_balanced_schedule_reaches_codegen(layout: str, network: str) -> None:
    code = _code(
        5,
        128,
        130,
        8,
        selection_layout=layout,
        lanes=4,
        sort_network=network,
        merge_schedule="balanced",
    )
    assert f"8, 4, '{network}', 'balanced')" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,network,key_dtype,layout,rank_mode,value_mode,padding,offset",
    [
        (128, 8, 4, "batcher", "float32_bits", "distributed", "signed", "gather", 0, 0),
        (1024, 8, 8, "batcher", "float32", "distributed", "ordinal", "decode", 2, 1),
        (
            65,
            6,
            2,
            "compact",
            "float32_native",
            "replicated",
            "ordinal",
            "decode",
            2,
            1,
        ),
        (
            128,
            32,
            4,
            "compact_pruned",
            "int32",
            "distributed",
            "signed",
            "gather",
            0,
            0,
        ),
        (33, 8, 16, "batcher", "int32", "distributed", "ordinal", "decode", 0, 0),
    ],
)
def test_balanced_gpu_exact_selected_bits(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    network: str,
    key_dtype: str,
    layout: str,
    rank_mode: str,
    value_mode: str,
    padding: int,
    offset: int,
) -> None:
    rows = 5
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original_storage = storage.clone()
    values_storage = torch.full((rows * k + 2,), 7, dtype=dtype, device=DEVICE)
    indices_storage = torch.full((rows * k + 2,), -7, dtype=index_dtype, device=DEVICE)
    values = values_storage[offset : offset + rows * k].view(rows, k)
    indices = indices_storage[offset : offset + rows * k].view(rows, k)
    code, _output = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_selection_layout=layout,
        cute_topk_sort_network=network,
        cute_topk_merge_schedule="balanced",
    )
    assert "'balanced')" in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))
    assert bool((values_storage[offset + rows * k :] == 7).all())
    assert bool((indices_storage[offset + rows * k :] == -7).all())
    if offset:
        assert bool((values_storage[:offset] == 7).all())
        assert bool((indices_storage[:offset] == -7).all())


@pytest.mark.parametrize("encoder", ["dsl", "asm"])
@pytest.mark.parametrize("largest", [False, True])
def test_balanced_endpoint_codegen_composition(encoder: str, largest: bool) -> None:
    # P=128 exceeds the compact table catalog, so compact_pruned selects its
    # chunked fallback and the balanced schedule is active. K/(L*V)=1 makes
    # every output group complete, activating the ordinal endpoint guard.
    code = _code(
        5,
        256,
        256,
        8,
        4,
        lanes=2,
        largest=largest,
        value_mode="decode",
        rank_mode="ordinal",
        key_encoder=encoder,
        sort_network="compact_pruned",
        merge_schedule="balanced",
        defer_value_gathers=True,
    )
    assert "'compact_pruned', 'balanced')" in code
    assert "topk_output_max_rank" in code
    assert (
        "_cute_encode_ordered_topk" in code
        if encoder == "asm"
        else "topk_magnitude" in code
    )
    assert "topk_output_keys[0]" in code if largest else "topk_output_keys[3]" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
def test_balanced_asm_endpoint_exact_selected_bits(
    dtype: torch.dtype, largest: bool
) -> None:
    rows, width, k = 5, 256, 8
    x, storage = _layout_input(rows, width, dtype, 0, 0)
    original = storage.clone()
    value_storage = torch.full((rows * k + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full((rows * k + 2,), -7, dtype=torch.int32, device=DEVICE)
    values = value_storage[: rows * k].view(rows, k)
    indices = index_storage[: rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout="replicated",
        cute_topk_sort_network="compact_pruned",
        cute_topk_merge_schedule="balanced",
        cute_topk_key_encoder="asm",
        cute_topk_defer_value_gathers=True,
    )
    assert "'compact_pruned', 'balanced')" in code
    assert "topk_output_max_rank" in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=torch.int32)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((value_storage[rows * k :] == 7).all())
    assert bool((index_storage[rows * k :] == -7).all())


# Seeds top-k tests.


@pytest.fixture
def _cpu_compile_environment() -> Iterator[None]:
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
        yield


@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize(
    "cols,k,primary_lanes",
    [(1, 1, 1), (17, 3, 1), (64, 32, 2), (128, 32, 4), (1024, 32, 32), (32768, 1, 1)],
)
def test_topk_seeds_follow_fragment_geometry(
    cols: int, k: int, primary_lanes: int
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=torch.bfloat16)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    assert "cute_topk" in spec.autotuner_heuristics
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_lanes_per_row"] == 16
    assert spec.default_config()["cute_topk_value_mode"] == "gather"
    assert spec.default_config()["cute_topk_sort_network"] == "batcher"
    seeds = [
        seed for seed in spec.compiler_seed_configs if "cute_topk_lanes_per_row" in seed
    ]
    legacy = [seed for seed in seeds if not seed["cute_topk_defer_value_gathers"]]
    assert 4 < len(legacy) <= 8
    assert len(legacy) < len(seeds) <= 14
    pairs = ConfigGeneration(spec).seed_flat_config_pairs()
    normalized = copy.deepcopy(seeds)
    for config in normalized:
        spec.normalize(config)
    assert [config for _flat, config in pairs[: len(seeds)]] == normalized
    primary = pairs[0][1]
    assert primary["cute_topk_lanes_per_row"] == primary_lanes
    assert primary["cute_topk_rows_per_block"] == 128 // primary_lanes
    assert primary["cute_topk_rank_mode"] == "ordinal"
    assert primary["cute_topk_value_mode"] == "decode"
    for _flat, config in pairs[:4]:
        spec.normalize(config)
        lanes = config["cute_topk_lanes_per_row"]
        rows = config["cute_topk_rows_per_block"]
        assert isinstance(lanes, int) and isinstance(rows, int)
        assert lanes <= min(32, 1 << (k - 1).bit_length())
        assert lanes * rows in (64, 128)
    assert pairs[3][1]["cute_topk_selection_layout"] == "distributed"
    assert all(seed["cute_topk_sort_network"] == "batcher" for seed in seeds[:4])
    assert all(seed["cute_topk_sort_network"] == "compact_pruned" for seed in seeds[4:])
    host_function = bound.host_function
    assert host_function is not None
    with bound.env, host_function:
        assert (
            CuteTopKHeuristic.get_seed_config(bound.env, host_function.device_ir)
            == spec.compiler_seed_configs[0]
        )


@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cols,k,distributed_lanes,native_lanes",
    [
        (3, 1, [1], [1]),
        (64, 32, [8, 4], [2, 1]),
        (65, 3, [16, 8], [4, 2]),
        (128, 32, [16, 8], [4, 2]),
        (256, 8, [32, 16], [8, 4]),
        (512, 8, [32], [8]),
        (32768, 1, [32], [1]),
    ],
)
def test_topk_new_seeds_cover_growing_and_native_fragments(
    dtype: torch.dtype,
    cols: int,
    k: int,
    distributed_lanes: list[int],
    native_lanes: list[int],
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, False))
    spec = bound.config_spec
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_topk_sort_network") == "compact_pruned"
        and not seed["cute_topk_defer_value_gathers"]
    ]
    assert [
        seed["cute_topk_lanes_per_row"]
        for seed in seeds
        if seed["cute_topk_selection_layout"] == "distributed"
    ] == distributed_lanes
    assert [
        seed["cute_topk_lanes_per_row"]
        for seed in seeds
        if seed["cute_topk_key_dtype"] == "float32_native"
    ] == native_lanes
    for seed in seeds:
        spec.normalize(copy.deepcopy(seed))
        lanes = seed["cute_topk_lanes_per_row"]
        rows = seed["cute_topk_rows_per_block"]
        vector = seed["cute_topk_vector_width"]
        assert isinstance(lanes, int) and isinstance(rows, int)
        assert isinstance(vector, int)
        assert lanes * rows == 128
        assert vector <= min(8, (cols + lanes - 1) // lanes)
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_output_vector_width"] == 4
        assert seed["cute_topk_key_dtype"] == (
            "int32"
            if seed["cute_topk_selection_layout"] == "distributed"
            else "float32_native"
        )


@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize(
    "dtype,inner_stride", [(torch.float32, 1), (torch.bfloat16, 2)]
)
def test_topk_seeds_reject_unsupported_roots(
    dtype: torch.dtype, inner_stride: int
) -> None:
    with FakeTensorMode():
        x = torch.empty_strided(
            (17, 64), (64 * inner_stride, inner_stride), dtype=dtype
        )
    bound = _allocating_topk._bind_isolated((x, 32, True))
    assert "cute_topk" not in bound.config_spec.autotuner_heuristics
    assert not any(
        "cute_topk_lanes_per_row" in config
        for config in bound.config_spec.compiler_seed_configs
    )


@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("alias_kind", ["separate", "view", "dlpack"])
def test_topk_seeds_require_final_runtime_alias_proof(alias_kind: str) -> None:
    x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    values = (
        x.clone()
        if alias_kind == "separate"
        else x.view_as(x)
        if alias_kind == "view"
        else torch.from_dlpack(x)
    )
    indices = torch.empty_like(x, dtype=torch.int64)
    bound = _out_topk._bind_isolated((x, values, indices))
    runtime_args: dict[str, object] = {"x": x, "values": values, "indices": indices}
    host_function = bound.host_function
    assert host_function is not None
    with bound.env, host_function, bound.env.use_runtime_arg_values(runtime_args):
        seeds = CuteTopKHeuristic.get_seed_configs(bound.env, host_function.device_ir)
    assert bool(seeds) == (alias_kind == "separate")


@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("alias_kind", ["separate", "view", "dlpack"])
@pytest.mark.parametrize("disable_heuristics", [False, True])
@pytest.mark.parametrize("width", [1, 32])
def test_topk_alias_fallback_retains_generic_search(
    alias_kind: str, disable_heuristics: bool, width: int
) -> None:
    x = torch.empty((17, width), dtype=torch.bfloat16)
    values = (
        torch.empty_like(x)
        if alias_kind == "separate"
        else x.view_as(x)
        if alias_kind == "view"
        else torch.from_dlpack(x)
    )
    indices = torch.empty_like(x, dtype=torch.int32)
    with patch.object(
        _out_topk.settings, "disable_autotuner_heuristics", disable_heuristics
    ):
        bound = _out_topk._bind_isolated((x, values, indices))
    spec = bound.config_spec
    specialized = alias_kind == "separate"
    assert spec.cute_topk_search_enabled == specialized
    config = spec.default_config()
    assert ("cute_topk_lanes_per_row" in config) == specialized
    assert ("_cute_local_topk" in bound.to_triton_code(config)) == specialized
    row = spec.block_sizes[0]
    if specialized:
        assert row.autotuner_min == row.max_size
        config.config.update(cute_topk_lanes_per_row=2, cute_topk_rows_per_block=64)
        spec.normalize(config)
        assert config["cute_topk_lanes_per_row"] == 2
        assert config["cute_topk_rows_per_block"] == 64
    else:
        assert row.max_size is None or row.autotuner_min < row.max_size
        small, large = copy.deepcopy(config), copy.deepcopy(config)
        small.config["block_sizes"] = [1]
        large.config["block_sizes"] = [16]
        spec.normalize(small)
        spec.normalize(large)
        assert small["block_sizes"] != large["block_sizes"]
        # Saved specialized configs can still be repaired for this fallback.
        large.config["cute_topk_lanes_per_row"] = 2
        spec.normalize(large, _fix_invalid=True)
        assert "cute_topk_lanes_per_row" not in large


@pytest.mark.usefixtures("_cpu_compile_environment")
def test_topk_seeds_registry_cache_and_disable_integration() -> None:
    assert CuteTopKHeuristic in get_heuristics("cute")
    assert CuteTopKHeuristic not in get_heuristics("triton")
    with FakeTensorMode():
        x = torch.empty((17, 64), dtype=torch.bfloat16)
    bound = _allocating_topk._bind_isolated((x, 32, True))
    spec = bound.config_spec
    structural_hash = spec.structural_fingerprint_hash()
    cache_hash = spec.cache_fingerprint_hash()
    spec.compiler_seed_configs = list(reversed(spec.compiler_seed_configs))
    assert spec.structural_fingerprint_hash() == structural_hash
    assert spec.cache_fingerprint_hash() != cache_hash
    with patch.object(_allocating_topk.settings, "disable_autotuner_heuristics", True):
        disabled = _allocating_topk._bind_isolated((x, 32, True))
    assert disabled.config_spec.compiler_seed_configs == []
    assert disabled.config_spec.cute_topk_search_enabled
    assert disabled.config_spec.compiler_default_config is None


@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cols,k,expected_lanes",
    [
        (1, 1, [1]),
        (17, 3, [1]),
        (64, 32, [2, 1]),
        (128, 32, [4, 2]),
        (129, 24, [8, 4]),
        (1024, 8, [8]),
        (32768, 1, [1]),
    ],
)
def test_topk_endpoint_encoder_seeds_are_unpromoted(
    dtype: torch.dtype, cols: int, k: int, expected_lanes: list[int]
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    default = spec.default_config()
    assert default["cute_topk_key_encoder"] == "dsl"
    assert default["cute_topk_defer_value_gathers"] is False
    assert default["cute_topk_merge_schedule"] == "sequential"
    assert spec.compiler_default_config is None
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_topk_defer_value_gathers", False)
    ]
    assert [
        (seed["cute_topk_lanes_per_row"], seed["cute_topk_key_encoder"])
        for seed in seeds
    ] == [
        (lanes, encoder)
        for lanes in expected_lanes
        for encoder in ("dsl", "asm", "paired")
    ]
    for seed in seeds:
        spec.normalize(copy.deepcopy(seed))
        lanes = seed["cute_topk_lanes_per_row"]
        assert isinstance(lanes, int)
        rows = seed["cute_topk_rows_per_block"]
        assert isinstance(rows, int)
        assert rows * lanes == 128
        assert seed["cute_topk_vector_width"] == 8
        assert seed["cute_topk_output_vector_width"] == 4
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_key_dtype"] == "int32"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_selection_layout"] == "replicated"
        assert seed["cute_topk_sort_network"] == "compact_pruned"
        assert seed["cute_topk_merge_schedule"] == "sequential"
    generation = ConfigGeneration(spec)
    flattened = generation.seed_flat_config_pairs()
    normalized = copy.deepcopy(seeds)
    for config in normalized:
        spec.normalize(config)
    assert [
        config
        for _, config in flattened
        if config.get("cute_topk_defer_value_gathers", False)
    ] == normalized
    for flat, config in flattened:
        assert generation.unflatten(flat) == config


# Paired top-k tests.


def _prmt(word: np.ndarray, selectors: int) -> np.ndarray:
    result = np.zeros_like(word)
    for output_byte in range(4):
        selector = (selectors >> (4 * output_byte)) & 15
        byte = (word >> (8 * (selector & 3))) & 255
        if selector & 8:
            byte = np.where(byte & 128, 255, 0).astype(np.uint32)
        result |= byte << (8 * output_byte)
    return result


def _lop3(a: np.ndarray, b: np.ndarray, c: int, table: int) -> np.ndarray:
    result = np.zeros_like(a)
    for index in range(8):
        if table & (1 << index):
            result |= (
                (a if index & 4 else ~a)
                & (b if index & 2 else ~b)
                & np.uint32(c if index & 1 else c ^ 0xFFFFFFFF)
            )
    return result


@pytest.mark.parametrize("infinity", [0x7C00, 0x7F80])
@pytest.mark.parametrize("bits", range(1, 16))
@pytest.mark.parametrize("largest", [False, True])
def test_paired_ordinal_exhaustive(infinity: int, bits: int, largest: bool) -> None:
    low = np.arange(65536, dtype=np.uint32)
    # A bijection exercises every possible word in both halves; additional
    # extremes catch cross-half carry and mixed finite/exceptional pairs.
    partners = (
        (low * 32771 + 1) & 65535,
        np.zeros_like(low),
        np.full_like(low, 65535),
        np.full_like(low, infinity),
    )
    bias = 32767 - infinity
    mask = (1 << bits) - 1
    for high in partners:
        words = low | (high << 16)
        sign = _prmt(words, 0xBB99)
        rank = _lop3(words, sign, 0x7FFF7FFF, 0x78)
        magnitude = words & 0x7FFF7FFF
        assert bool(((magnitude & 65535) + bias < 65536).all())
        assert bool(((magnitude >> 16) + bias < 65536).all())
        nan_mask = _prmt(magnitude + np.uint32(bias | (bias << 16)), 0xBB99)
        rank = _lop3(rank, nan_mask, 0x7FFF7FFF, 0xB8)
        ranks = (_prmt(rank, 0x9910).view(np.int32), _prmt(rank, 0xBB32).view(np.int32))
        for actual, word in zip(ranks, (low, high), strict=True):
            mag = (word & 32767).astype(np.int32)
            expected = np.where(word & 32768, -1 - mag, mag)
            expected = np.where(mag > infinity, 32767, expected).astype(np.int32)
            np.testing.assert_array_equal(actual, expected)
        for column in (0, (mask // 2) & ~1, mask - 1):
            for half, (rank, word) in enumerate(zip(ranks, (low, high), strict=True)):
                ordered = rank if largest else -rank
                packed = (ordered << bits) | (mask - column - half)
                restored = packed >> bits
                if not largest:
                    restored = -restored
                np.testing.assert_array_equal(restored, rank)
                np.testing.assert_array_equal(
                    mask - (packed & mask), np.full_like(rank, column + half)
                )
                recovered = (restored ^ ((restored >> 31) & 32767)) & 65535
                non_nan = (word & 32767) <= infinity
                np.testing.assert_array_equal(recovered[non_nan], word[non_nan])
                assert bool((restored[~non_nan] == 32767).all())
                if bits <= 9:
                    np.testing.assert_array_equal(
                        packed.astype(np.float32).astype(np.int32), packed
                    )
                if bits <= 14:
                    floating = (packed + 0x40000000).view(np.float32)
                    assert bool(np.isfinite(floating).all())
                    assert bool((floating > 0).all())


@pytest.mark.parametrize(
    "cols,stride,vector,rank,key,active",
    [
        (128, 128, 8, "ordinal", "int32", True),
        (128, 130, 2, "ordinal", "int32", True),
        (128, 130, 2, "ordinal", "float32_bits", True),
        (130, 132, 2, "ordinal", "int32", True),
        (130, 132, 2, "ordinal", "float32_bits", True),
        (128, 128, 4, "ordinal", "float32", True),
        (65, 66, 8, "ordinal", "int32", False),
        (128, 130, 8, "ordinal", "int32", False),
        (128, 128, 1, "ordinal", "int32", False),
        (128, 128, 8, "signed", "int32", False),
        (128, 128, 8, "ordinal", "float32_native", False),
        (1, 1, 8, "ordinal", "int32", False),
    ],
)
def test_paired_encoder_codegen_scope(
    cols: int, stride: int, vector: int, rank: str, key: str, active: bool
) -> None:
    code = _code(
        5,
        cols,
        stride,
        min(cols, 32),
        4,
        lanes=2,
        input_vector=vector,
        key_encoder="paired",
        rank_mode=rank,
        key_dtype=key,
        value_mode="decode",
        defer_value_gathers=True,
    )
    assert ("_cute_encode_ordinal_pair" in code) == active
    if cols > 1 and key != "float32_native":
        assert "_cute_encode_ordered_topk" in code
    if key == "float32_native":
        assert "_cute_encode_ordered_topk" not in code
    if active:
        if vector == 2:
            assert "ir.VectorType.get([1], cutlass.Uint32.mlir_type)" not in code
            assert "dtype=cutlass.Uint32), cutlass.Uint32)" in code
            assert "(topk_pairs, topk_col," in code
        else:
            assert (
                f"ir.VectorType.get([{vector // 2}], cutlass.Uint32.mlir_type)" in code
            )
        assert f"topk_input_bits.iterator.alignment >= {vector * 2}" in code
        assert "topk_keys[topk_i + 1]" in code
        assert "_cute_encode_ordered_topk" in code  # Scalar ABI fallback.


def test_paired_encoder_keeps_wide_address_math() -> None:
    code = _code(
        1, 128, 2**35, 32, 4, lanes=2, key_encoder="paired", rank_mode="ordinal"
    )
    assert "_cute_encode_ordinal_pair" in code
    assert "topk_row * cutlass.Int64(34359738368)" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,padding,offset,vector,key,layout",
    [
        (128, 32, 0, 0, 8, "int32", "replicated"),
        (128, 24, 2, 0, 2, "int32", "distributed"),
        (128, 24, 2, 0, 2, "float32_bits", "distributed"),
        (130, 24, 2, 0, 2, "int32", "distributed"),
        (130, 24, 2, 0, 2, "float32_bits", "distributed"),
        (65, 6, 1, 1, 4, "float32", "replicated"),
        (128, 32, 0, 1, 8, "int32", "replicated"),
    ],
)
def test_paired_encoder_exact_selected_bits(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    padding: int,
    offset: int,
    vector: int,
    key: str,
    layout: str,
) -> None:
    rows = 5
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original = storage.clone()
    value_storage = torch.full((rows * k + offset + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full(
        (rows * k + offset + 2,), -7, dtype=index_dtype, device=DEVICE
    )
    values = value_storage[offset : offset + rows * k].view(rows, k)
    indices = index_storage[offset : offset + rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=vector,
        cute_topk_output_vector_width=4,
        cute_topk_key_encoder="paired",
        cute_topk_key_dtype=key,
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout=layout,
        cute_topk_sort_network="compact_pruned",
        cute_topk_defer_value_gathers=True,
    )
    assert "_cute_local_topk" in code or "_cute_distributed_topk" in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((value_storage[:offset] == 7).all()) and bool(
        (value_storage[offset + rows * k :] == 7).all()
    )
    assert bool((index_storage[:offset] == -7).all()) and bool(
        (index_storage[offset + rows * k :] == -7).all()
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
def test_paired_encoder_same_bound_alignment_transition(
    dtype: torch.dtype, largest: bool
) -> None:
    rows, width, k = 5, 128, 32
    _, source = _layout_input(rows, width, dtype, 0, 0)
    storage = torch.cat((source, source[-1:]))
    values = torch.empty((rows, k), dtype=dtype, device=DEVICE)
    indices = torch.empty((rows, k), dtype=torch.int32, device=DEVICE)
    x = storage[: rows * width].view(rows, width)
    bound = _extra_out_topk._bind_isolated((x, values, indices, k, largest))
    bound.set_config(
        helion.Config(
            block_sizes=[1],
            cute_topk_lanes_per_row=2,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=4,
            cute_topk_key_encoder="paired",
            cute_topk_rank_mode="ordinal",
            cute_topk_value_mode="decode",
            cute_topk_defer_value_gathers=True,
        )
    )
    for offset in (0, 1, 0):
        x = storage[offset : offset + rows * width].view(rows, width)
        bound(x, values, indices, k, largest)
        _assert_topk_output(x, values, indices, k, largest, index_dtype=torch.int32)


def _composition_config(lanes: int, layout: str) -> dict[str, object]:
    return {
        "block_sizes": [1],
        "cute_topk_lanes_per_row": lanes,
        "cute_topk_rows_per_block": 4,
        "cute_topk_vector_width": 8,
        "cute_topk_output_vector_width": 4,
        "cute_topk_selection_layout": layout,
        "cute_topk_sort_network": "compact_pruned",
        "cute_topk_value_mode": "decode",
        "cute_topk_rank_mode": "ordinal",
    }


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_network_sort(
    x: torch.Tensor, descending: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(x)
    indices = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.sort(x[row, :], dim=-1, descending=descending)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("descending", [False, True])
def test_sort_uses_register_selection_network(descending: bool) -> None:
    with FakeTensorMode():
        x = torch.empty((17, 65), dtype=torch.bfloat16)
    bound = _row_network_sort._bind_isolated((x, descending))
    assert bound.config_spec.cute_topk_search_enabled
    config = bound.config_spec.default_config()
    config.config.update(_composition_config(8, "distributed"))
    code = bound.to_code(config)
    assert "_cute_distributed_topk" in code
    assert "sort_rank" not in code
    assert code.count("@cute.kernel") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("width,layout", [(16, "replicated"), (65, "distributed")])
def test_sort_network_preserves_values_indices_and_ties(
    dtype: torch.dtype, descending: bool, width: int, layout: str
) -> None:
    x = _inputs(width, dtype)
    original = x.clone()
    config = _composition_config(4, layout)
    # Sorting must enforce first-index ties even when a top-k tuning choice
    # would otherwise distinguish signed zeros or use native float keys.
    config.update(cute_topk_rank_mode="ordinal", cute_topk_key_dtype="float32_native")
    code, (values, indices) = code_and_output(
        _row_network_sort, (x, descending), **config
    )
    assert "sort_rank" not in code
    expected_values, expected_indices = torch.sort(
        original, dim=-1, descending=descending, stable=True
    )
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    assert torch.equal(values.view(torch.int16), expected_values.view(torch.int16))
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))
