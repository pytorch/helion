"""CuTe top-k lowering, selection networks, tuning, and correctness tests."""

from __future__ import annotations

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
    "rows,width,k,dtype,lanes,block_rows,vector,padding,offset",
    [
        pytest.param(3, 1, 1, torch.bfloat16, 1, 1, 1, 0, 0, id="singleton-min-cta"),
        pytest.param(
            17, 65, 65, torch.float16, 8, 4, 2, 4, 1, id="all-elements-odd-stride"
        ),
        pytest.param(
            35, 65, 3, torch.bfloat16, 32, 32, 4, 2, 1, id="max-cta-odd-stride"
        ),
        pytest.param(9, 33, 17, torch.float16, 2, 2, 4, 0, 1, id="vector4-offset"),
        pytest.param(
            7, 31, 1, torch.bfloat16, 8, 1, 1, 5, 0, id="unit-vector-row-stride"
        ),
        pytest.param(17, 64, 3, torch.bfloat16, 16, 8, 2, 0, 0, id="vector2-aligned"),
        pytest.param(
            129, 64, 32, torch.bfloat16, 1, 128, 8, 0, 0, id="one-lane-128-rows-tail"
        ),
        pytest.param(
            65, 128, 32, torch.float16, 2, 64, 8, 0, 0, id="two-lanes-64-rows-tail"
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
    )
    assert "_cute_local_topk" in code
    assert "cute.autovec_copy" not in code
    _assert_topk_output(original, values, indices, k)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


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
                (rows, cols), (stride, 1), dtype=dtype, device=torch.device("cpu")
            )
        bound = kernel._bind_isolated((x, k, largest, index_dtype))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_topk_lanes_per_row=lanes,
            cute_topk_rows_per_block=8,
            cute_topk_vector_width=input_vector,
        )
        return bound.to_triton_code(config)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vector", [1])
@pytest.mark.parametrize("layout", ["replicated"])
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
