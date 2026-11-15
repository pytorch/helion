"""torch.sort and torch.topk on Triton: padding lanes, 64-bit values and the
selections whose indices go unused.

Source checks and runs of the generated code in Triton's CPU interpreter make
up the CPU lane; the GPU check is marked ``cuda``.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.triton_codegen import cpu_codegen as cpu_codegen
from helion._testing.triton_codegen import interpret_triton_selection
from helion._utils import triton_is_available
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    # The kernels name their backend, so other backends' lanes would only
    # repeat these tests.
    skipUnlessBackends(["triton"]),
    pytest.mark.skipif(
        not triton_is_available(), reason="Triton is required for source generation"
    ),
]


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _row_selections(x: torch.Tensor):
    m, n = x.shape
    ascending = torch.empty_like(x)
    ascending_order = torch.empty((m, n), dtype=torch.int64, device=x.device)
    descending = torch.empty_like(x)
    descending_order = torch.empty((m, n), dtype=torch.int64, device=x.device)
    smallest = torch.empty((m, 3), dtype=x.dtype, device=x.device)
    smallest_order = torch.empty((m, 3), dtype=torch.int64, device=x.device)
    largest = torch.empty((m, 3), dtype=x.dtype, device=x.device)
    largest_order = torch.empty((m, 3), dtype=torch.int64, device=x.device)
    for row in hl.tile(m):
        up, up_order = torch.sort(x[row, :], dim=-1)
        ascending[row, :] = up
        ascending_order[row, :] = up_order
        down, down_order = torch.sort(x[row, :], dim=-1, descending=True)
        descending[row, :] = down
        descending_order[row, :] = down_order
        low, low_order = torch.topk(x[row, :], 3, dim=-1, largest=False)
        smallest[row, :] = low
        smallest_order[row, :] = low_order
        high, high_order = torch.topk(x[row, :], 3, dim=-1)
        largest[row, :] = high
        largest_order[row, :] = high_order
    return (
        ascending,
        ascending_order,
        descending,
        descending_order,
        smallest,
        smallest_order,
        largest,
        largest_order,
    )


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _row_selected_values(x: torch.Tensor):
    m, n = x.shape
    ascending = torch.empty_like(x)
    descending = torch.empty_like(x)
    smallest = torch.empty((m, 3), dtype=x.dtype, device=x.device)
    largest = torch.empty((m, 3), dtype=x.dtype, device=x.device)
    for row in hl.tile(m):
        ascending[row, :] = torch.sort(x[row, :], dim=-1).values
        descending[row, :] = torch.sort(x[row, :], dim=-1, descending=True).values
        smallest[row, :] = torch.topk(x[row, :], 3, dim=-1, largest=False).values
        largest[row, :] = torch.topk(x[row, :], 3, dim=-1).values
    return ascending, descending, smallest, largest


_SIGNED_OF_SIZE = {2: torch.int16, 4: torch.int32, 8: torch.int64}

# NaNs of either sign with payloads of their own, quiet and signaling.
_NAN_BITS = {
    torch.float16: (0xFE05, 0x7C01, 0xFC03),
    torch.bfloat16: (0xFFC5, 0x7F81, 0xFF83),
    torch.float32: (0xFFC00005, 0x7F800001, 0xFF800003),
    torch.float64: (0xFFF8000000000005, 0x7FF0000000000001, 0xFFF0000000000003),
}


def _interpret_on(backend, kernel, inputs, path, monkeypatch):
    """Run ``kernel``, a Triton kernel of this module, compiled for
    ``backend`` from its default config, on ``inputs`` in Triton's CPU
    interpreter."""
    with patch.dict(os.environ, {"ENABLE_TILE": "1"} if backend == "tileir" else {}):
        if backend != "triton":
            kernel = helion.kernel(
                kernel.fn, backend=backend, static_shapes=True, autotune_effort="none"
            )
        bound = kernel._bind_isolated(inputs)
        code = bound.to_code(bound.config_spec.default_config())
        # TileIR cannot legalize gather.
        assert backend != "tileir" or "tl.gather" not in code
        return interpret_triton_selection(code, inputs, path, monkeypatch)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of x, as signed integers of its width."""
    return x.view(_SIGNED_OF_SIZE[x.element_size()])


def _nans(dtype: torch.dtype) -> torch.Tensor:
    width = dtype.itemsize * 8
    signed = [bits - (bits >> (width - 1) << width) for bits in _NAN_BITS[dtype]]
    return torch.tensor(signed, dtype=_SIGNED_OF_SIZE[dtype.itemsize]).view(dtype)


def _padded_rows(dtype: torch.dtype) -> torch.Tensor:
    """Rows of 17 elements, which a block of 32 lanes holds, where zero padding
    would sort among the elements: positive ones (zeros first ascending),
    negative ones (zeros first descending), and ties with the extremes of the
    dtype (for floating point infinities, signed zeros and NaNs of either sign
    with payloads of their own)."""
    positive = torch.arange(17) * 5 % 7 + 1
    negative = -(torch.arange(17) * 3 % 5 + 1)
    if dtype.is_floating_point:
        nan, inf = float("nan"), float("inf")
        mixed = torch.tensor(
            [nan, 3, -inf, -0.0, 0, inf, -2, nan, 0, -0.0, 3, -2, 1, -1, inf, -inf, nan]
        ).to(dtype)
        _bits(mixed)[mixed.isnan()] = _bits(_nans(dtype))
    else:
        low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        mixed = torch.tensor(
            [high, 3, low, 0, 0, high, -2, low, 0, 3, -2, 1, -1, high, low, 5, -5],
            dtype=dtype,
        )
    return torch.stack([positive.to(dtype), negative.to(dtype), mixed])


def _stable_selections(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """_row_selections in eager: NaNs order above infinity, equal elements
    keep their input order, and top-3 takes the first three of a stable sort."""
    up = torch.sort(x, dim=-1, stable=True)
    down = torch.sort(x, dim=-1, descending=True, stable=True)
    return (
        *up,
        *down,
        up.values[:, :3],
        up.indices[:, :3],
        down.values[:, :3],
        down.indices[:, :3],
    )


def _assert_selections(x, actual, expected):
    """(values, indices) pairs match eager's, and float values are the bits of
    the elements their indices name, signed zeros and NaN payloads included."""
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
    if x.dtype.is_floating_point:
        for values, indices in zip(actual[::2], actual[1::2], strict=True):
            assert torch.equal(_bits(values), _bits(x).gather(-1, indices))


def _assert_selected_values(actual, expected):
    """Values-only selections match eager's and hold the same bits, NaN signs
    and payloads included, in any order among elements that compare equal."""
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
        if result.dtype.is_floating_point:
            assert torch.equal(
                _bits(result).sort(dim=-1).values, _bits(reference).sort(dim=-1).values
            )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    # Packed value/index keys for the floats of up to 32 bits and for
    # integers (without used indices, the sorts and the integer top-ks order
    # images of the values' own bits, 16-bit ones too, as FP16 to FP32
    # conversion drops NaN payloads on GPUs, while float top-k keeps the
    # keys); ranks for 64-bit integer and float values.
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int32,
        torch.int64,
        torch.float64,
    ],
)
@skipIfRefEager("eager's unstable sort and top-k order ties their own way")
@skipIfTileIR(
    "the kernels name the Triton backend, which gathers the selected values; "
    "this job's TileIR toolchain does not legalize tt.gather"
)
def test_selection_ignores_padding_lanes(dtype):
    x = _padded_rows(dtype).to(DEVICE)
    expected = _stable_selections(x)
    _assert_selections(x, _row_selections(x), expected)
    # Selections whose indices go unused order the values alone.
    _assert_selected_values(_row_selected_values(x), expected[::2])


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@skipIfRefEager("executes generated source through Triton's CPU interpreter")
def test_selected_values_keep_their_bits(cpu_codegen, dtype, tmp_path, monkeypatch):
    # The CPU lane's view of the values-only half of the GPU test above.
    x = _padded_rows(dtype)
    bound = _row_selected_values._bind_isolated((x,))
    actual = interpret_triton_selection(
        bound.to_code(bound.config_spec.default_config()),
        (x,),
        tmp_path / "selected_values.py",
        monkeypatch,
    )
    _assert_selected_values(actual, _stable_selections(x)[::2])


@skipIfRefEager("inspects generated source")
def test_selection_without_indices_orders_values_alone(cpu_codegen):
    def code(kernel, dtype):
        bound = kernel._bind_isolated((torch.ones(3, 17, dtype=dtype),))
        return bound.to_code(bound.config_spec.default_config())

    # Without used indices, 32-bit values sort as 32-bit images rather than
    # value/index keys; top-k keeps its keys to gather its k outputs.
    values_only = code(_row_selected_values, torch.float32)
    assert "sort_keys" not in values_only
    assert values_only.count("tl.gather(") == 2
    assert "sort_keys" in code(_row_selections, torch.float32)
    # 64-bit values skip the quadratic ranks behind the indices, and used
    # indices gather the values rather than sorting them too.
    for dtype in (torch.int64, torch.float64):
        assert "tl.sum(" not in code(_row_selected_values, dtype)
        with_indices = code(_row_selections, dtype)
        assert "tl.sum(" in with_indices
        assert "tl.sort(" not in with_indices
        assert "tl.topk(" not in with_indices


# Packed keys for float32; ranks for float64, whose values must still be the
# elements their indices name.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@skipIfRefEager("executes generated TileIR source through Triton's CPU interpreter")
def test_tileir_selection_ignores_padding_lanes(
    cpu_codegen, dtype, tmp_path, monkeypatch
):
    x = _padded_rows(dtype)
    actual = _interpret_on(
        "tileir", _row_selections, (x,), tmp_path / "tileir_selection.py", monkeypatch
    )
    _assert_selections(x, actual, _stable_selections(x))


def _narrow_rows(dtype: torch.dtype) -> torch.Tensor:
    """Two 17-element rows with ties and the extremes of ``dtype``. Triton's
    CPU interpreter decodes float8_e4m3fn NaNs as +-480, so they stay out."""
    if dtype == torch.bool:
        return torch.stack([torch.arange(17) % 3 == 0, torch.arange(17) % 2 == 1])
    if dtype.is_floating_point:
        row = torch.tensor(
            [3, 0.25, -448, -0.0, 0, 448, -2, 3, 0, -0.0, 3, -2, 1, -1, 448, -448, 0.5]
        )
    else:
        low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        row = torch.tensor(
            [high, 3, low, 0, 0, high, 2, low, 0, 3, 5, 1, 7, high, low, 5, 6]
        )
    return torch.stack([row, row.flip(0)]).to(dtype)


def _assert_narrow_selections(x, selections, selected_values):
    """_row_selections and _row_selected_values of the one-byte ``x`` match
    eager's over FP32, which holds every value of these dtypes and sorts
    stably: the same indices, and values that are the elements their indices
    name, signed zeros included."""
    expected = _stable_selections(x.to(torch.float32))
    for values, indices, reference, order in zip(
        selections[::2], selections[1::2], expected[::2], expected[1::2], strict=True
    ):
        assert torch.equal(indices, order)
        assert torch.equal(
            values.view(torch.uint8), x.view(torch.uint8).gather(-1, indices)
        )
        assert torch.equal(values.to(torch.float32), reference)
    for values, reference in zip(selected_values, expected[::2], strict=True):
        assert torch.equal(values.to(torch.float32), reference)


# Bools and uint8 take the unsigned images, int8 widens and truncates, and
# float8, which has no bit-exact image, goes through packed keys in every
# selection and reads its values at their indices: by tl.gather, or on
# TileIR as 8-bit payloads.
@pytest.mark.parametrize(
    ("dtype", "backend"),
    [
        (torch.bool, "triton"),
        (torch.uint8, "triton"),
        (torch.int8, "triton"),
        (torch.float8_e4m3fn, "triton"),
        (torch.float8_e4m3fn, "tileir"),
    ],
)
@skipIfRefEager("executes generated selections through Triton's CPU interpreter")
def test_narrow_values_select_like_eager(
    cpu_codegen, dtype, backend, tmp_path, monkeypatch
):
    x = _narrow_rows(dtype)
    selections = _interpret_on(
        backend, _row_selections, (x,), tmp_path / "narrow_selections.py", monkeypatch
    )
    selected_values = _interpret_on(
        backend, _row_selected_values, (x,), tmp_path / "narrow_values.py", monkeypatch
    )
    _assert_narrow_selections(x, selections, selected_values)


# int8 images and float8's casts to FP32 and gathers, compiled for a GPU; the
# interpreter above does not run those lowerings as a GPU does.
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
@skipIfRefEager("eager sorts neither float8 nor ties in a stable order")
@skipIfTileIR(
    "the kernels name the Triton backend, which gathers the selected values; "
    "this job's TileIR toolchain does not legalize tt.gather"
)
def test_narrow_values_select_like_eager_on_gpu(dtype):
    if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Triton's fp8e4nv needs CUDA capability 8.9")
    x = _narrow_rows(dtype).to(DEVICE)
    _assert_narrow_selections(x, _row_selections(x), _row_selected_values(x))


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _divided_sort(x: torch.Tensor, divisor: torch.Tensor):
    values = torch.empty_like(x)
    widened = torch.empty(x.size(), dtype=torch.float32, device=x.device)
    indices = torch.empty(x.size(), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        sorted_values, order = torch.sort(x[row, :] / divisor[row, :], dim=-1)
        values[row, :] = sorted_values
        # No store rounds these: the sort must return the rounded values.
        widened[row, :] = sorted_values.to(torch.float32)
        indices[row, :] = order
    return values, widened, indices


def _divided_rows(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows and divisors of 3 whose quotients differ in FP32 but tie in pairs
    once rounded to ``dtype`` (1538 / 3 and 1537 / 3 round to 512.5 in
    float16; 388 / 3 and 386 / 3 to 129 in bfloat16)."""
    start = 1538 if dtype == torch.float16 else 388
    x = (start - torch.arange(34)).to(dtype).view(2, 17)
    return x, torch.full_like(x, 3)


def _assert_divided_sort(x, divisor, values, widened, indices) -> None:
    # The division keeps an FP32 quotient, but ties keep their input order,
    # as eager's stable sort of the rounded quotients does, and the values
    # are the rounded quotients wherever they go.
    expected = torch.sort(x / divisor, dim=-1, stable=True)
    assert torch.equal(indices, expected.indices)
    assert torch.equal(values, expected.values)
    assert torch.equal(widened, expected.values.to(torch.float32))


# tl.gather reads the values, or TileIR's per-slot payloads.
@pytest.mark.parametrize("backend", ["triton", "tileir"])
@skipIfRefEager("executes generated sorts through Triton's CPU interpreter")
def test_sort_indices_follow_rounded_values(
    cpu_codegen, backend, tmp_path, monkeypatch
):
    # Triton's CPU interpreter rounds FP32 to bfloat16 by truncation, so
    # bfloat16 runs on the GPU only.
    inputs = _divided_rows(torch.float16)
    results = _interpret_on(
        backend, _divided_sort, inputs, tmp_path / "divided_sort.py", monkeypatch
    )
    _assert_divided_sort(*inputs, *results)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipIfRefEager("eager's unstable sort orders ties its own way")
@skipIfTileIR(
    "the kernels name the Triton backend, which gathers the selected values; "
    "this job's TileIR toolchain does not legalize tt.gather"
)
def test_sort_indices_follow_rounded_values_on_gpu(dtype):
    inputs = tuple(t.to(DEVICE) for t in _divided_rows(dtype))
    _assert_divided_sort(*inputs, *_divided_sort(*inputs))


def _jagged_sorts(x: torch.Tensor, lengths: torch.Tensor):
    ascending = torch.zeros_like(x)
    descending = torch.zeros_like(x)
    order = torch.zeros(x.size(), dtype=torch.int64, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        row_lengths = lengths[tile_b]
        for tile_k in hl.jagged_tile(row_lengths):
            ascending[tile_b, tile_k] = torch.sort(x[tile_b, tile_k], dim=-1).values
            down, down_order = torch.sort(x[tile_b, tile_k], dim=-1, descending=True)
            descending[tile_b, tile_k] = down
            order[tile_b, tile_k] = down_order
    return ascending, descending, order


# Packed keys and ranks, each with and without used indices.
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@skipIfRefEager("executes generated jagged sorts through Triton's CPU interpreter")
def test_sort_inside_a_jagged_tile(cpu_codegen, dtype, tmp_path, monkeypatch):
    # A jagged tile's mask spans its parent's rows, [rows, columns]. Each
    # 4-wide tile of a row sorts the elements before the row's length.
    x = _ties((5, 11), dtype)
    lengths = torch.tensor([11, 3, 0, 7, 5])
    bound = helion.kernel(
        _jagged_sorts, backend="triton", static_shapes=True, autotune_effort="none"
    )._bind_isolated((x, lengths))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [2, 4]
    ascending, descending, order = interpret_triton_selection(
        bound.to_code(config), (x, lengths), tmp_path / "jagged_sort.py", monkeypatch
    )
    for row, length in enumerate(lengths.tolist()):
        for start in range(0, length, 4):
            tile = slice(start, min(start + 4, length))
            _assert_selections(
                x[row, tile],
                (descending[row, tile], order[row, tile]),
                torch.sort(x[row, tile], descending=True, stable=True),
            )
            _assert_selected_values(
                (ascending[row, tile],), (torch.sort(x[row, tile]).values,)
            )
        # The lanes past the row's length are not stored.
        for result in (ascending, descending, order):
            assert not result[row, length:].any()


def _ties(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Elements of both signs with many ties; for floating point also signed
    zeros and NaNs of either sign with payloads of their own."""
    x = (torch.arange(torch.Size(shape).numel()).reshape(shape) * 7 % 11 - 5).to(dtype)
    if dtype.is_floating_point:
        x[..., 1::8] = -0.0
        nans = _bits(_nans(dtype))
        for i, column in enumerate(range(5, shape[-1], 12)):
            _bits(x)[..., column] = nans[i % len(nans)]
    return x


def _tile_sort(x: torch.Tensor):
    values = torch.empty_like(x)
    indices = torch.empty(x.size(), dtype=torch.int64, device=x.device)
    for tile_m, tile_n in hl.tile(x.size()):
        sorted_values, order = torch.sort(x[tile_m, tile_n], dim=-1, descending=True)
        values[tile_m, tile_n] = sorted_values
        indices[tile_m, tile_n] = order
    return values, indices


# Packed keys, and ranks for 64-bit integers and floats.
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float64])
@skipIfRefEager("executes generated selection over a tile through the CPU interpreter")
def test_sort_indices_over_a_partial_tile(cpu_codegen, dtype, tmp_path, monkeypatch):
    # 48-element rows in tiles of 32: the second tile holds 16 elements and 16
    # lanes past the end of the row, which take no rank. Each tile sorts on its
    # own; its indices count from the tile's start.
    x = _ties((5, 48), dtype)
    bound = helion.kernel(
        _tile_sort, backend="triton", static_shapes=True, autotune_effort="none"
    )._bind_isolated((x,))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [8, 32]
    values, indices = interpret_triton_selection(
        bound.to_code(config), (x,), tmp_path / "tile_sort.py", monkeypatch
    )
    for start in (0, 32):
        tile = slice(start, start + 32)
        _assert_selections(
            x[:, tile],
            (values[:, tile], indices[:, tile]),
            torch.sort(x[:, tile], dim=-1, descending=True, stable=True),
        )


def _tile_topk(x: torch.Tensor):
    values = torch.empty((x.size(0), 4), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), 4), dtype=torch.int64, device=x.device)
    for tile_m, tile_n in hl.tile(x.size()):
        selected, order = torch.topk(x[tile_m, tile_n], 4, dim=-1, largest=False)
        values[tile_m, :] = selected
        indices[tile_m, :] = order
    return values, indices


# Packed keys for float32 and int32, ranks for int64.
@pytest.mark.parametrize(
    ("operation", "dtype"),
    [("sort", torch.float32), ("topk", torch.int32), ("topk", torch.int64)],
)
@skipIfRefEager("executes generated selection over a tile through the CPU interpreter")
def test_selection_indices_over_a_tile(
    cpu_codegen, operation, dtype, tmp_path, monkeypatch
):
    # The positions behind the indices span the selected value's last dim:
    # the tile's block of 8, not the size hint of its extent.
    if operation == "sort":
        x = torch.randn(8, 32, dtype=dtype)
        kernel = _tile_sort
    else:
        x = torch.randperm(64).to(dtype).view(8, 8)
        kernel = _tile_topk
    bound = helion.kernel(
        kernel, backend="triton", static_shapes=True, autotune_effort="none"
    )._bind_isolated((x,))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [2, 8]
    values, indices = interpret_triton_selection(
        bound.to_code(config), (x,), tmp_path / "tile_selection.py", monkeypatch
    )
    if operation == "sort":
        # Each 8-wide tile of a 32-wide row sorts on its own; its indices
        # count from the tile's start.
        for start in range(0, 32, 8):
            tile = slice(start, start + 8)
            _assert_selections(
                x[:, tile],
                (values[:, tile], indices[:, tile]),
                torch.sort(x[:, tile], dim=-1, descending=True, stable=True),
            )
    else:
        _assert_selections(
            x, (values, indices), torch.topk(x, 4, dim=-1, largest=False)
        )


def _batched_selections(x: torch.Tensor):
    batches, rows, n = x.shape
    values = torch.empty_like(x)
    indices = torch.empty(x.size(), dtype=torch.int64, device=x.device)
    smallest = torch.empty((batches, rows, 3), dtype=x.dtype, device=x.device)
    smallest_order = torch.empty((batches, rows, 3), dtype=torch.int64, device=x.device)
    for batch, row in hl.tile([batches, rows]):
        sorted_values, order = torch.sort(x[batch, row, :], dim=-1)
        values[batch, row, :] = sorted_values
        indices[batch, row, :] = order
        low, low_order = torch.topk(x[batch, row, :], 3, dim=-1, largest=False)
        smallest[batch, row, :] = low
        smallest_order[batch, row, :] = low_order
    return values, indices, smallest, smallest_order


# Packed keys, and ranks for 64-bit integers and floats.
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float64])
@skipIfRefEager("executes generated dynamic-shape selections through the interpreter")
def test_selections_over_3d_values_of_dynamic_width(
    cpu_codegen, dtype, tmp_path, monkeypatch
):
    # Two tiled dims of blocks 2 make each value 3-D, with partial tiles in
    # both. One kernel, bound at width 17, serves every width.
    bound = helion.kernel(
        _batched_selections,
        backend="triton",
        static_shapes=False,
        autotune_effort="none",
    )._bind_isolated((_ties((3, 5, 17), dtype),))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [2, 2]
    code = bound.to_code(config)
    for width in (17, 19, 33):
        x = _ties((3, 5, width), dtype)
        actual = interpret_triton_selection(
            code, (x,), tmp_path / "batched_selections.py", monkeypatch
        )
        up = torch.sort(x, dim=-1, stable=True)
        expected = (*up, up.values[..., :3], up.indices[..., :3])
        _assert_selections(x, actual, expected)


@skipIfRefEager("executes generated sorts through the CPU interpreter")
def test_sort_beside_a_flattened_tile(cpu_codegen, tmp_path, monkeypatch):
    # flatten_loops makes the two tile dims one dimension of the value. Rows
    # behind it sort on their own; a sort along it would run across the tile,
    # so it pins the loop unflattened, and codegen refuses one that still
    # reaches it.
    def sort_rows(x: torch.Tensor):
        values = torch.empty_like(x)
        indices = torch.empty(x.size(), dtype=torch.int64, device=x.device)
        for batch, row in hl.tile(x.shape[:-1]):
            sorted_values, order = torch.sort(x[batch, row, :], dim=-1)
            values[batch, row, :] = sorted_values
            indices[batch, row, :] = order
        return values, indices

    def sort_tiles(x: torch.Tensor):
        values = torch.empty_like(x)
        for tile_a, tile_b in hl.tile(x.size()):
            values[tile_a, tile_b] = torch.sort(x[tile_a, tile_b], dim=-1).values
        return values

    def flattened(kernel, x: torch.Tensor):
        bound = helion.kernel(
            kernel, backend="triton", static_shapes=True, autotune_effort="none"
        )._bind_isolated((x,))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[2, 2], flatten_loops=[True])
        return bound, config

    x = torch.randn(2, 4, 17)
    bound, config = flattened(sort_rows, x)
    values, indices = interpret_triton_selection(
        bound.to_code(config), (x,), tmp_path / "flattened_sort.py", monkeypatch
    )
    expected = torch.sort(x, dim=-1, stable=True)
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected.indices, rtol=0, atol=0)
    bound, config = flattened(sort_tiles, x[0])
    (spec,) = bound.config_spec.flatten_loops
    assert spec.pinned
    with patch("helion._compiler.compile_environment.BlockSizeInfo.pin_unflattened"):
        bound, config = flattened(sort_tiles, x[0])
        with pytest.raises(helion.exc.InvalidConfig, match="dim 1 of a flattened"):
            bound.to_code(config)
