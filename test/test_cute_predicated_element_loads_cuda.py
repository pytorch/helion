"""GPU tests of ``cute_predicated_element_loads``: masked element loads as
predicated loads, with and without the constant of the select they feed,
store what eager PyTorch does: padded windows over bf16, fp16 and fp32
images at even and odd sizes, one element per thread and lanes, partial
tiles on every axis, NaNs, infinities and negative zeros, images that are
offset, strided or permuted views, an output in the storage of the image,
a store into the tensor the gather reads, gathers through loaded int32 and
int64 indices with out-of-range entries, a mask over another lane than the
gathered one, kernels compiled for symbolic sizes, and the predicated
load's cache policy; and, for a fill near a rounding tie of bf16 and fp16,
what the kernel without the knob stores."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import pairwise
from helion._testing.predicated_load_kernels import gathered_or_fill
from helion._testing.predicated_load_kernels import gathered_or_fill_reference
from helion._testing.predicated_load_kernels import gathered_under_row_mask
from helion._testing.predicated_load_kernels import gathered_under_row_mask_reference
from helion._testing.predicated_load_kernels import interior_or_fill_in_place
from helion._testing.predicated_load_kernels import interior_or_fill_reference
from helion._testing.predicated_load_kernels import padded_window_max
from helion._testing.predicated_load_kernels import padded_window_max_reference
from helion._testing.predicated_load_kernels import padded_window_shape
from helion._testing.predicated_load_kernels import strided_pair_sum
from helion._testing.predicated_load_kernels import strided_pair_sum_reference

_KNOB = "cute_predicated_element_loads"
_KNOBS: dict[str, dict[str, object]] = {
    "off": {},
    "predicated": {_KNOB: True},
    "guarded": {_KNOB: True, "cute_guarded_lanes": True},
    "guarded_fmnmx": {
        _KNOB: True,
        "cute_guarded_lanes": True,
        "cute_fmnmx_extrema": True,
    },
}
# The settings with the knob on.  Eager PyTorch is the reference of these
# tests, so the kernel without the knob is built only where a test compares
# against it.
_ON = ("predicated", "guarded", "guarded_fmnmx")
# The window's tiles: one element per thread with partial last tiles of the
# planes and the columns; two rows and two columns per thread; a vector
# loop along the columns.
_WINDOW_TILES: dict[str, dict[str, object]] = {
    "elements": {"block_sizes": [2, 1, 16], "num_threads": [2, 1, 16]},
    "lanes": {"block_sizes": [4, 2, 8], "num_threads": [4, 1, 4]},
    "vector": {
        "block_sizes": [1, 1, 16],
        "num_threads": [1, 1, 2],
        "cute_vector_widths": [1, 1, 8],
    },
}
_KERNELS: dict[tuple[object, str, bool], object] = {}


def _kernel(
    kernel: object, config: dict[str, object], *, static_shapes: bool = True
) -> object:
    """``kernel`` at ``config``, built once per module."""
    key = (kernel, repr(sorted(config.items())), static_shapes)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            kernel.fn,  # type: ignore[attr-defined]
            backend="cute",
            static_shapes=static_shapes,
            config=helion.Config.from_dict(config),
        )
    return _KERNELS[key]


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(
        {2: torch.int16, 4: torch.int32, 8: torch.int64}[value.element_size()]
    )


def _assert_same_values(
    actual: torch.Tensor, expected: torch.Tensor, what: object
) -> None:
    """NaNs in the same places and the same bits everywhere else.  (Which
    NaN an extremum yields is not part of the result: without the selects
    the assembler narrows an fp32 chain over widened half elements to
    half-precision instructions, whose NaN differs in its sign bit.)"""
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan), what
    assert torch.equal(_bits(actual)[~nan], _bits(expected)[~nan]), what


def _image(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
    *,
    specials: bool = True,
) -> torch.Tensor:
    """An image without zeros of both signs in one window (so the sign of a
    zero maximum cannot differ), with NaNs and infinities among its values
    on request."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=generator) * 2.0 - 1.0
    x = torch.where(x.abs() < 0.01, torch.ones_like(x), x)
    if specials:
        flat = x.view(-1)
        flat[::17] = float("nan")
        flat[3::11] = float("inf")
        flat[5::13] = float("-inf")
    return x.to(dtype).to(DEVICE)


def _pooled(kernel: object, x: torch.Tensor) -> torch.Tensor:
    out = torch.full(padded_window_shape(x), 123.0, dtype=x.dtype, device=x.device)
    kernel(x, out)  # type: ignore[operator]
    return out


_SIZES = {"even": (5, 12, 20), "odd": (5, 13, 27)}


# All pairs of the tile shapes, the image sizes and the knob settings (9 of
# their 18 combinations, one compile each).  The tiles decide which loads
# are single elements, the knobs which of them take a fill, and the odd
# size puts the last tap of a row and of a column in the padding and makes
# the last tiles partial on every axis.
@pytest.mark.parametrize(
    ("tiles", "size", "knobs"),
    pairwise(tiles=list(_WINDOW_TILES), size=list(_SIZES), knobs=_ON),
)
def test_padded_window_maxima_match_eager(tiles: str, size: str, knobs: str) -> None:
    x = _image(_SIZES[size])
    expected = padded_window_max_reference(x)
    assert torch.isnan(expected).any() and torch.isinf(expected).any()
    kernel = _kernel(padded_window_max, {**_WINDOW_TILES[tiles], **_KNOBS[knobs]})
    _assert_same_values(_pooled(kernel, x), expected, (tiles, size, knobs))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_padded_window_maxima_of_other_floats_match_eager(dtype: torch.dtype) -> None:
    """fp16 (a 2-byte element through the 32-bit carrier) and fp32 (the
    select's own type: no cast between the load and the select)."""
    x = _image((5, 13, 27), dtype)
    expected = padded_window_max_reference(x)
    for knobs in _ON:
        kernel = _kernel(
            padded_window_max, {**_WINDOW_TILES["elements"], **_KNOBS[knobs]}
        )
        _assert_same_values(_pooled(kernel, x), expected, (dtype, knobs))


def test_all_padding_and_negative_zero_windows() -> None:
    """A one-pixel image (every other tap is padding), and windows whose
    only finite values are ``-0.0``: the fill is ``-inf``, not a zero."""
    pixel = _kernel(
        padded_window_max,
        {"block_sizes": [2, 1, 1], "num_threads": [2, 1, 1], **_KNOBS["guarded"]},
    )
    one = torch.tensor([[[-3.0]], [[float("-inf")]]], dtype=torch.bfloat16)
    one = one.to(DEVICE)
    assert torch.equal(_bits(_pooled(pixel, one)), _bits(one))
    kernel = _kernel(
        padded_window_max, {**_WINDOW_TILES["elements"], **_KNOBS["guarded"]}
    )
    zeros = torch.full((3, 6, 10), -0.0, dtype=torch.bfloat16, device=DEVICE)
    pooled = _pooled(kernel, zeros)
    assert torch.equal(_bits(pooled), _bits(padded_window_max_reference(zeros)))
    assert torch.signbit(pooled).all()


@pytest.mark.parametrize("view", ["offset", "strided", "permuted", "expanded"])
def test_padded_window_of_a_view_matches_eager(view: str) -> None:
    """The image as a view of a larger tensor: a window that starts one
    plane and one row into its storage, every other row of it, a
    permutation whose last dim is not contiguous, and one row expanded to
    thirteen (a stride of zero: every tap of a column reads one element)."""
    base = _image((7, 28, 20))
    if view == "offset":
        x = base[1:6, 1:14]
    elif view == "strided":
        x = base[1:6, 1::2]
    elif view == "permuted":
        x = _image((20, 5, 13)).permute(1, 2, 0)
    else:
        x = _image((5, 1, 20)).expand(5, 13, 20)
    assert not x.is_contiguous()
    expected = padded_window_max_reference(x)
    for tiles in ("elements", "lanes"):
        for knobs in ("predicated", "guarded_fmnmx"):
            kernel = _kernel(
                padded_window_max, {**_WINDOW_TILES[tiles], **_KNOBS[knobs]}
            )
            _assert_same_values(_pooled(kernel, x), expected, (view, tiles, knobs))


def test_output_in_the_storage_of_the_image_matches_eager() -> None:
    """The output is another region of the buffer the image lives in, so
    the kernel cannot tell the two apart: the predicated loads keep their
    order with the stores."""
    buffer = _image((5, 13, 27 + 14))
    x = buffer[:, :, :27]
    out = buffer[:, :7, 27:]
    expected = padded_window_max_reference(x.clone())
    untouched = x.clone()
    for tiles in ("elements", "lanes"):
        kernel = _kernel(
            padded_window_max, {**_WINDOW_TILES[tiles], **_KNOBS["guarded_fmnmx"]}
        )
        kernel(x, out)  # type: ignore[operator]
        _assert_same_values(out, expected, tiles)
        nan = torch.isnan(untouched)
        assert torch.equal(_bits(x)[~nan], _bits(untouched)[~nan])


def test_store_into_the_gathered_tensor_matches_eager() -> None:
    """Each element is read through a masked gather and stored back in
    place; the select keeps its place here (a decline of the fill)."""
    x = _image((3, 9, 21), specials=False)
    y = _image((3, 9, 21), specials=False).flip(0)
    expected = interior_or_fill_reference(x, y)
    config = {"block_sizes": [2, 2, 16], "num_threads": [2, 2, 16]}
    for knobs in ("predicated", "guarded"):
        actual = x.clone()
        _kernel(interior_or_fill_in_place, {**config, **_KNOBS[knobs]})(actual, y)  # type: ignore[operator]
        assert torch.equal(_bits(actual), _bits(expected)), knobs


def test_added_zero_of_a_masked_gather_matches_eager() -> None:
    x = _image((6, 37), specials=False)
    expected = strided_pair_sum_reference(x)
    config = {"block_sizes": [2, 16], "num_threads": [2, 16]}
    for knobs in ("predicated", "guarded"):
        actual = _kernel(strided_pair_sum, {**config, **_KNOBS[knobs]})(x)  # type: ignore[operator]
        assert torch.equal(_bits(actual), _bits(expected)), knobs


# The fill in the tensor's own type for four element types (fp32, an
# integer, a rounded bf16 constant, an 8-byte element), each through an int32
# and an int64 index: 4 x 2, the knobs looped inside.
@pytest.mark.parametrize(
    ("dtype", "fill"),
    [
        (torch.float32, 2.5),
        (torch.int32, -1),
        (torch.bfloat16, 0.1),
        (torch.float64, -2.0),
    ],
    ids=["fp32", "int32", "bf16", "fp64"],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
def test_gather_through_a_loaded_index_matches_eager(
    dtype: torch.dtype, fill: float, index_dtype: torch.dtype
) -> None:
    """Indices below zero and past the row are masked off and read as the
    fill; an int64 index far outside the int32 range must not load."""
    generator = torch.Generator().manual_seed(1)
    if dtype.is_floating_point:
        x = _image((6, 37), dtype, specials=False)
    else:
        x = torch.randint(-50, 50, (6, 37), generator=generator).to(dtype).to(DEVICE)
    index = torch.randint(-5, 42, (21,), generator=generator).to(index_dtype)
    if index_dtype is torch.int64:
        index[3] = (1 << 32) + 4
        index[8] = -(1 << 32) + 2
    index = index.to(DEVICE)
    expected = gathered_or_fill_reference(x, index, fill)
    config = {"block_sizes": [2, 16], "num_threads": [2, 16]}
    for knobs in ("predicated", "guarded"):
        kernel = _kernel(gathered_or_fill, {**config, **_KNOBS[knobs]})
        actual = kernel(x, index, fill)  # type: ignore[operator]
        assert torch.equal(_bits(actual), _bits(expected)), knobs


# A constant within an fp32 rounding of a tie of the half type: one value
# rounded once, as the DSL casts a Python float, another through fp32, as
# eager PyTorch rounds it.
@pytest.mark.parametrize(
    ("dtype", "fill"),
    [
        (torch.bfloat16, 1.0 + 2.0**-8 + 2.0**-40),
        (torch.float16, 1.0 + 2.0**-11 + 2.0**-40),
    ],
    ids=["bf16", "fp16"],
)
def test_half_fill_near_a_rounding_tie_stores_what_the_select_did(
    dtype: torch.dtype, fill: float
) -> None:
    """The masked-off elements hold what the kernel without the knob
    stores, bit for bit (its select's cast of the constant, one unit in the
    last place above eager PyTorch's value): the fill of such a constant
    is not folded into the load."""
    x = _image((6, 37), dtype, specials=False)
    index = torch.tensor(
        [0, -1, 5, 40, 36, 37, 3, -7], dtype=torch.int32, device=DEVICE
    )
    config = {"block_sizes": [2, 8], "num_threads": [2, 8]}
    outputs = {
        knobs: _kernel(gathered_or_fill, {**config, **_KNOBS[knobs]})(x, index, fill)  # type: ignore[operator]
        for knobs in ("off", "predicated", "guarded")
    }
    inside = (index >= 0) & (index < 37)
    expected = gathered_or_fill_reference(x, index, fill)
    for knobs, actual in outputs.items():
        assert torch.equal(_bits(actual), _bits(outputs["off"])), knobs
        assert torch.equal(_bits(actual[:, inside]), _bits(expected[:, inside])), knobs
    assert not inside.all()


def test_mask_over_another_lane_keeps_the_gathered_bound() -> None:
    """The mask's ``col < columns`` is about the row's lane, not the
    gathered one: columns past the row, under a mask that holds, must not
    be read (they would read the next row) and store the masked load's
    zero.  One tile holds the whole output, so the mask's lane is the
    row."""
    x = _image((8, 37), torch.float32, specials=False)
    index = torch.tensor([0, 36, 37, 5, 40, 3, 64, 1], dtype=torch.int32, device=DEVICE)
    expected = gathered_under_row_mask_reference(x, index)
    assert (expected == 0).any() and (expected == -1).any()
    config = {"block_sizes": [8, 8], "num_threads": [8, 8]}
    for knobs in ("predicated", "guarded"):
        actual = _kernel(gathered_under_row_mask, {**config, **_KNOBS[knobs]})(x, index)  # type: ignore[operator]
        assert torch.equal(_bits(actual), _bits(expected)), knobs


def test_dynamic_shapes_match_eager() -> None:
    """One kernel compiled for symbolic sizes and run at several: the
    window's bound is then a symbol the mask cannot be compared with (the
    taps load predicated with their bounds and selects kept), and so is the
    gathered dim's."""
    for knobs in ("predicated", "guarded_fmnmx"):
        window = _kernel(
            padded_window_max,
            {**_WINDOW_TILES["elements"], **_KNOBS[knobs]},
            static_shapes=False,
        )
        for shape in ((5, 13, 27), (3, 8, 10), (2, 9, 1)):
            x = _image(shape)
            _assert_same_values(
                _pooled(window, x), padded_window_max_reference(x), (knobs, shape)
            )
    generator = torch.Generator().manual_seed(2)
    for knobs in ("predicated", "guarded"):
        gather = _kernel(
            gathered_or_fill,
            {"block_sizes": [2, 16], "num_threads": [2, 16], **_KNOBS[knobs]},
            static_shapes=False,
        )
        for columns, count in ((37, 21), (9, 40)):
            x = _image((6, columns), torch.float32, specials=False)
            index = torch.randint(-5, columns + 5, (count,), generator=generator)
            index[1] = (1 << 32) + 4
            index[2] = -(1 << 32) + 2
            index = index.to(DEVICE)
            actual = gather(x, index, 2.5)  # type: ignore[operator]
            expected = gathered_or_fill_reference(x, index, 2.5)
            assert torch.equal(_bits(actual), _bits(expected)), (knobs, columns)


def test_hinted_predicated_loads_match_eager() -> None:
    """The predicated load with the tensor's cache policy on it: an L1
    priority and an L2 family on bf16 elements (``cute_narrow_load_hints``)
    and an L2 family on fp32 ones (``cute_scalar_load_hints``)."""
    for dtype, hint_knob, policy in (
        (torch.bfloat16, "cute_narrow_load_hints", "last"),
        (torch.bfloat16, "cute_narrow_load_hints", "l2_last"),
        (torch.float32, "cute_scalar_load_hints", "l1_l2_last"),
    ):
        x = _image((5, 13, 27), dtype)
        kernel = _kernel(
            padded_window_max,
            {
                **_WINDOW_TILES["elements"],
                **_KNOBS["guarded_fmnmx"],
                hint_knob: True,
                "load_eviction_policies": [policy] * 9,
            },
        )
        _assert_same_values(
            _pooled(kernel, x), padded_window_max_reference(x), (dtype, policy)
        )
