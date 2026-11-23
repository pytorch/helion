"""GPU tests: reductions that are not fp32, or whose input is a narrower
conversion, return with their vector lanes folded per thread what they return
with one element per packet: float64 sums to the last bit where every partial
sum is representable, integers exactly.  Integer maxima and minima of negated
values, whose integer fold ptxas miscompiles, return what eager returns."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.integer_extremum_kernels import adversarial_rows
from helion._testing.integer_extremum_kernels import row_maxima_of_negated
from helion._testing.integer_extremum_kernels import row_minima_of_negated
from helion._testing.typed_fold_kernels import ONE_WARP
from helion._testing.typed_fold_kernels import SCALAR_LANES
from helion._testing.typed_fold_kernels import SPLIT_MOMENTS
from helion._testing.typed_fold_kernels import SPLIT_MOMENTS_SCALAR_LANES
from helion._testing.typed_fold_kernels import TWO_PACKETS
from helion._testing.typed_fold_kernels import TWO_WARPS
from helion._testing.typed_fold_kernels import TWO_WARPS_TWO_PACKETS
from helion._testing.typed_fold_kernels import row_dots_int32
from helion._testing.typed_fold_kernels import row_logsumexp
from helion._testing.typed_fold_kernels import row_logsumexp_of_doubled
from helion._testing.typed_fold_kernels import row_maxima
from helion._testing.typed_fold_kernels import row_minima
from helion._testing.typed_fold_kernels import row_square_sums
from helion._testing.typed_fold_kernels import row_sums
from helion._testing.typed_fold_kernels import row_sums_as_half
from helion._testing.typed_fold_kernels import split_column_moments
from helion._testing.typed_fold_kernels import split_column_moments_reference

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from helion.runtime.kernel import Kernel

_ROWS, _COLUMNS = 8, 256
_LAYOUTS = {
    "scalar_lanes": SCALAR_LANES,
    "one_warp": ONE_WARP,
    "two_packets": TWO_PACKETS,
    "two_warps": TWO_WARPS,
    "two_warps_two_packets": TWO_WARPS_TWO_PACKETS,
    "split_scalar_lanes": SPLIT_MOMENTS_SCALAR_LANES,
    "split_knobs": SPLIT_MOMENTS,
}


@functools.cache
def _with_layout(kernel: Kernel[Any], layout: str) -> Kernel[Any]:
    """``kernel`` pinned to a layout; compiled once per process."""
    return helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict(dict(_LAYOUTS[layout])),
    )


# One or two packets per thread, on one warp (a direct warp reduction) and on
# two (a grouped reduce); with two packets the lane fold combines the lanes.
_ALL = ("one_warp", "two_packets", "two_warps", "two_warps_two_packets")


def _integers(low: int, high: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randint(low, high, (_ROWS, _COLUMNS), device=DEVICE).to(dtype)


def _float64_squares() -> tuple[tuple[object, ...], torch.Tensor]:
    # Squares of up to 40 bits and row sums of up to 48: exact in float64,
    # not in fp32.
    x = _integers(-(2**20), 2**20, torch.float64)
    return (x,), (x * x).sum(-1)


def _int64_sums() -> tuple[tuple[object, ...], torch.Tensor]:
    x = _integers(-(2**40), 2**40, torch.int64)
    return (x,), x.sum(-1)


def _int32_dots() -> tuple[tuple[object, ...], torch.Tensor]:
    # Products of up to 22 bits and row sums of up to 30: within int32.
    x = _integers(-(2**11), 2**11, torch.int32)
    y = _integers(-(2**11), 2**11, torch.int32)
    return (x, y), (x * y).sum(-1, dtype=torch.int32)


def _int32_maxima() -> tuple[tuple[object, ...], torch.Tensor]:
    # Neighbours that fp32 cannot tell apart.
    x = _integers(2**30, 2**31 - 1, torch.int32)
    return (x, -(2**31)), x.amax(-1)


def _float64_maxima() -> tuple[tuple[object, ...], torch.Tensor]:
    x = 1.0 + _integers(0, 2**30, torch.float64) * 2.0**-40
    return (x, float("-inf")), x.amax(-1)


def _half_rounded_sums() -> tuple[tuple[object, ...], torch.Tensor]:
    # fp32 values just above the integers +-2..7: fp16 rounds each to its
    # integer, and the integers' sums are exact in fp16; the unrounded values
    # sum 0.1 higher, more than half an fp16 step of a sum below 256.
    magnitude = _integers(2, 8, torch.float32)
    sign = _integers(0, 2, torch.float32) * 2.0 - 1.0
    x = magnitude * sign + 0.0004
    return (x,), (magnitude * sign).sum(-1).to(torch.float16)


def _uint8_maxima() -> tuple[tuple[object, ...], torch.Tensor]:
    x = _integers(0, 200, torch.uint8)
    return (x, 0), x.amax(-1)


def _int16_minima() -> tuple[tuple[object, ...], torch.Tensor]:
    x = _integers(-30000, 30000, torch.int16)
    return (x, 32767), x.amin(-1)


@pytest.mark.parametrize(
    ("kernel", "make", "layouts"),
    [
        (row_square_sums, _float64_squares, _ALL),
        (row_sums, _int64_sums, _ALL),
        (row_dots_int32, _int32_dots, ("two_packets", "two_warps_two_packets")),
        (row_maxima, _int32_maxima, ("one_warp", "two_warps_two_packets")),
        (row_maxima, _float64_maxima, ("two_warps",)),
        # Small integers fold in an fp32 accumulator, which holds them, and
        # are reduced across the threads in their own dtype, unsigned and
        # narrower than a register.
        (row_maxima, _uint8_maxima, _ALL),
        # The one-warp minimum is a generic warp reduction, which is not folded.
        (row_minima, _int16_minima, ("two_warps", "two_warps_two_packets")),
        # An fp32 fold of fp32 values that the kernel rounds to fp16 first.
        (row_sums_as_half, _half_rounded_sums, ("one_warp", "two_packets")),
    ],
    ids=[
        "float64_square_sum",
        "int64_sum",
        "int32_dot",
        "int32_max",
        "float64_max",
        "uint8_max",
        "int16_min",
        "float32_rounded_to_half_sum",
    ],
)
def test_vector_lane_folds_return_what_scalar_lanes_return(
    kernel: Kernel[Any],
    make: Callable[[], tuple[tuple[object, ...], torch.Tensor]],
    layouts: tuple[str, ...],
) -> None:
    torch.manual_seed(0)
    args, expected = make()
    assert torch.equal(_with_layout(kernel, "scalar_lanes")(*args), expected)
    for layout in layouts:
        assert torch.equal(_with_layout(kernel, layout)(*args), expected), layout


# Ten cases, two axes.  The dtype decides the accumulator and its conversions:
# FP64 for int32, fp32 for int16, int8 and uint8 (whose negation wraps), an
# integer one for int64.  The op decides which reduces fold: a maximum folds
# behind a direct warp reduction and behind a grouped reduce, a minimum behind
# the grouped one only.  The int32 folds are the ones an integer accumulator
# got wrong: every layout that folds them, and the one-warp minimum, which
# keeps its loop.  The other dtypes run a direct and a grouped reduce each,
# the grouped maximum behind the lane fold.  Every case also runs with one
# element per packet, whose accumulate has no fold.
_OTHER = ("one_warp", "two_warps_two_packets")


@pytest.mark.parametrize(
    ("dtype", "kernel", "layouts"),
    [
        (torch.int32, row_maxima_of_negated, _ALL),
        (
            torch.int32,
            row_minima_of_negated,
            ("one_warp", "two_warps", "two_warps_two_packets"),
        ),
        (torch.int16, row_maxima_of_negated, _OTHER),
        (torch.int16, row_minima_of_negated, ("two_warps",)),
        (torch.int8, row_maxima_of_negated, _OTHER),
        (torch.int8, row_minima_of_negated, ("two_warps",)),
        (torch.uint8, row_maxima_of_negated, _OTHER),
        (torch.uint8, row_minima_of_negated, ("two_warps",)),
        (torch.int64, row_maxima_of_negated, _OTHER),
        (torch.int64, row_minima_of_negated, ("two_warps",)),
    ],
    ids=[
        "int32_max",
        "int32_min",
        "int16_max",
        "int16_min",
        "int8_max",
        "int8_min",
        "uint8_max",
        "uint8_min",
        "int64_max",
        "int64_min",
    ],
)
def test_integer_extrema_of_negated_values_equal_eager(
    dtype: torch.dtype, kernel: Kernel[Any], layouts: tuple[str, ...]
) -> None:
    """``torch.amax(-x)`` and ``torch.amin(-x)`` with vector lanes, against
    eager: ptxas loses one operand's negation in a chain of 32-bit integer
    maxima, so the fold holds such an extremum in a float.  The scalar-lane
    config, whose accumulate compares int32 and int16 through unsigned
    keys, returns the same."""
    x = adversarial_rows(dtype, DEVICE)
    info = torch.iinfo(dtype)
    if kernel is row_maxima_of_negated:
        start, expected = info.min, (-x).amax(-1)
    else:
        start, expected = info.max, (-x).amin(-1)
    for layout in ("scalar_lanes", *layouts):
        assert torch.equal(_with_layout(kernel, layout)(x, start), expected), layout


def test_an_fp8_maximum_is_exact_where_its_lanes_fold() -> None:
    """An fp8 maximum folds in fp32, which holds every fp8 value; across two
    warps with two packets per thread the lane fold combines the packets
    (that layout was rejected: its site key lost the fp8 identity).  The
    reference is the fp32 maximum of the same values: this kernel's
    scalar-lane and one-packet one-warp layouts do not get through NVVM."""
    torch.manual_seed(0)
    x = (torch.randn(_ROWS, _COLUMNS, device=DEVICE) * 4).to(torch.float8_e5m2)
    expected = x.to(torch.float32).amax(-1)
    for layout in ("two_packets", "two_warps", "two_warps_two_packets"):
        out = _with_layout(row_maxima, layout)(x, float("-inf"))
        assert torch.equal(out.to(torch.float32), expected), layout


@pytest.mark.parametrize(
    ("kernel", "accumulators", "atol"),
    [
        (row_logsumexp_of_doubled, torch.float32, 1e-3),
        (row_logsumexp, torch.bfloat16, 0.1),
    ],
    ids=["fused_pair", "two_folds"],
)
def test_half_precision_online_folds_are_close_to_float64(
    kernel: Kernel[Any], accumulators: torch.dtype, atol: float
) -> None:
    """The online log-sum-exp of a bf16 tensor with two packets per thread,
    where the half conversion kept in a fold decides the lane fold's form:
    fused with fp32 accumulators and ``2 * x``, two folds with bf16
    accumulators (a config the DSL rejected when the fold dropped the
    conversion and fused them).  Each is as close to the float64 result of
    the same inputs as its scalar-lane config is: within fp32 rounding, or
    within the bf16 rounding of each exponential and of the result."""
    torch.manual_seed(0)
    x = (torch.randn(_ROWS, _COLUMNS, device=DEVICE) * 1.5).to(torch.bfloat16)
    scale = 2 if kernel is row_logsumexp_of_doubled else 1
    expected = torch.logsumexp(scale * x.double(), dim=-1)
    for layout in ("scalar_lanes", "two_packets"):
        out = _with_layout(kernel, layout)(x, accumulators)
        torch.testing.assert_close(
            out.double(), expected, rtol=0.0, atol=atol, msg=layout
        )


def test_float64_sums_keep_float64_accuracy_on_random_values() -> None:
    """Sums that round: the folded lanes add in another order, within float64
    rounding of the exact sum (an fp32 accumulator is off by 1e-7 of it)."""
    torch.manual_seed(0)
    x = torch.randn(_ROWS, _COLUMNS, device=DEVICE, dtype=torch.float64)
    expected = (x * x).sum(-1)
    for layout in ("one_warp", "two_warps"):
        torch.testing.assert_close(
            _with_layout(row_square_sums, layout)(x), expected, rtol=1e-13, atol=0.0
        )


def test_split_float64_moments_are_exact_under_the_split_reduction_knobs() -> None:
    """Deferred tile reduce, tail peel and lane unroll on the split column
    moments: sums of squares of up to 40 bits are exact in float64."""
    torch.manual_seed(0)
    rows, rows_per_split = 8100, 1024
    x = torch.randint(-(2**15), 2**15, (rows, 64), device=DEVICE).to(torch.float32)
    expected = split_column_moments_reference(x, rows_per_split)
    for layout in ("split_scalar_lanes", "split_knobs"):
        actual = _with_layout(split_column_moments, layout)(x, rows_per_split)
        for got, reference in zip(actual, expected, strict=True):
            assert torch.equal(got, reference), layout
