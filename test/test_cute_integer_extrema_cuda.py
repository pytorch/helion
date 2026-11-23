"""GPU tests: the maxima and minima of negated integers that a CuTe reduction
accumulates one element after another equal eager's.  ptxas (CUDA 13.3)
loses one operand's negation when it fuses a chain of int32 or int16 extrema,
so their rows returned the extremum of a mix of negated and plain values; the
reduction now compares int32 and int16 through unsigned keys.  Int8 keeps its
signed chain, which is right as ptxas compiles it today: its cases are here
so that a toolchain that breaks it fails a test.  The extrema of one
expression are a chain as well: the maximum and the minimum of a register
region lost a negation the same way."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.integer_extremum_kernels import LOOPED
from helion._testing.integer_extremum_kernels import ONE_THREAD
from helion._testing.integer_extremum_kernels import ONE_WARP
from helion._testing.integer_extremum_kernels import SEQUENCE
from helion._testing.integer_extremum_kernels import SEQUENCE_FOUR_WARPS
from helion._testing.integer_extremum_kernels import TWO_WARPS
from helion._testing.integer_extremum_kernels import adversarial_rows
from helion._testing.integer_extremum_kernels import register_maxima_of_negated
from helion._testing.integer_extremum_kernels import register_minima_of_negated
from helion._testing.integer_extremum_kernels import row_argmin_of_negated
from helion._testing.integer_extremum_kernels import row_maxima_of_negated
from helion._testing.integer_extremum_kernels import row_minima_of_negated
from helion._testing.integer_extremum_kernels import rows_minus_maxima_of_negated
from helion._testing.integer_extremum_kernels import whole_row_maxima_of_negated
from helion._testing.integer_extremum_kernels import whole_row_minima_of_negated

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import Kernel

_LAYOUTS = {
    "one_warp": ONE_WARP,
    "two_warps": TWO_WARPS,
    "one_thread": ONE_THREAD,
    "looped": LOOPED,
    "sequence": SEQUENCE,
    "sequence_four_warps": SEQUENCE_FOUR_WARPS,
}


@functools.cache
def _with_layout(kernel: Kernel[Any], layout: str) -> Kernel[Any]:
    """``kernel`` pinned to a layout; compiled once per process and shape."""
    return helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict(dict(_LAYOUTS[layout])),
    )


# Ten cases, two axes.  The dtype decides the form: int32 and int16 each
# compare through keys of their own width (their rows were wrong), int8
# keeps its signed chain (right today, and watched here), uint8 and int64
# the plain extremum.  The maximum and the minimum are separate reductions.
# Each case runs four columns per thread on one warp and on two (which
# combine through shared memory), sixteen columns on one thread, and 250
# columns, whose last tile is masked.
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize(
    "dtype",
    [torch.int32, torch.int16, torch.int8, torch.uint8, torch.int64],
    ids=str,
)
def test_lane_accumulated_extrema_of_negated_integers_equal_eager(
    dtype: torch.dtype, op: str
) -> None:
    x = adversarial_rows(dtype, DEVICE)
    info = torch.iinfo(dtype)
    if op == "max":
        kernel, start, reduce = row_maxima_of_negated, info.min, torch.amax
    else:
        kernel, start, reduce = row_minima_of_negated, info.max, torch.amin
    expected = reduce(-x, dim=-1)
    for layout in ("one_warp", "two_warps", "one_thread"):
        assert torch.equal(_with_layout(kernel, layout)(x, start), expected), layout
    masked = x[:, :250].contiguous()
    assert torch.equal(
        _with_layout(kernel, "one_warp")(masked, start), reduce(-masked, dim=-1)
    )


@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.int8], ids=str)
def test_loop_accumulated_extrema_of_negated_integers_equal_eager(
    dtype: torch.dtype, op: str
) -> None:
    """A reduction loop's accumulate: int32 compared through keys already,
    int16 rows were wrong, int8 keeps its signed chain."""
    x = adversarial_rows(dtype, DEVICE)
    if op == "max":
        kernel, reduce = whole_row_maxima_of_negated, torch.amax
    else:
        kernel, reduce = whole_row_minima_of_negated, torch.amin
    assert torch.equal(_with_layout(kernel, "looped")(x), reduce(-x, dim=-1))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.int8], ids=str)
def test_sequence_accumulated_maxima_of_negated_integers_equal_eager(
    dtype: torch.dtype,
) -> None:
    """The ordered lowering of a tile's reductions accumulates eight columns
    per thread: int32 and int16 rows were wrong there too, int8 keeps its
    signed chain."""
    x = adversarial_rows(dtype, DEVICE)
    top = torch.amax(-x, dim=-1).to(torch.float32)
    expected = x.to(torch.float32) - top[:, None]
    out = _with_layout(rows_minus_maxima_of_negated, "sequence")(x)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.int8], ids=str)
def test_an_argmin_of_negated_integers_equals_eager(dtype: torch.dtype) -> None:
    """An argmin's lanes accumulate the minimum of its values first: with an
    int32 or int16 minimum that lost a negation, no element or the wrong one
    held it."""
    x = adversarial_rows(dtype, DEVICE)
    highest = torch.iinfo(dtype).max
    expected = torch.argmin(-x, dim=-1)
    for layout in ("one_warp", "one_thread"):
        out = _with_layout(row_argmin_of_negated, layout)(x, highest)
        assert torch.equal(out, expected), layout


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint32, torch.int32], ids=str)
def test_sequence_maxima_combine_across_four_warps(dtype: torch.dtype) -> None:
    """The ordered lowering on four warps combines the warps' partials from
    shared memory: an unsigned maximum did not compile there (its partials
    read back signed), and an int32 one accumulates through keys first.
    torch negates no uint32, so every reference is taken in int64 with the
    dtype's wrap-around."""
    x = adversarial_rows(dtype, DEVICE)
    info = torch.iinfo(dtype)
    span = info.max - info.min + 1
    negated = (-x.to(torch.int64) - info.min) % span + info.min
    top = negated.amax(dim=-1).to(torch.float32)
    expected = x.to(torch.float32) - top[:, None]
    out = _with_layout(rows_minus_maxima_of_negated, "sequence_four_warps")(x)
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    ("kernel", "extremum"),
    [
        (register_maxima_of_negated, torch.maximum),
        (register_minima_of_negated, torch.minimum),
    ],
    ids=["maximum", "minimum"],
)
def test_register_extrema_of_negated_int32_equal_eager(
    kernel: Kernel[Any], extremum: Any
) -> None:
    """The extrema of three negated rows of a register region (16 rows of
    four groups of eight elements; the second operand's groups are swapped
    in pairs) are those of a register program: an int32 maximum or minimum
    lost a negation there too."""
    x = adversarial_rows(torch.int32, DEVICE)
    a, b, c = (
        x[:, 32 * part : 32 * part + 32].reshape(16, 4, 8).contiguous()
        for part in range(3)
    )
    peer = b[:, torch.arange(4, device=DEVICE) ^ 1, :]
    expected = extremum(extremum(-a, -peer), -c)
    assert torch.equal(kernel(a, b, c), expected)
