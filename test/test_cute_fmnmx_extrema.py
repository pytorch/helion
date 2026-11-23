"""GPU-free coverage of ``cute_fmnmx_extrema``: an fp32 ``torch.maximum`` /
``torch.minimum`` as the single NaN-propagating instruction of the extremum
reductions (``handler_ops._nan_extremum``), and of the unsigned keys that
lowering takes the Int32 and Int16 extrema of a chain over."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion import exc
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.guarded_lane_kernels import IDENTITY_CHAINS
from helion._testing.guarded_lane_kernels import LITERAL_IDENTITY_CHAINS
from helion._testing.guarded_lane_kernels import extrema
from helion._testing.guarded_lane_kernels import extrema_bound_chains
from helion._testing.guarded_lane_kernels import extrema_carried_max
from helion._testing.guarded_lane_kernels import extrema_carried_max_copies
from helion._testing.guarded_lane_kernels import extrema_carried_max_identities
from helion._testing.guarded_lane_kernels import extrema_chains
from helion._testing.guarded_lane_kernels import extrema_identity_chains
from helion._testing.guarded_lane_kernels import extrema_literal_identity_chains
from helion._testing.guarded_lane_kernels import extrema_outside_chains
from helion._testing.guarded_lane_kernels import extrema_relu_clamp
from helion._testing.guarded_lane_kernels import extrema_view_chains
from helion._testing.guarded_lane_kernels import window_max
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _relu_only(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.relu(x[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_extrema(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.maximum(x[tile], y[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_with_the_previous_column(x: torch.Tensor) -> torch.Tensor:
    # A serial loop hands the previous column's value on, not the maximum.
    out = torch.empty_like(x)
    for tile_r in hl.tile(x.size(0)):
        previous = hl.zeros([tile_r], dtype=x.dtype)
        for column in hl.grid(x.size(1)):
            value = -x[tile_r, column]
            out[tile_r, column] = torch.maximum(previous, value)
            previous = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_a_merge(
    x: torch.Tensor, flag: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A maximum, then a loop or a branch that adds to its variable, then a
    # maximum of the sum, or of a plain copy of the sum: the loop and the
    # branch assign the variable of the first maximum again.
    rows, columns = x.shape
    looped = torch.empty([rows], dtype=x.dtype, device=x.device)
    copied = torch.empty([rows], dtype=x.dtype, device=x.device)
    branched = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        first = x[tile_r, 0]
        total = torch.maximum(-first, -(first * 2))
        for column in hl.grid(columns):
            total = total + x[tile_r, column]
        looped[tile_r] = torch.maximum(total, -(first + 1))
        kept = total
        copied[tile_r] = torch.maximum(kept, -(first + 2))
        other = torch.maximum(-first, -(first * 3))
        if flag > 0:
            other = other + 1
        branched[tile_r] = torch.maximum(other, -(first - 1))
    return looped, copied, branched


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_of_a_copied_value(x: torch.Tensor) -> torch.Tensor:
    # A serial loop hands on the maximum of a plain copy of a value computed
    # in its body: the loop carries the maximum, but none goes into it.
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        last = hl.zeros([tile_r], dtype=x.dtype)
        for column in hl.grid(columns):
            value = -x[tile_r, column]
            kept = value
            last = torch.maximum(kept, value * 2)
        out[tile_r] = last
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_max_plus_one(x: torch.Tensor, lowest: hl.constexpr) -> torch.Tensor:
    # A serial loop hands on the maximum plus one: a computed value.
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best, -x[tile_r, column]) + 1
        out[tile_r] = best
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _extrema_apart(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # A view between a maximum and a minimum, and a sum between two maxima.
    mixed = torch.empty_like(x)
    summed = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        inner = torch.maximum(-value, -(value * 2))
        mixed[tile_r, tile_c] = torch.minimum(inner[:, :], -(value + 1))
        summed[tile_r, tile_c] = torch.maximum(inner + 1, -(value + 1))
    return mixed, summed


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_a_computation(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # Between two maxima, operations that look like the value identities and
    # compute another value: with another literal (``* 2``, ``// 2``,
    # ``| 1``, ``<< 1``, ``** 2``), one negation, a conversion to a narrower
    # integer and back, and a ``where`` with two values.
    doubled = torch.empty_like(x)
    halved = torch.empty_like(x)
    ored = torch.empty_like(x)
    shifted = torch.empty_like(x)
    squared = torch.empty_like(x)
    negated = torch.empty_like(x)
    narrowed = torch.empty_like(x)
    selected = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        last = -(value + 1)
        inner = torch.maximum(-value, -(value * 2))
        doubled[tile_r, tile_c] = torch.maximum(inner * 2, last)
        inner = torch.maximum(-value, -(value * 3))
        halved[tile_r, tile_c] = torch.maximum(inner // 2, last)
        inner = torch.maximum(-value, -(value * 4))
        ored[tile_r, tile_c] = torch.maximum(inner | 1, last)
        inner = torch.maximum(-value, -(value * 5))
        shifted[tile_r, tile_c] = torch.maximum(inner << 1, last)
        inner = torch.maximum(-value, -(value * 6))
        squared[tile_r, tile_c] = torch.maximum(inner**2, last)
        inner = torch.maximum(-value, -(value * 7))
        negated[tile_r, tile_c] = torch.maximum(-inner, last)
        inner = torch.maximum(-value, -(value * 8))
        narrowed[tile_r, tile_c] = torch.maximum(inner.to(torch.int8).to(x.dtype), last)
        inner = torch.maximum(-value, -(value * 9))
        selected[tile_r, tile_c] = torch.maximum(
            torch.where(value > 0, inner, value), last
        )
    return doubled, halved, ored, shifted, squared, negated, narrowed, selected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_a_copied_negation(x: torch.Tensor) -> torch.Tensor:
    # A plain copy of the variable between the two negations.
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        inner = torch.maximum(-value, -(value * 2))
        once = torch.neg(inner)
        kept = once
        out[tile_r, tile_c] = torch.maximum(torch.neg(kept), -(value + 1))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_an_identity_inside_a_pair(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Another value identity between the two negations (a sum with zero, a
    # rounding) and between the two conversions (a sum with zero).
    summed = torch.empty_like(x)
    rounded = torch.empty_like(x)
    widened = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        last = -(value + 1)
        inner = torch.maximum(-value, -(value * 2))
        summed[tile_r, tile_c] = torch.maximum(torch.neg(torch.neg(inner) + 0), last)
        inner = torch.maximum(-value, -(value * 3))
        rounded[tile_r, tile_c] = torch.maximum(
            torch.neg(torch.round(torch.neg(inner))), last
        )
        inner = torch.maximum(-value, -(value * 4))
        widened[tile_r, tile_c] = torch.maximum(
            (inner.to(torch.int64) + 0).to(x.dtype), last
        )
    return summed, rounded, widened


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_half_an_identity(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Copies and views around what is no negation of a negation and no
    # conversion to a wider integer and back: one negation and a copy; a
    # sum of one between two negations; a narrower integer between two
    # conversions; a sum of one between two conversions.
    once = torch.empty_like(x)
    summed = torch.empty_like(x)
    narrowed = torch.empty_like(x)
    widened = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        last = -(value + 1)
        inner = torch.maximum(-value, -(value * 2))
        once[tile_r, tile_c] = torch.maximum(torch.neg(inner).clone(), last)
        inner = torch.maximum(-value, -(value * 3))
        summed[tile_r, tile_c] = torch.maximum(
            torch.neg(torch.neg(inner)[:, :] + 1), last
        )
        inner = torch.maximum(-value, -(value * 4))
        narrowed[tile_r, tile_c] = torch.maximum(
            inner.to(torch.int8).clone().to(x.dtype), last
        )
        inner = torch.maximum(-value, -(value * 5))
        widened[tile_r, tile_c] = torch.maximum(
            (inner.to(torch.int64) + 1).to(x.dtype), last
        )
    return once, summed, narrowed, widened


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_a_narrowing_conversion(x: torch.Tensor) -> torch.Tensor:
    # An Int32 maximum of widened int16 values, converted to int64 and then
    # to int16, under an Int16 maximum: the second conversion narrows the
    # maximum's own dtype.
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        wide = value.to(torch.int32)
        inner = torch.maximum(-wide, -(wide * 2))
        out[tile_r, tile_c] = torch.maximum(
            inner.to(torch.int64).to(torch.int16), -(value + 1)
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_a_constexpr_zero(
    x: torch.Tensor, bias: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        inner = torch.maximum(-value, -(value * 2))
        out[tile_r, tile_c] = torch.maximum(inner + bias, -(value + 1))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chains_through_other_copies_and_views(
    x: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    # Chains through the copies and views that extrema_view_chains does not
    # spell: a view to the same shape; the three squeeze overloads (of a dim,
    # of every dim, of a tuple of dims); a detach; a fresh copy; a plain copy
    # of the variable; and the two that are copies by their arguments, a
    # repeat of one along every dim (also of a plain copy) and a flip of no
    # dim (here the second operand).
    reshaped = torch.empty_like(x)
    squeezed = torch.empty_like(x)
    detached = torch.empty_like(x)
    lifted = torch.empty_like(x)
    renamed = torch.empty_like(x)
    repeated = torch.empty_like(x)
    flipped = torch.empty_like(x)
    copied = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        inner = torch.maximum(-value, -(value * 2))
        reshaped[tile_r, tile_c] = torch.maximum(inner.view(inner.shape), -(value + 1))
        inner = torch.minimum(-value, -(value * 2))
        squeezed[tile_r, tile_c] = torch.minimum(
            inner.unsqueeze(0).squeeze(0).squeeze().squeeze((0,)), -(value + 1)
        )
        inner = torch.maximum(-value, -(value * 3))
        detached[tile_r, tile_c] = torch.maximum(inner.detach(), -(value - 1))
        inner = torch.minimum(-value, -(value * 3))
        lifted[tile_r, tile_c] = torch.minimum(
            torch.ops.aten.lift_fresh_copy(inner), -(value - 1)
        )
        inner = torch.maximum(-value, -(value * 5))
        kept = inner
        renamed[tile_r, tile_c] = torch.maximum(kept, -(value - 2))
        inner = torch.minimum(-value, -(value * 5))
        repeated[tile_r, tile_c] = torch.minimum(inner.repeat(1, 1), -(value - 2))
        inner = torch.maximum(-value, -(value * 7))
        flipped[tile_r, tile_c] = torch.maximum(-(value - 3), inner.flip(()))
        inner = torch.minimum(-value, -(value * 7))
        kept = inner
        copied[tile_r, tile_c] = torch.minimum(kept.repeat(1, 1), -(value - 3))
    return reshaped, squeezed, detached, lifted, renamed, repeated, flipped, copied


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chains_without_a_statement_between(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Two extrema of one kind with an operation between them that is no
    # copy, view or value identity by its kind and its literal arguments,
    # but whose lowering returns its operand for this shape: a select on a
    # dim of size one, as the first operand, as the second, and of a plain
    # copy of the variable.
    selected = torch.empty_like(x)
    second = torch.empty_like(x)
    copied = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        inner = torch.maximum(-value, -(value * 2))
        selected[tile_r, tile_c] = torch.maximum(
            inner.unsqueeze(0).select(0, 0), -(value + 1)
        )
        inner = torch.minimum(-value, -(value * 2))
        second[tile_r, tile_c] = torch.minimum(
            -(value + 1), inner.unsqueeze(0).select(0, 0)
        )
        inner = torch.minimum(-value, -(value * 3))
        kept = inner
        copied[tile_r, tile_c] = torch.minimum(
            kept.unsqueeze(0).select(0, 0), -(value - 1)
        )
    return selected, second, copied


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_max_through_views(x: torch.Tensor, lowest: hl.constexpr) -> torch.Tensor:
    # extrema_carried_max with every copy and view of ``_SAME_ELEMENTS`` that
    # a kernel reaches between the value the body takes in and the maximum,
    # and a plain copy between the maximum and the value it hands back.
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            taken = best[:].unsqueeze(0).squeeze(0).squeeze().squeeze((0,))
            taken = taken.view(best.shape).expand(best.shape).permute(0).detach()
            taken = torch.ops.aten.lift_fresh_copy(taken.clone())
            larger = torch.maximum(taken, -x[tile_r, column])
            best = larger
        out[tile_r] = best
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _maxima_around_a_narrowing_subscript(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
        value = x[tile_r, tile_c]
        inner = torch.maximum(-value, -(value * 2))
        out[tile_r, tile_c] = torch.maximum(inner[:, 0:1], -(value + 1))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_extrema_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # An fp32 maximum of two tensors, but in a kernel with a reduction.
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(x.size(0)):
        out[tile_r] = torch.maximum(x[tile_r, :], y[tile_r, :]).sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_extrema(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Maxima of fp32, int32 and bf16 operands, each in its own dtype.
    single = torch.empty_like(a)
    integer = torch.empty_like(c)
    half = torch.empty_like(e)
    for tile in hl.tile(a.size(0)):
        single[tile] = torch.maximum(a[tile], b[tile])
        integer[tile] = torch.maximum(c[tile], d[tile])
        half[tile] = torch.maximum(e[tile], f[tile])
    return single, integer, half


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_extrema(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.maximum(x[tile], y[tile]) * 1.5 + 0.25
    return out


def _code(kernel: object, args: tuple[torch.Tensor, ...], **config: object) -> str:
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
        return bound.to_code(helion.Config.from_dict({"block_sizes": [32], **config}))


def test_fp32_extrema_take_the_single_instruction_form() -> None:
    args = (torch.zeros(64), torch.zeros(64))
    before = _code(extrema, args)
    assert "cute.math.max(" in before and "cute.math.min(" in before
    assert "cute.arch.fmax(" not in before and "cute.arch.fmin(" not in before
    after = _code(extrema, args, cute_fmnmx_extrema=True)
    assert "cute.arch.fmax(" in after and "cute.arch.fmin(" in after
    assert after.count("nan=True") == 2
    assert "cute.math.max(" not in after and "cute.math.min(" not in after


def test_widened_half_extrema_take_the_single_instruction_form() -> None:
    """The maxima of a bf16 window are computed in fp32, on widened lanes:
    the form the assembler folds into packed half-precision maxima."""
    x = torch.zeros([18, 14, 16], dtype=torch.bfloat16)
    config = {
        "block_sizes": [4, 4, 16],
        "num_threads": [4, 4, 2],
        "cute_vector_widths": [1, 1, 8],
    }
    with mock_cuda_unavailable():
        bound = cpu_bind(window_max, (x,))
        before = bound.to_code(helion.Config.from_dict(config))
        after = bound.to_code(
            helion.Config.from_dict({**config, "cute_fmnmx_extrema": True})
        )
    assert before.count("cute.math.max(") == 3 and "cute.arch.fmax(" not in before
    assert after.count("cute.arch.fmax(") == 3 and "cute.math.max(" not in after


def test_only_fp32_extrema_take_the_single_instruction_form() -> None:
    """``cute.arch.fmax`` computes in fp32: an integer or a half-precision
    maximum beside an fp32 one keeps the form of its own dtype."""
    single = torch.zeros(64)
    integer = torch.zeros(64, dtype=torch.int32)
    half = torch.zeros(64, dtype=torch.bfloat16)
    args = (single, single, integer, integer, half, half)
    before = _code(_mixed_extrema, args)
    assert before.count("cute.math.max(") == 3 and "cute.arch.fmax(" not in before
    after = _code(_mixed_extrema, args, cute_fmnmx_extrema=True)
    assert after.count("cute.arch.fmax(") == 1
    assert after.count("cute.math.max(") == 2


# The CuTe dtypes whose chains take keys, each with its unsigned dtype and
# sign bit, and the torch dtypes of every integer case below.
_SIGNED_KEYS = {"Int32": ("Uint32", 2147483648), "Int16": ("Uint16", 32768)}
_INTEGERS = {
    "Int32": torch.int32,
    "Int16": torch.int16,
    "Int8": torch.int8,
    "Uint8": torch.uint8,
    "Int64": torch.int64,
}


def _assert_extrema(code: str, ops: str, dtype: str, keyed: bool | str) -> None:
    """The statements of ``code`` that take an extremum are, in order, the
    maxima (``M``) and minima (``m``) of ``ops`` over two names cast to
    ``dtype``: over unsigned keys when ``keyed``, else plain.  ``keyed`` may
    say which form each statement has, in order (``k`` keyed, ``p`` plain)."""
    statements = re.findall(
        r"^\s*v_\d+ = (.*cute\.math\.m(?:ax|in)\(.*)$", code, re.MULTILINE
    )
    assert len(statements) == len(ops), statements
    if isinstance(keyed, bool):
        keyed = ("k" if keyed else "p") * len(ops)
    value = rf"cutlass\.{dtype}\(\w+\)"
    for statement, letter, form in zip(statements, ops, keyed, strict=True):
        op = "max" if letter == "M" else "min"
        if form == "k":
            unsigned, sign = _SIGNED_KEYS[dtype]
            bit = rf"cutlass\.{unsigned}\({sign}\)"
            key = rf"cutlass\.{unsigned}\({value}\) \^ {bit}"
            form = rf"cutlass\.{dtype}\(cute\.math\.{op}\({key}, {key}\) \^ {bit}\)"
        else:
            form = rf"cute\.math\.{op}\({value}, {value}, propagate_nan=True\)"
        assert re.fullmatch(form, statement), statement


@pytest.mark.parametrize("dtype", sorted(_SIGNED_KEYS))
def test_the_links_of_an_int32_or_int16_chain_compare_unsigned_keys(
    dtype: str,
) -> None:
    """ptxas loses the negation of an operand when it fuses two signed
    Int32 or Int16 extrema of one kind: every node of such a chain flips
    the sign bit of both operands, compares them as unsigned values and
    flips the result back.  Two nested maxima, two nested minima, three
    minima through a value a static loop renames and two maxima through a
    named value are nine links."""
    x = torch.zeros([16, 256], dtype=_INTEGERS[dtype])
    code = _code(extrema_chains, (x, x, x, x), block_sizes=[4, 64])
    _assert_extrema(code, "MMmmmmmMM", dtype, keyed=True)


@pytest.mark.parametrize("dtype", sorted(_SIGNED_KEYS))
def test_extrema_outside_a_chain_keep_the_plain_form(dtype: str) -> None:
    """A single maximum, a ``clamp`` between constants and between tensors
    (a maximum under a minimum) and a maximum of a minimum are no chain of
    one kind: their text, which other lowerings read, is unchanged."""
    x = torch.zeros([16, 256], dtype=_INTEGERS[dtype])
    code = _code(extrema_outside_chains, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MMmMmmM", dtype, keyed=False)
    assert "Uint" not in code
    # A maximum with a value its loop carries, but which the loop does not
    # carry on, does not chain across the trips.
    code = _code(_maxima_with_the_previous_column, (x,), block_sizes=[4])
    _assert_extrema(code, "M", dtype, keyed=False)
    # ... nor does one whose sum with one the loop carries on.
    lowest = torch.iinfo(x.dtype).min
    code = _code(_carried_max_plus_one, (x, lowest), block_sizes=[4])
    _assert_extrema(code, "M", dtype, keyed=False)
    # A loop or a branch that adds to the variable of a maximum assigns it
    # again: a maximum of that variable afterwards reads a sum.
    code = _code(_maxima_around_a_merge, (x, 1), block_sizes=[4])
    _assert_extrema(code, "MMMMM", dtype, keyed=False)
    # A view does not make a maximum and a minimum a chain, and a sum between
    # two maxima is a computed value, which ends one.
    code = _code(_extrema_apart, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MmM", dtype, keyed=False)


@pytest.mark.parametrize("dtype", sorted(_SIGNED_KEYS))
def test_lower_clamp_bounds_a_relu_and_a_carried_maximum_are_links(
    dtype: str,
) -> None:
    """A lower ``clamp`` bound and a ``relu`` are maxima: one over another,
    or over a ``torch.maximum``, chains.  So does a maximum that accumulates
    into a value its loop hands from trip to trip."""
    x = torch.zeros([16, 256], dtype=_INTEGERS[dtype])
    code = _code(extrema_bound_chains, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MMMM", dtype, keyed=True)
    lowest = torch.iinfo(x.dtype).min
    code = _code(extrema_carried_max, (x, lowest), block_sizes=[4])
    _assert_extrema(code, "M", dtype, keyed=True)
    # ... also through a copy of the value the body takes in or hands back
    # (a clone, a repeat of one, a flip of no dim), through every copy and
    # view of the set on the way in and a plain copy on the way out.  The
    # sixth loop of that kernel carries nothing: its two maxima are links
    # through a repeat of one and a plain copy of the variable.  The later
    # one of the last loop reads a copy of a sum, and is keyed because the
    # loop hands it on: a copy of a value computed in the body counts as
    # well as one of a value the body takes in.  That sum is the earlier
    # maximum plus a difference that is zero, which is no value identity by
    # its kind, so the key is needed there, and counting only the values
    # the body takes in would take it away.
    code = _code(extrema_carried_max_copies, (x, lowest), block_sizes=[4])
    _assert_extrema(code, "MMMMMMMMM", dtype, keyed="kkkkkkkpk")
    code = _code(_carried_max_through_views, (x, lowest), block_sizes=[4])
    _assert_extrema(code, "M", dtype, keyed=True)
    # ... and so does the copy of a value that is no maximum at all.
    code = _code(_maxima_of_a_copied_value, (x,), block_sizes=[4])
    _assert_extrema(code, "M", dtype, keyed=True)


@pytest.mark.parametrize("dtype", sorted(_SIGNED_KEYS))
def test_links_that_copies_and_views_separate_compare_unsigned_keys(
    dtype: str,
) -> None:
    """A copy or a view computes nothing, so two extrema of one kind with
    only those between them (``[:, None]``, ``unsqueeze`` and ``expand``, a
    full slice, ``permute``, ``clone``; ``view``, the ``squeeze`` overloads,
    ``detach``, a fresh copy, ``kept = inner``, a ``repeat`` of one along
    every dim, a flip of no dim) are fused like adjacent ones: both links
    are keyed.  (The last chain of ``extrema_view_chains`` is through
    ``** 1``, a value identity: see the next test.)"""
    inputs = _view_chain_inputs(dtype)
    code = _code(extrema_view_chains, inputs, block_sizes=[4, 64])
    _assert_extrema(code, "MMmmMMmmmmMM", dtype, keyed=True)
    code = _code(
        _chains_through_other_copies_and_views, inputs[2:3], block_sizes=[4, 64]
    )
    _assert_extrema(code, "MMmmMMmmMMmmMMmm", dtype, keyed=True)


def _chain_ops(chains: tuple[tuple[str, str], ...]) -> str:
    """The extremum statements of an identity chain kernel, two per chain."""
    return "".join("MM" if kind == "max" else "mm" for _, kind in chains)


@pytest.mark.parametrize("dtype", sorted(_SIGNED_KEYS))
def test_links_that_value_identities_separate_compare_unsigned_keys(
    dtype: str,
) -> None:
    """A value identity leaves an integer as it is and the optimizer removes
    its statement, so two extrema of one kind with only such operations
    between them are fused like adjacent ones: both links are keyed.  The
    identities are an operation with the literal that is neutral for it
    (``+ 0``, ``- 0``, ``* 1``, ``// 1``, ``| 0``, ``^ 0``, ``& -1``,
    ``<< 0``, ``>> 0``; ``** 1`` is in the test above), the four roundings,
    ``torch.where`` with one value for both branches, the negation of a
    negation and a conversion to a wider integer and back, the last two
    also with a view, a copy or another identity between their two
    operations."""
    x = torch.zeros([16, 256], dtype=_INTEGERS[dtype])
    code = _code(extrema_literal_identity_chains, (x,), block_sizes=[4, 64])
    _assert_extrema(code, _chain_ops(LITERAL_IDENTITY_CHAINS), dtype, keyed=True)
    code = _code(extrema_identity_chains, (x,), block_sizes=[4, 64])
    _assert_extrema(code, _chain_ops(IDENTITY_CHAINS), dtype, keyed=True)
    # ... and a carried maximum with one at the loop's boundary: on the
    # value the body takes in, and on the one it hands back.
    lowest = torch.iinfo(x.dtype).min
    code = _code(extrema_carried_max_identities, (x, lowest), block_sizes=[4])
    _assert_extrema(code, "MM", dtype, keyed=True)
    # ... and with a plain copy of the variable between two negations.
    code = _code(_maxima_around_a_copied_negation, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MM", dtype, keyed=True)
    # ... and with another identity between the two operations of a pair.
    code = _code(_maxima_around_an_identity_inside_a_pair, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MM" * 3, dtype, keyed=True)
    # A copy or a view next to half of such a pair, or a computation
    # between its two operations, joins nothing.
    code = _code(_maxima_around_half_an_identity, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MM" * 4, dtype, keyed=False)
    # An operation of the same kind that computes another value ends the
    # chain: another literal, one negation, a narrower integer, a ``where``
    # with two values.
    code = _code(_maxima_around_a_computation, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MM" * 8, dtype, keyed=False)
    # The literal may come from a constexpr argument: zero is the identity,
    # one is not.
    code = _code(_maxima_around_a_constexpr_zero, (x, 0), block_sizes=[4, 64])
    _assert_extrema(code, "MM", dtype, keyed=True)
    code = _code(_maxima_around_a_constexpr_zero, (x, 1), block_sizes=[4, 64])
    _assert_extrema(code, "MM", dtype, keyed=False)


def test_a_conversion_that_narrows_the_value_is_no_identity() -> None:
    """A conversion to a wider integer and back gives the elements back
    only when it ends at the dtype it started from: an Int32 maximum that
    is converted to int64 and then to int16 is another value, and neither
    it nor the Int16 maximum over it is keyed."""
    x = torch.zeros([16, 256], dtype=torch.int16)
    code = _code(_maxima_around_a_narrowing_conversion, (x,), block_sizes=[4, 64])
    assert len(re.findall(r"cute\.math\.max\(", code)) == 2
    assert "Uint" not in code


@pytest.mark.parametrize("dtype", sorted(_SIGNED_KEYS))
def test_an_extremum_that_reads_the_variable_of_another_is_keyed(dtype: str) -> None:
    """An operation that is no copy, view or value identity by its kind and
    its literal arguments may still emit no statement (a select on a dim of
    size one: its lowering returns its operand for that shape).  The later
    extremum then reads the variable the earlier one assigned, or a plain
    copy of it, the generated code is a chain, and the later link is keyed:
    the earlier one is not known to be a link when it is emitted, and keys
    on one of two extrema keep them apart."""
    x = torch.zeros([16, 256], dtype=_INTEGERS[dtype])
    code = _code(_chains_without_a_statement_between, (x,), block_sizes=[4, 64])
    _assert_extrema(code, "MMmmmm", dtype, keyed="pk" * 3)


def test_a_subscript_that_narrows_a_computed_value_is_refused() -> None:
    """The chain rule follows a ``subscript`` whatever its index: the
    backend admits a subscript of a computed value for ``None`` and full
    slices only, so none that selects part of a link's result reaches it."""
    x = torch.zeros([16, 256], dtype=torch.int32)
    with pytest.raises(
        exc.BackendUnsupported, match="narrowing kernel-tensor subscripts"
    ):
        _code(_maxima_around_a_narrowing_subscript, (x,), block_sizes=[4, 64])


def _view_chain_inputs(dtype: str) -> tuple[torch.Tensor, ...]:
    kind = _INTEGERS[dtype]
    return (
        torch.zeros([16], dtype=kind),
        torch.zeros([256], dtype=kind),
        torch.zeros([16, 256], dtype=kind),
        torch.zeros([256, 16], dtype=kind),
    )


@pytest.mark.parametrize("dtype", ["Int8", "Uint8", "Int64"])
def test_int8_unsigned_and_64_bit_chains_keep_the_plain_form(dtype: str) -> None:
    """ptxas compiles an Int8 chain right (each negation is sign-extended
    first), keeps every negation of an unsigned one and takes a 64-bit
    extremum by a compare and a select: none has keys."""
    x = torch.zeros([16, 256], dtype=_INTEGERS[dtype])
    code = _code(extrema_chains, (x, x, x, x), block_sizes=[4, 64])
    _assert_extrema(code, "MMmmmmmMM", dtype, keyed=False)
    lowest = torch.iinfo(x.dtype).min
    code = _code(extrema_carried_max, (x, lowest), block_sizes=[4])
    _assert_extrema(code, "M", dtype, keyed=False)
    code = _code(extrema_view_chains, _view_chain_inputs(dtype), block_sizes=[4, 64])
    _assert_extrema(code, "MMmmMMmmmmMM", dtype, keyed=False)


def test_relu_and_clamp_beside_an_extremum_take_the_form_too() -> None:
    """A ``relu`` and a ``clamp`` lower through the handler of
    ``torch.maximum`` / ``minimum``: in a kernel that sets the knob they are
    the single instructions as well (one maximum each for the maximum, the
    relu and the clamp's lower bound, one minimum for its upper bound)."""
    args = (torch.zeros(64), torch.zeros(64))
    before = _code(extrema_relu_clamp, args)
    assert before.count("cute.math.max(") == 3 and before.count("cute.math.min(") == 1
    assert "cute.arch.fmax(" not in before and "cute.arch.fmin(" not in before
    after = _code(extrema_relu_clamp, args, cute_fmnmx_extrema=True)
    assert after.count("cute.arch.fmax(") == 3 and after.count("cute.arch.fmin(") == 1
    assert after.count("nan=True") == 4
    assert "cute.math.max(" not in after and "cute.math.min(" not in after


def test_product_next_to_an_extremum_still_contracts() -> None:
    """``fuse_fma`` types the single-instruction extremum as the fp32 value
    it is, so ``maximum(x, y) * 1.5 + 0.25`` is one fma under the knob as it
    is without it."""
    args = (torch.zeros(64),) * 2
    for knob in (False, True):
        code = _code(_scaled_extrema, args, cute_fmnmx_extrema=knob)
        assert code.count("cute.math.fma(") == 1, knob
        assert ("cute.arch.fmax(" in code) is knob


def test_knob_is_offered_on_fp32_extrema_of_two_tensors_only() -> None:
    with mock_cuda_unavailable():
        spec = cpu_bind(extrema, (torch.zeros(64), torch.zeros(64))).config_spec
        assert "cute_fmnmx_extrema" in spec._flat_fields()
        config = spec.default_config().config | {"cute_fmnmx_extrema": True}
        spec.normalize(config)
        assert config["cute_fmnmx_extrema"] is True
        # A relu is one instruction either way, an integer has no NaN, and
        # the knob is a pointwise kernel's (no reduction, matmul or atomic).
        integers = torch.zeros(64, dtype=torch.int32)
        rows = torch.zeros([9, 16])
        for kernel, args in (
            (_relu_only, (torch.zeros(64),)),
            (_integer_extrema, (integers, integers)),
            (_row_extrema_sums, (rows, rows)),
        ):
            spec = cpu_bind(kernel, args).config_spec
            assert "cute_fmnmx_extrema" not in spec._flat_fields()
            with pytest.raises(exc.InvalidConfig, match="cute_fmnmx_extrema requires"):
                spec.normalize(
                    spec.default_config().config | {"cute_fmnmx_extrema": True}
                )
            config = spec.default_config().config | {"cute_fmnmx_extrema": False}
            spec.normalize(config)
            assert "cute_fmnmx_extrema" not in config
