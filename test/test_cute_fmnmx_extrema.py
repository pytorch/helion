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
from helion._testing.guarded_lane_kernels import extrema
from helion._testing.guarded_lane_kernels import extrema_bound_chains
from helion._testing.guarded_lane_kernels import extrema_carried_max
from helion._testing.guarded_lane_kernels import extrema_chains
from helion._testing.guarded_lane_kernels import extrema_outside_chains
from helion._testing.guarded_lane_kernels import extrema_relu_clamp
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


def _assert_extrema(code: str, ops: str, dtype: str, keyed: bool) -> None:
    """The statements of ``code`` that take an extremum are, in order, the
    maxima (``M``) and minima (``m``) of ``ops`` over two names cast to
    ``dtype``: over unsigned keys when ``keyed``, else plain."""
    statements = re.findall(
        r"^\s*v_\d+ = (.*cute\.math\.m(?:ax|in)\(.*)$", code, re.MULTILINE
    )
    assert len(statements) == len(ops), statements
    value = rf"cutlass\.{dtype}\(\w+\)"
    for statement, letter in zip(statements, ops, strict=True):
        op = "max" if letter == "M" else "min"
        if keyed:
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
