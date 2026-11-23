"""CPU tests: a CuTe reduction takes an int32 or int16 maximum or minimum
over unsigned keys.  It accumulates one element after another, a chain of
extrema that ptxas (CUDA 13.3) fuses into three-operand instructions, losing
a negation of the operands of an int32 or int16 one; an unsigned chain keeps
every negation, and so does int8's, which stays as it is.  The three emitters
of such a chain (the accumulate of a thread's lanes, the accumulate of a
reduction loop and the accumulate of the ordered tile lowering) take the
form from ``math_templates.nan_extremum_expr``.  The extrema of one
expression are a chain too: the maxima and minima of a register program
(``helion.runtime.cute.register_tensor``) compare int32 through the same
keys.  ``torch.fmax`` / ``torch.fmin`` keep their signed form: the sorting
network of a top-k is made of them."""

from __future__ import annotations

import ast
import functools
import itertools
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import helion
from helion._compiler.cute.lane_loops.markers import _combine_expr
from helion._compiler.cute.lane_loops.markers import _LaneReduceMarker
from helion._compiler.cute.math_templates import nan_extremum_expr
from helion._compiler.cute.resident_sequence import _combine as sequence_combine
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.integer_extremum_kernels import LOOPED
from helion._testing.integer_extremum_kernels import NESTED
from helion._testing.integer_extremum_kernels import ONE_WARP
from helion._testing.integer_extremum_kernels import SEQUENCE
from helion._testing.integer_extremum_kernels import SEQUENCE_FOUR_WARPS
from helion._testing.integer_extremum_kernels import register_maxima_of_negated
from helion._testing.integer_extremum_kernels import register_minima_of_negated
from helion._testing.integer_extremum_kernels import row_argmin_of_negated
from helion._testing.integer_extremum_kernels import row_argmin_over_tiles
from helion._testing.integer_extremum_kernels import row_maxima_of_negated
from helion._testing.integer_extremum_kernels import row_minima_of_negated
from helion._testing.integer_extremum_kernels import rows_minus_maxima_of_negated
from helion._testing.integer_extremum_kernels import whole_row_argmin_in_nested_tiles
from helion._testing.integer_extremum_kernels import whole_row_maxima_of_negated
from helion._testing.integer_extremum_kernels import whole_row_minima_of_negated
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

# Signed dtype -> (unsigned dtype of its width, sign bit): the dtypes whose
# chains ptxas compiles with one operand's negation lost.
_KEYS = {
    "Int16": ("Uint16", 32768),
    "Int32": ("Uint32", 2147483648),
}
_TENSORS = {
    name: torch.empty(8, 256, dtype=dtype)
    for name, dtype in (
        ("Int8", torch.int8),
        ("Int16", torch.int16),
        ("Int32", torch.int32),
        ("Int64", torch.int64),
        ("Uint8", torch.uint8),
        ("Uint32", torch.uint32),
    )
}
_FLOAT32 = torch.empty(8, 256, dtype=torch.float32)
_FLOAT32_3D = torch.empty(4, 16, 1024, dtype=torch.float32)
# Rows of four groups of eight elements: a register region.
_REGISTER_ROWS = torch.empty(16, 4, 8, dtype=torch.int32)
_LAYOUTS = {
    "one_warp": ONE_WARP,
    "looped": LOOPED,
    "sequence": SEQUENCE,
    "sequence_four_warps": SEQUENCE_FOUR_WARPS,
    "nested": NESTED,
    "elements": {"block_sizes": [4, 64]},
    "default": {},
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _numeric_extrema_of_negated(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.fmax`` and ``torch.fmin`` of three negated values per element."""
    m, n = x.size()
    high = torch.empty_like(x)
    low = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            value = x[tile_m, tile_n]
            high[tile_m, tile_n] = torch.fmax(
                torch.fmax(-value, -(value * 2)), -(value + 1)
            )
            low[tile_m, tile_n] = torch.fmin(
                torch.fmin(-value, -(value * 2)), -(value + 1)
            )
    return high, low


def _keyed(signed: str, op: str, acc: str, value: str) -> str:
    """The extremum of ``acc`` and ``value`` over ``signed``'s unsigned keys,
    as generated code spells it."""
    unsigned, sign = _KEYS[signed]
    bit = f"cutlass.{unsigned}({sign})"
    return (
        f"cutlass.{signed}(cute.math.{op}(cutlass.{unsigned}({acc}) ^ {bit}, "
        f"cutlass.{unsigned}({value}) ^ {bit}) ^ {bit})"
    )


def _plain(op: str, acc: str, value: str) -> str:
    return f"cute.math.{op}({acc}, {value}, propagate_nan=True)"


def _unparsed(expression: str) -> str:
    return ast.unparse(ast.parse(expression, mode="eval"))


@functools.cache
def _code(kernel: Kernel[Any], args: tuple[object, ...], layout: str) -> str:
    """The code of ``kernel`` at a layout; each one is rendered once per
    process (``args`` are the module's tensors, hashed by identity)."""
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config.from_dict(dict(_LAYOUTS[layout]))
            )
        )


def _accumulates(code: str, accumulator: str) -> list[str]:
    """The right-hand sides that combine into ``accumulator``: those of its
    assignments that read it."""
    return [
        value
        for value in re.findall(rf"^\s*{accumulator} = (.*)$", code, re.MULTILINE)
        if accumulator in value
    ]


@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("signed", sorted(_KEYS))
def test_a_signed_chain_extremum_compares_unsigned_keys(signed: str, op: str) -> None:
    """Both operands are flipped at the sign bit and compared as unsigned
    values of the same width; the result is flipped back."""
    expression = nan_extremum_expr(
        op, "acc", "value", float32=False, chain_dtype=f"cutlass.{signed}"
    )
    assert _unparsed(expression) == _keyed(signed, op, "acc", "value")


def _numpy_cutlass() -> SimpleNamespace:
    """``cutlass``'s integer constructors as numpy conversions, which wrap
    like the DSL's."""

    def constructor(dtype: type) -> Any:
        return lambda value: np.asarray(value).astype(dtype)[()]

    return SimpleNamespace(
        **{
            name: constructor(getattr(np, name.lower()))
            for name in ("Int16", "Int32", "Uint16", "Uint32")
        }
    )


@pytest.mark.parametrize("signed", sorted(_KEYS))
def test_keyed_extrema_equal_the_signed_extrema(signed: str) -> None:
    """Over the bounds of the dtype, the values next to them and around
    zero, the keyed expression returns the signed maximum and minimum, in
    the signed dtype."""
    scalar = getattr(np, signed.lower())
    info = np.iinfo(scalar)
    values = [info.min, info.min + 1, -2, -1, 0, 1, 2, info.max - 1, info.max]
    namespace = {
        "cutlass": _numpy_cutlass(),
        "cute": SimpleNamespace(math=SimpleNamespace(max=max, min=min)),
    }
    for op, expected in (("max", max), ("min", min)):
        expression = nan_extremum_expr(
            op, "a", "b", float32=False, chain_dtype=f"cutlass.{signed}"
        )
        for a, b in itertools.product(values, repeat=2):
            result = eval(expression, namespace, {"a": scalar(a), "b": scalar(b)})
            assert result.dtype == scalar, (op, a, b)
            assert result == expected(a, b), (op, a, b)


# Nine cases on one axis, the dtype a caller names: each has a reason of its
# own to keep the plain form, and none may pick up keys by accident (a
# prefix match on the dtype's name, a default for unknown dtypes).  Int8 is
# the signed integer whose chain ptxas compiles right; Uint8, Uint16 and
# Uint32 are the unsigned widths, the last two the key dtypes themselves;
# Int64 is the signed integer whose extremum is a compare and a select;
# Float64 and BFloat16 are the floats whose callers pass ``float32=False``;
# Boolean is the remaining non-float dtype; None is a caller outside a chain.
@pytest.mark.parametrize(
    "dtype",
    [
        "cutlass.Int8",
        "cutlass.Uint8",
        "cutlass.Uint16",
        "cutlass.Uint32",
        "cutlass.Int64",
        "cutlass.Float64",
        "cutlass.BFloat16",
        "cutlass.Boolean",
        None,
    ],
)
def test_other_chains_keep_the_plain_extremum(dtype: str | None) -> None:
    """ptxas keeps every negation of an Int8 chain (each is sign-extended
    before the packed 16-bit instruction) and of an unsigned one, a 64-bit
    extremum is a compare and a select, a Float64 or BFloat16 chain keeps
    the NaN-propagating extremum of its dtype (only fp32 has the
    one-instruction form), a Boolean has no sign to lose, and a caller that
    does not know the dtype gets what it got."""
    assert nan_extremum_expr(
        "max", "acc", "value", float32=False, chain_dtype=dtype
    ) == _plain("max", "acc", "value")


def test_an_fp32_chain_keeps_its_one_instruction() -> None:
    assert (
        nan_extremum_expr(
            "min", "acc", "value", float32=True, chain_dtype="cutlass.Float32"
        )
        == "cute.arch.fmin(acc, value, nan=True)"
    )


def test_an_extremum_outside_a_chain_is_not_keyed() -> None:
    """Without ``chain_dtype`` (the combines of the threads' partials, the
    candidate indices of an argmin) the form does not depend on the dtype."""
    assert nan_extremum_expr("max", "a", "b", float32=False) == _plain("max", "a", "b")


def test_the_sequence_lowering_keys_the_element_accumulate_only() -> None:
    """The ordered lowering of a tile's reductions (``cute_reduction_sequence``)
    passes the reduction's dtype where it accumulates a tile's elements; the
    warps' partials and the steps of its warp reduction pass none, and fp32
    keeps the form this lowering emits."""
    assert _unparsed(
        sequence_combine("min", "acc", "cutlass.Int16(v_0)", "cutlass.Int16")
    ) == _keyed("Int16", "min", "acc", "cutlass.Int16(v_0)")
    assert sequence_combine("max", "a", "b") == _plain("max", "a", "b")
    assert sequence_combine(
        "max", "acc", "cutlass.Float32(v_0)", "cutlass.Float32"
    ) == _plain("max", "acc", "cutlass.Float32(v_0)")
    assert sequence_combine("sum", "acc", "v_0", "cutlass.Int16") == "(acc) + (v_0)"


# Four cases: each keyed dtype has its own key width and sign bit, and the
# maximum and the minimum are separate reductions (a one-warp minimum reduces
# through the generic warp reduction, a maximum through the native one).
@pytest.mark.parametrize(
    ("kernel", "op", "bound"),
    [(row_maxima_of_negated, "max", "min"), (row_minima_of_negated, "min", "max")],
    ids=["max", "min"],
)
@pytest.mark.parametrize("signed", sorted(_KEYS))
def test_the_lane_accumulate_of_a_signed_extremum_is_keyed(
    signed: str, kernel: Kernel[Any], op: str, bound: str
) -> None:
    """Four columns per thread, one element per load: the thread combines
    its columns into ``a{op}_lane_acc`` over unsigned keys."""
    x = _TENSORS[signed]
    start = getattr(torch.iinfo(x.dtype), bound)
    code = _code(kernel, (x, start), "one_warp")
    accumulator = f"a{op}_lane_acc"
    assert _accumulates(code, accumulator) == [
        _keyed(signed, op, accumulator, f"cutlass.{signed}(v_0)")
    ]


@pytest.mark.parametrize(
    ("kernel", "op"),
    [(whole_row_maxima_of_negated, "max"), (whole_row_minima_of_negated, "min")],
    ids=["max", "min"],
)
@pytest.mark.parametrize("signed", sorted(_KEYS))
def test_the_loop_accumulate_of_a_signed_extremum_is_keyed(
    signed: str, kernel: Kernel[Any], op: str
) -> None:
    """A reduction loop accumulates its trips per thread the same way (Int32
    did already; Int16 took the signed form), and the kernel has no other
    extremum in its text: the threads combine in a runtime helper."""
    code = _code(kernel, (_TENSORS[signed],), "looped")
    accumulator = f"a{op}_acc"
    assert _accumulates(code, accumulator) == [
        _keyed(signed, op, accumulator, f"cutlass.{signed}(v_0)")
    ]
    assert "propagate_nan" not in code


@pytest.mark.parametrize(
    ("kernel", "x", "layout"),
    [
        (row_argmin_over_tiles, _FLOAT32, "one_warp"),
        (whole_row_argmin_in_nested_tiles, _FLOAT32_3D, "nested"),
    ],
    ids=["over_tiles", "whole_row"],
)
def test_the_candidate_indices_of_an_argmin_keep_the_plain_minimum(
    kernel: Kernel[Any], x: torch.Tensor, layout: str
) -> None:
    """The lanes of an argmin accumulate its value, then the lowest index
    that holds it: a minimum over Int32 candidates, each an index or the
    highest index and never a negation.  Both emitters of that minimum mark
    it, and its accumulate keeps the plain form, which ptxas turns into
    selects."""
    code = _code(kernel, (x,), layout)
    assert _accumulates(code, "argmin_lane_acc") == [
        _plain("min", "argmin_lane_acc", "cutlass.Int32(lane_reduce_index)")
    ]


@pytest.mark.parametrize("signed", sorted(_KEYS))
def test_the_value_of_an_integer_argmin_is_keyed_and_its_candidates_are_not(
    signed: str,
) -> None:
    """The value an argmin's lanes accumulate is a chain of the kernel's
    values like any other minimum, so an int32 or int16 one compares
    unsigned keys; the candidate indices behind it stay plain."""
    x = _TENSORS[signed]
    code = _code(row_argmin_of_negated, (x, torch.iinfo(x.dtype).max), "one_warp")
    assert _accumulates(code, "lane_reduce_value_lane_acc") == [
        _keyed(signed, "min", "lane_reduce_value_lane_acc", f"cutlass.{signed}(v_0)")
    ]
    assert _accumulates(code, "argmin_lane_acc") == [
        _plain("min", "argmin_lane_acc", "cutlass.Int32(lane_reduce_index)")
    ]


def test_the_lane_accumulate_exempts_marked_candidates_only() -> None:
    assert _combine_expr(
        "min", "acc", "cutlass.Int32(i)", "cutlass.Int32", index_candidates=True
    ) == _plain("min", "(acc)", "(cutlass.Int32(i))")
    assert _unparsed(
        _combine_expr("min", "acc", "cutlass.Int32(i)", "cutlass.Int32")
    ) == _keyed("Int32", "min", "acc", "cutlass.Int32(i)")


def test_a_marker_accumulates_its_input_in_its_accumulators_dtype() -> None:
    """Every lane accumulate is the marker's: its input (or a name that
    holds a share of it) cast to the dtype of its identity, combined as its
    reduction and its kind of minimum say."""

    def marker(reduction_type: str, identity: str, **fields: bool) -> Any:
        return _LaneReduceMarker(
            result_var="r",
            input_name="x",
            reduction_type=reduction_type,
            identity_expr=identity,
            threads_in_group=32,
            wrap_template="",
            **fields,
        )

    values = marker("min", "cutlass.Int32(2147483647)")
    assert _unparsed(values.accumulate_expr("acc")) == _keyed(
        "Int32", "min", "acc", "cutlass.Int32(x)"
    )
    assert _unparsed(values.accumulate_expr("acc", "share")) == _keyed(
        "Int32", "min", "acc", "cutlass.Int32(share)"
    )
    candidates = marker("min", "cutlass.Int32(2147483647)", index_candidates=True)
    assert _unparsed(candidates.accumulate_expr("acc")) == _plain(
        "min", "acc", "cutlass.Int32(x)"
    )
    assert (
        _unparsed(
            marker("max", "cutlass.Float32(float('-inf'))").accumulate_expr("acc")
        )
        == "cute.arch.fmax(acc, cutlass.Float32(x), nan=True)"
    )
    assert marker("sum", "0.0").accumulate_expr("acc") == "(acc) + (x)"


@pytest.mark.parametrize("signed", sorted(_KEYS))
def test_the_sequence_accumulate_of_a_signed_extremum_is_keyed(signed: str) -> None:
    """Eight columns per thread, accumulated by the ordered lowering of the
    tile's reductions: the third emitter of an element chain."""
    code = _code(rows_minus_maxima_of_negated, (_TENSORS[signed],), "sequence")
    assert _accumulates(code, "sequence_acc") == [
        _keyed(signed, "max", "sequence_acc", f"cutlass.{signed}(v_0)")
    ]


@pytest.mark.parametrize(
    ("dtype", "cast"),
    [("Uint8", True), ("Uint32", True), ("Int32", False), ("Int8", False)],
)
def test_the_sequence_lowering_casts_the_unsigned_partials_it_reads_back(
    dtype: str, cast: bool
) -> None:
    """On several warps the ordered lowering combines the warps' partials
    from shared memory.  A tensor of an unsigned dtype reads back signed
    there, which the DSL refuses to combine with the unsigned accumulator
    (``All operands to max must have the same numeric type``), so that read
    is cast; a signed partial is read as it was."""
    code = _code(
        rows_minus_maxima_of_negated, (_TENSORS[dtype],), "sequence_four_warps"
    )
    (combine,) = _accumulates(code, "sequence_group_value")
    read = re.search(r"sequence_smem\[[^\]]*\]", combine)
    assert read is not None, combine
    partial = f"cutlass.{dtype}({read.group()})" if cast else read.group()
    assert combine == _plain("max", "sequence_group_value", partial)


@pytest.mark.parametrize("dtype", ["Int8", "Uint8", "Uint32", "Int64"])
def test_int8_unsigned_and_64_bit_accumulates_keep_their_form(dtype: str) -> None:
    """Int8 keeps its signed extremum in each of the three accumulates (its
    rows were right, and the keys would slow it down); the unsigned dtypes
    and Int64 have nothing to key."""
    x = _TENSORS[dtype]
    code = _code(row_maxima_of_negated, (x, torch.iinfo(x.dtype).min), "one_warp")
    assert _accumulates(code, "amax_lane_acc") == [
        _plain("max", "amax_lane_acc", f"cutlass.{dtype}(v_0)")
    ]
    code = _code(whole_row_maxima_of_negated, (x,), "looped")
    assert _accumulates(code, "amax_acc") == [
        _plain("max", "amax_acc", f"cutlass.{dtype}(v_0)")
    ]
    code = _code(rows_minus_maxima_of_negated, (x,), "sequence")
    assert _accumulates(code, "sequence_acc") == [
        _plain("max", "sequence_acc", f"cutlass.{dtype}(v_0)")
    ]


@pytest.mark.parametrize("dtype", ["Int32", "Int16", "Int8"])
def test_numeric_extrema_keep_the_plain_form(dtype: str) -> None:
    """``torch.fmax`` / ``torch.fmin`` are the operations the sorting network
    of a top-k is built from, over encoded keys, and keys would slow it: an
    integer one keeps ``cute.math.max`` / ``min`` on its own values, in a
    chain of negated values too (whose int32 and int16 results are not
    eager's)."""
    code = _code(_numeric_extrema_of_negated, (_TENSORS[dtype],), "elements")
    value = rf"cutlass\.{dtype}\(\w+\)"
    for op in ("max", "min"):
        statements = re.findall(rf"^\s*f{op}(?:_\d+)? = (.*)$", code, re.MULTILINE)
        assert len(statements) == 2, statements
        form = rf"cute\.math\.{op}\({value}, {value}, propagate_nan=False\)"
        for statement in statements:
            assert re.fullmatch(form, statement), statement
    assert "Uint" not in code


@pytest.mark.parametrize(
    ("kernel", "op"),
    [
        (register_maxima_of_negated, "aten.maximum.default"),
        (register_minima_of_negated, "aten.minimum.default"),
    ],
    ids=["maximum", "minimum"],
)
def test_the_extrema_of_a_register_region_are_its_programs(
    kernel: Kernel[Any], op: str
) -> None:
    """The rows of these kernels lower to a register program, which holds
    both extrema of the chain: the kernel's text has none, and the runtime
    helper of the program takes them."""
    code = _code(kernel, (_REGISTER_ROWS,) * 3, "default")
    assert "_cute_execute_register_plan" in code
    assert code.count(op) == 2
    assert "cute.math.max(" not in code and "cute.math.min(" not in code


@pytest.mark.parametrize("minimum", [False, True], ids=["max", "min"])
def test_a_register_program_takes_an_int32_extremum_over_unsigned_keys(
    minimum: bool,
) -> None:
    """The helper flips the sign bit of both register tensors, takes the
    unsigned extremum and flips the result back.  An Int64 extremum (the
    other integer dtype of a register program) stays the signed one, and so
    does an Int32 ``torch.fmax`` / ``torch.fmin``, which the sorting network
    of a top-k is made of."""
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    from helion.runtime.cute.register_tensor import _minimum_maximum

    def emitted(dtype: Any, propagate_nan: bool = True) -> str:
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            lanes = ir.VectorType.get([8], dtype.mlir_type)
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("entry", ([lanes, lanes], [lanes]))
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    left = cute.TensorSSA(block.arguments[0], (8,), dtype)
                    right = cute.TensorSSA(block.arguments[1], (8,), dtype)
                    result = _minimum_maximum(
                        left, right, minimum=minimum, propagate_nan=propagate_nan
                    )
                    func.ReturnOp([result.ir_value()])
            assert module.operation.verify()
            return str(module)

    op = "min" if minimum else "max"
    initialized = torch.cuda.is_initialized()
    keyed = emitted(cutlass.Int32)
    assert keyed.count(f"arith.{op}ui") == 1 and f"arith.{op}si" not in keyed
    assert keyed.count("arith.xori") == 3
    assert "arith.constant -2147483648 : i32" in keyed
    for plain in (emitted(cutlass.Int64), emitted(cutlass.Int32, False)):
        assert plain.count(f"arith.{op}si") == 1 and "arith.xori" not in plain
    assert torch.cuda.is_initialized() == initialized
