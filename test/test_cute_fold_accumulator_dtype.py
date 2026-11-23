"""CPU tests: the accumulator ``hoist_warp_reduce`` folds a reduction's vector
lanes into has the dtype the reduction combines in.  A grouped helper names
that dtype in its identity argument and the emitter of a direct warp
reduction records it; the text of the reduce input's definition (a load, a
product) does not show it, and a reduce whose dtype nothing states keeps its
loop instead of folding in fp32.  The maximum or minimum of an integer of at
most 32 bits is held in a float that holds it (fp32 up to 16 bits, FP64 for
32) and converted back for the reduce."""

from __future__ import annotations

import ast
import functools
import re
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.fold_lane_reductions import fold_lane_reductions
from helion._compiler.cute.fold_lane_reductions import fold_site_key
from helion._compiler.cute.hoist_warp_reduce import hoist_warp_reduce_from_vloop
from helion._compiler.cute.state import CuteLaneState
from helion._compiler.device_function import DeviceFunction
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.integer_extremum_kernels import row_maxima_of_negated
from helion._testing.integer_extremum_kernels import row_minima_of_negated
from helion._testing.typed_fold_kernels import ONE_WARP
from helion._testing.typed_fold_kernels import SPLIT_MOMENTS
from helion._testing.typed_fold_kernels import TWO_PACKETS
from helion._testing.typed_fold_kernels import TWO_WARPS
from helion._testing.typed_fold_kernels import TWO_WARPS_TWO_PACKETS
from helion._testing.typed_fold_kernels import row_dots_int32
from helion._testing.typed_fold_kernels import row_dots_int64
from helion._testing.typed_fold_kernels import row_logsumexp
from helion._testing.typed_fold_kernels import row_logsumexp_of_doubled
from helion._testing.typed_fold_kernels import row_maxima
from helion._testing.typed_fold_kernels import row_minima
from helion._testing.typed_fold_kernels import row_square_sums
from helion._testing.typed_fold_kernels import row_sums
from helion._testing.typed_fold_kernels import row_sums_as_half
from helion._testing.typed_fold_kernels import split_column_moments

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

_FLOAT64 = torch.empty(8, 256, dtype=torch.float64)
_FLOAT32 = torch.empty(8, 256, dtype=torch.float32)
_INT64 = torch.empty(8, 256, dtype=torch.int64)
_INT32 = torch.empty(8, 256, dtype=torch.int32)
_INT16 = torch.empty(8, 256, dtype=torch.int16)
_INT8 = torch.empty(8, 256, dtype=torch.int8)
_UINT8 = torch.empty(8, 256, dtype=torch.uint8)
_FLOAT8 = torch.empty(8, 256, dtype=torch.float8_e5m2)
_HALF = torch.empty(8, 256, dtype=torch.bfloat16)
_SPLIT_ROWS = torch.empty(8100, 64, dtype=torch.float32)
_LAYOUTS = {
    "one_warp": ONE_WARP,
    "two_packets": TWO_PACKETS,
    "two_warps": TWO_WARPS,
    "two_warps_two_packets": TWO_WARPS_TWO_PACKETS,
    "split_knobs": SPLIT_MOMENTS,
}


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


def _fold_inits(code: str) -> list[str]:
    """The right-hand sides of the V-fold accumulators' initializations."""
    return re.findall(r"^\s*_helion_vfold_acc_\d+ = (cutlass\..*)$", code, re.MULTILINE)


def _fold_lines(code: str) -> str:
    """The lines of the folds and of the reduces they feed: a V-fold
    accumulator's initialization and updates, the lane fold's tree, and
    every cross-thread reduce call."""
    return "\n".join(
        line
        for line in code.splitlines()
        if "_helion_vfold_acc_" in line
        or "_fold_tree_" in line
        or "warp_reduction" in line
        or "_cute_grouped_reduce_" in line
    )


def _folded_reduces(code: str) -> list[ast.Call]:
    """The cross-thread reduce calls that reduce a V-fold accumulator, or its
    conversion to the reduce's dtype."""
    reduces = []
    for node in ast.walk(ast.parse(code)):
        if not (
            isinstance(node, ast.Call)
            and node.args
            and "reduc" in ast.unparse(node.func)
        ):
            continue
        source = node.args[0]
        if isinstance(source, ast.Call) and len(source.args) == 1:
            source = source.args[0]
        if isinstance(source, ast.Name) and source.id.startswith("_helion_vfold_acc_"):
            reduces.append(node)
    return reduces


@pytest.mark.parametrize(
    ("kernel", "args", "dtype", "init", "unfolded"),
    [
        (row_sums, (_FLOAT64,), "cutlass.Float64", "cutlass.Float64(0)", ()),
        (row_square_sums, (_FLOAT64,), "cutlass.Float64", "cutlass.Float64(0)", ()),
        (row_sums, (_INT64,), "cutlass.Int64", "cutlass.Int64(0)", ()),
        (row_dots_int32, (_INT32, _INT32), "cutlass.Int32", "cutlass.Int32(0)", ()),
        # Across warps the grouped reduce takes the product's conversion to
        # int64, not a name: the hoist leaves that loop alone, and the lane
        # fold rejects the two-packet layout for it.
        (
            row_dots_int64,
            (_INT32, _INT32),
            "cutlass.Int64",
            "cutlass.Int64(0)",
            ("two_warps", "two_warps_two_packets"),
        ),
        (
            row_maxima,
            (_FLOAT64, float("-inf")),
            "cutlass.Float64",
            "cutlass.Float64(float('-inf'))",
            (),
        ),
        # The maximum of an integer of at most 32 bits is held in a float
        # that holds it, from the integer's lowest value.
        (
            row_maxima,
            (_INT32, -(2**31)),
            "cutlass.Int32",
            "cutlass.Float64(-2147483648)",
            (),
        ),
        (row_maxima, (_UINT8, 0), "cutlass.Uint8", "cutlass.Float32(0)", ()),
    ],
    ids=[
        "float64_sum",
        "float64_square_sum",
        "int64_sum",
        "int32_dot",
        "int32_dot_summed_in_int64",
        "float64_max",
        "int32_max",
        "uint8_max",
    ],
)
def test_vector_lanes_fold_in_the_reductions_dtype(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    dtype: str,
    init: str,
    unfolded: tuple[str, ...],
) -> None:
    """One warp (a direct warp reduction) and two warps (a grouped reduce
    through shared memory), each with one packet per thread and with two
    (the lane loop's reduce is collapsed or folded across its lanes too)."""
    for layout in ("one_warp", "two_packets", "two_warps", "two_warps_two_packets"):
        if layout == "two_warps_two_packets" and layout in unfolded:
            with pytest.raises(exc.BackendUnsupported, match="cross-lane"):
                _code(kernel, args, layout)
            continue
        code = _code(kernel, args, layout)
        # Nothing in these folds is fp32 but the accumulator that carries the
        # uint8 maximum: no conversion of a lane's value, no identity.
        assert ("cutlass.Float32" in _fold_lines(code)) == (
            init.startswith("cutlass.Float32") and layout not in unfolded
        ), layout
        if layout in unfolded:
            assert not _fold_inits(code), layout
            continue
        assert _fold_inits(code) == [init], layout
        (reduce,) = _folded_reduces(code)
        # The reduce combines in the reduction's dtype: it reads the
        # accumulator, converted where a float holds an integer's extremum.
        assert ast.unparse(reduce.args[0]) == (
            "_helion_vfold_acc_0"
            if init.startswith(f"{dtype}(")
            else f"{dtype}(_helion_vfold_acc_0)"
        ), layout
        if isinstance(reduce.func, ast.Name):
            # A grouped helper: its identity keeps the reduction's dtype.
            assert layout.startswith("two_warps")
            assert ast.unparse(reduce.args[2]).startswith(f"{dtype}("), layout
        else:
            assert not layout.startswith("two_warps")


def test_integer_extrema_fold_across_the_lanes_of_a_packet_loop() -> None:
    """The lane fold reads the accumulator's identity: an integer maximum
    starts from its dtype's lowest value, not from an infinity, and its
    FP64 accumulators are combined in FP64 ahead of the one conversion."""
    code = _code(row_maxima, (_INT32, -(2**31)), "two_packets")
    assert "_fold_parts_.append(_helion_vfold_acc_0)" in code
    assert (
        "_fold_tree_ = cute.math.max(_fold_parts_[0], _fold_parts_[1], "
        "propagate_nan=True)" in code
    )
    assert code.count("cutlass.Int32(_helion_vfold_acc_0)") == 1


def test_a_grouped_integer_minimum_folds_from_its_dtypes_highest_value() -> None:
    """Across warps a minimum is a grouped reduce, which the hoist folds (the
    direct one-warp minimum is a generic warp reduction it leaves alone), and
    with two packets per thread the lane fold combines the lanes first."""
    for layout in ("two_warps", "two_warps_two_packets"):
        code = _code(row_minima, (_INT16, 32767), layout)
        assert _fold_inits(code) == ["cutlass.Float32(32767)"]
        (reduce,) = _folded_reduces(code)
        assert ast.unparse(reduce.args[0]) == "cutlass.Int16(_helion_vfold_acc_0)"
        assert ast.unparse(reduce.args[1]) == "'min'"
        assert ast.unparse(reduce.args[2]).startswith("cutlass.Int16(")
    assert (
        "_fold_tree_ = cute.arch.fmin(_fold_parts_[0], _fold_parts_[1], nan=True)"
        in code
    )


# The fold of a lane's value into the accumulator: the hardware maximum in
# fp32, compares and selects in every other dtype.
_FOLD_COMBINE = re.compile(
    r"^\s*_helion_vfold_acc_0 = cute\.(?:math|arch)\.(f?(?:max|min))"
    r"\(_helion_vfold_acc_0, (.*), (?:propagate_)?nan=True\)$",
    re.MULTILINE,
)


@pytest.mark.parametrize(
    ("x", "bits", "carrier"),
    [
        (_INT8, 8, "cutlass.Float32"),
        (_INT16, 16, "cutlass.Float32"),
        (_INT32, 32, "cutlass.Float64"),
    ],
    ids=["int8", "int16", "int32"],
)
def test_an_integer_extremum_of_at_most_32_bits_folds_in_a_float(
    x: torch.Tensor, bits: int, carrier: str
) -> None:
    """ptxas loses a negation in a chain of 32-bit integer maxima over
    negated operands, which an integer fold of ``torch.amax(-x)`` is: the
    fold is one in the narrower float that holds the integer (fp32 up to 16
    bits, FP64 for 32), of each lane's integer converted, and the reduce
    takes the accumulator converted back.  A maximum on one warp (a direct
    warp reduction) and across two with two packets per thread (the lane fold
    combines them); a minimum across two warps (the one-warp minimum is a
    generic warp reduction, which is not folded)."""
    dtype = f"cutlass.Int{bits}"
    prefix = "f" if carrier == "cutlass.Float32" else ""
    lowest, highest = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    for kernel, start, layout in (
        (row_maxima_of_negated, lowest, "one_warp"),
        (row_maxima_of_negated, lowest, "two_warps_two_packets"),
        (row_minima_of_negated, highest, "two_warps"),
    ):
        code = _code(kernel, (x, start), layout)
        assert _fold_inits(code) == [f"{carrier}({start})"], layout
        ((op, addend),) = _FOLD_COMBINE.findall(code)
        extremum = "max" if kernel is row_maxima_of_negated else "min"
        assert op == prefix + extremum, layout
        assert addend.startswith(f"{carrier}({dtype}("), layout
        (reduce,) = _folded_reduces(code)
        assert ast.unparse(reduce.args[0]) == f"{dtype}(_helion_vfold_acc_0)", layout
    code = _code(row_minima_of_negated, (x, highest), "one_warp")
    assert not _fold_inits(code)


def test_an_int64_extremum_folds_in_int64() -> None:
    """FP64 does not hold every int64, and a 64-bit maximum is compiled to
    compares and selects, not to the instruction that loses the negation."""
    for kernel, start, layout in (
        (row_maxima_of_negated, -(2**63), "one_warp"),
        (row_minima_of_negated, 2**63 - 1, "two_warps_two_packets"),
    ):
        code = _code(kernel, (_INT64, start), layout)
        assert "cutlass.Float64" not in _fold_lines(code), layout
        assert _fold_inits(code) == [f"cutlass.Int64({start})"], layout
        (reduce,) = _folded_reduces(code)
        assert ast.unparse(reduce.args[0]) == "_helion_vfold_acc_0", layout


def test_a_narrower_conversion_of_the_input_is_folded_as_converted() -> None:
    """``x.to(torch.float16).sum(-1)`` of an fp32 ``x`` sums the rounded
    elements in fp32: the direct reduce's fold converts the fp16 value, where
    it used to convert the load and skip the rounding.  (Across warps the
    emitter's own conversion of the input is what the fold adds.)"""
    for layout in ("one_warp", "two_packets"):
        code = _code(row_sums_as_half, (_FLOAT32,), layout)
        assert _fold_inits(code) == ["cutlass.Float32(0)"], layout
        assert re.search(
            r"_helion_vfold_acc_0 \+ cutlass\.Float32\(cutlass\.Float16\(", code
        ), layout


def test_an_fp8_maximum_folds_in_fp32() -> None:
    """An fp8 reduction combines in fp8, in which no fold accumulates: like a
    half one it folds in fp32, which holds every fp8 value and an infinity.
    Across warps the hoist wraps the grouped helper's fp8 identity in fp32,
    and with two packets per thread the lane fold still knows the site: its
    key keeps the identity's own dtype."""
    for layout in ("one_warp", "two_packets", "two_warps", "two_warps_two_packets"):
        code = _code(row_maxima, (_FLOAT8, float("-inf")), layout)
        assert _fold_inits(code) == ["cutlass.Float32(float('-inf'))"], layout
        (reduce,) = _folded_reduces(code)
        if layout.startswith("two_warps"):
            assert ast.unparse(reduce.args[2]) == (
                "cutlass.Float32(cutlass.Float8E5M2(float('-inf')))"
            ), layout
    assert "_fold_tree_ = cute.arch.fmax(_fold_parts_[0], _fold_parts_[1]" in code
    recorded = "_cute_grouped_reduce_warp(v, 'max', {}, lane, pre=1, group_span=64)"
    assert fold_site_key(
        recorded.format("cutlass.Float32(cutlass.Float8E5M2(float('-inf')))")
    ) == fold_site_key(recorded.format("cutlass.Float8E5M2(float('-inf'))"))
    assert fold_site_key(
        recorded.format("cutlass.Float32(float('-inf'))")
    ) != fold_site_key(recorded.format("cutlass.Float8E5M2(float('-inf'))"))


def test_a_kept_half_conversion_decides_the_online_pair_fold() -> None:
    """The lane fold fuses a maximum with the sum of ``exp(value - maximum)``
    into one pass where the exponent reads the very value the maximum folded
    and the sum's terms are plain exponentials (``fold_lane_reductions``,
    "Online pair").  With the half conversion of a fold's input kept, both
    are decided on the values the kernel computes.  The maximum of ``2 * x``
    folds the rounded product, which the exponent reads: the folds fuse.
    With half-precision accumulators every exponential is rounded before it
    is summed, which is no plain exponential: the two folds stay apart (fused
    without that rounding, they were code the DSL rejects: the guarded frame
    mixed fp32 with the half dtype)."""
    code = _code(row_logsumexp_of_doubled, (_HALF, torch.float32), "two_packets")
    assert "_pair_root" in code and "_pair_frame" in code
    assert (
        "cute.arch.fmax(_helion_vfold_acc_0, cutlass.Float32(cutlass.BFloat16(" in code
    )
    code = _code(row_logsumexp, (_HALF, torch.bfloat16), "two_packets")
    assert "_pair_root" not in code
    assert code.count("_fold_tree_") == 4  # two trees, each bound and read
    assert re.search(
        r"_helion_vfold_acc_1 \+ cutlass\.Float32\(cutlass\.BFloat16\(", code
    )


def test_float64_moments_fold_in_float64_under_the_split_reduction_knobs() -> None:
    """Deferred tile reduce, tail peel and lane unroll on a split reduction:
    the sum's input is a select of conversions, the sum of squares' a bare
    product, and both fold in float64 through float64 grouped reduces."""
    code = _code(split_column_moments, (_SPLIT_ROWS, 1024), "split_knobs")
    assert _fold_inits(code) == ["cutlass.Float64(0)", "cutlass.Float64(0)"]
    assert "_helion_vfold_acc_1 + cutlass.Float64(shifted * shifted)" in code
    reduces = _folded_reduces(code)
    assert len(reduces) == 2
    for reduce in reduces:
        assert isinstance(reduce.func, ast.Name)
        assert ast.unparse(reduce.args[2]).startswith("cutlass.Float64(")


def test_the_emitter_records_one_dtype_per_reduced_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CuteBackend.reduction_expr`` records the dtype its caller passes
    under the name of the reduced value; a second dtype for that name leaves
    none, and a call without a dtype records nothing."""
    lanes = CuteLaneState()
    function = SimpleNamespace(cute_state=SimpleNamespace(lanes=lanes))
    monkeypatch.setattr(DeviceFunction, "current", staticmethod(lambda: function))
    backend = CuteBackend()
    assert (
        backend.reduction_expr("v", "sum", 0, threads_in_group=32, dtype=torch.float64)
        == "cute.arch.warp_reduction_sum(v, threads_in_group=32)"
    )
    backend.reduction_expr("v", "max", 0, threads_in_group=32, dtype=torch.float64)
    backend.reduction_expr("w", "sum", 0, threads_in_group=32)
    assert lanes.reduce_input_dtypes == {"v": "cutlass.Float64"}
    backend.reduction_expr("v", "sum", 0, threads_in_group=32, dtype=torch.int64)
    assert lanes.reduce_input_dtypes == {"v": None}
    # The argmax lowering passes the dtypes of its value and index reductions.
    lanes.reduce_input_dtypes.clear()
    backend.argreduce_result_expr(
        "u",
        "position",
        "argmax",
        0,
        torch.int64,
        index_dtype=torch.int32,
        threads_in_group=32,
        dtype=torch.float32,
    )
    assert lanes.reduce_input_dtypes.pop("u") == "cutlass.Float32"
    assert list(lanes.reduce_input_dtypes.values()) == ["cutlass.Int32"]


def _hoist(
    src: str,
    dtypes: dict[str, str | None] | None = None,
    rename_groups: dict[str, str] | None = None,
) -> str:
    body = hoist_warp_reduce_from_vloop(
        ast.parse(textwrap.dedent(src)).body,
        reduce_input_dtypes=dtypes,
        rename_groups=rename_groups,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


_DIRECT = """
for vec_lane_1 in cutlass.range_constexpr(4):
    value = values[vec_lane_1]
    square = value * value
    total = cutlass.Float64(cute.arch.warp_reduction_sum(square, threads_in_group=32))
    carried = carried + total
"""


def test_a_direct_reduce_folds_in_its_recorded_dtype() -> None:
    hoisted = _hoist(_DIRECT, {"square": "cutlass.Float64"})
    assert "_helion_vfold_acc_0 = cutlass.Float64(0)" in hoisted
    assert "_helion_vfold_acc_0 + cutlass.Float64(value * value)" in hoisted
    assert "warp_reduction_sum(_helion_vfold_acc_0, threads_in_group=32)" in hoisted


@pytest.mark.parametrize(
    "dtypes",
    [None, {"square": None}, {"square": "cutlass.Boolean"}],
    ids=["none", "two", "boolean"],
)
def test_a_direct_reduce_without_one_foldable_dtype_keeps_its_loop(
    dtypes: dict[str, str | None] | None,
) -> None:
    """The product's text does not show its dtype, a name recorded with two
    dtypes has none, and a Boolean accumulator has no identity (the
    constructor turns an infinity into True): the loop stays as written (its
    per-lane sums add up to the same carry)."""
    assert _hoist(_DIRECT, dtypes) == ast.unparse(ast.parse(_DIRECT))


def test_the_inputs_own_conversion_is_kept_under_the_accumulators() -> None:
    """The fold adds the value the loop reduced: a conversion in the input's
    definition stays (it rounds a wider operand) and is converted to the
    accumulator's dtype."""
    src = _DIRECT.replace("value * value", "cutlass.Float32(value * value)")
    hoisted = _hoist(src, {"square": "cutlass.Float64"})
    assert (
        "_helion_vfold_acc_0 + cutlass.Float64(cutlass.Float32(value * value))"
        in hoisted
    )
    # Without a record the definition's conversion is the dtype, and a
    # conversion to the accumulator's own dtype is not converted again.
    hoisted = _hoist(src)
    assert "_helion_vfold_acc_0 = cutlass.Float32(0)" in hoisted
    assert "_helion_vfold_acc_0 + cutlass.Float32(value * value)" in hoisted
    src = _DIRECT.replace("value * value", "cutlass.Float64(value * value)")
    hoisted = _hoist(src, {"square": "cutlass.Float64"})
    assert "_helion_vfold_acc_0 + cutlass.Float64(value * value)" in hoisted
    # A half conversion under the fp32 accumulator of a half reduction, bare
    # and in both arms of a select.
    src = _DIRECT.replace("value * value", "cutlass.Float16(value * value)")
    hoisted = _hoist(src, {"square": "cutlass.Float32"})
    assert (
        "_helion_vfold_acc_0 + cutlass.Float32(cutlass.Float16(value * value))"
        in hoisted
    )
    src = _DIRECT.replace(
        "value * value",
        "cutlass.BFloat16(value) if mask else cutlass.BFloat16(0)",
    )
    hoisted = _hoist(src, {"square": "cutlass.Float32"})
    assert (
        "_helion_vfold_acc_0 + (cutlass.Float32(cutlass.BFloat16(value)) if mask "
        "else cutlass.Float32(cutlass.BFloat16(0)))" in hoisted
    )


_FMAX = "cute.arch.fmax(_helion_vfold_acc_0, cutlass.Float32(value + bias), nan=True)"


@pytest.mark.parametrize(
    ("op", "identity", "init", "combine"),
    [
        # A reduction in a float narrower than fp32 (half precision, fp8)
        # folds in fp32, with the hardware maximum.
        (
            "max",
            "cutlass.BFloat16(float('-inf'))",
            "cutlass.Float32(float('-inf'))",
            _FMAX,
        ),
        (
            "max",
            "cutlass.Float8E5M2(float('-inf'))",
            "cutlass.Float32(float('-inf'))",
            _FMAX,
        ),
        (
            "sum",
            "cutlass.Int64(0)",
            "cutlass.Int64(0)",
            "_helion_vfold_acc_0 + cutlass.Int64(value + bias)",
        ),
        # An integer sum folds in its dtype (it wraps as the loop's does); an
        # extremum of at most 16 bits in fp32, of 32 bits in FP64, of 64 in
        # its dtype.
        (
            "sum",
            "cutlass.Int32(0)",
            "cutlass.Int32(0)",
            "_helion_vfold_acc_0 + cutlass.Int32(value + bias)",
        ),
        (
            "min",
            "cutlass.Uint8(255)",
            "cutlass.Float32(255)",
            (
                "cute.arch.fmin(_helion_vfold_acc_0, "
                "cutlass.Float32(cutlass.Uint8(value + bias)), nan=True)"
            ),
        ),
        (
            "min",
            "cutlass.Uint16(65535)",
            "cutlass.Float32(65535)",
            (
                "cute.arch.fmin(_helion_vfold_acc_0, "
                "cutlass.Float32(cutlass.Uint16(value + bias)), nan=True)"
            ),
        ),
        (
            "max",
            "cutlass.Uint32(0)",
            "cutlass.Float64(0)",
            (
                "cute.math.max(_helion_vfold_acc_0, "
                "cutlass.Float64(cutlass.Uint32(value + bias)), propagate_nan=True)"
            ),
        ),
        (
            "max",
            "cutlass.Int64(-9223372036854775808)",
            "cutlass.Int64(-9223372036854775808)",
            (
                "cute.math.max(_helion_vfold_acc_0, "
                "cutlass.Int64(value + bias), propagate_nan=True)"
            ),
        ),
    ],
    ids=[
        "half_max",
        "fp8_max",
        "int64_sum",
        "int32_sum",
        "uint8_min",
        "uint16_min",
        "uint32_max",
        "int64_max",
    ],
)
def test_a_grouped_reduce_folds_in_its_identitys_dtype(
    op: str, identity: str, init: str, combine: str
) -> None:
    src = f"""
    for vec_lane_1 in cutlass.range_constexpr(4):
        value = values[vec_lane_1]
        shifted = value + bias
        result = _cute_grouped_reduce_warp(shifted, {op!r}, {identity}, lane, pre=2, group_span=32)
        carried = merge(carried, result)
    """
    hoisted = _hoist(src)
    assert f"_helion_vfold_acc_0 = {init}" in hoisted
    assert f"_helion_vfold_acc_0 = {combine}" in hoisted
    dtype = init.split("(", 1)[0]
    reduced = identity.split("(", 1)[0]
    if dtype != reduced and reduced.startswith(("cutlass.Int", "cutlass.Uint")):
        # An integer's extremum, held in a float: the helper takes it
        # converted back, with the identity as the emitter wrote it.
        reduce = f"{reduced}(_helion_vfold_acc_0), {op!r}, {identity}, lane"
    else:
        # The helper's identity is converted to the accumulator's dtype, and
        # left as it is when it has that dtype already.
        folded = identity if dtype == reduced else f"{dtype}({identity})"
        reduce = f"_helion_vfold_acc_0, {op!r}, {folded}, lane"
    assert f"_cute_grouped_reduce_warp({reduce}" in hoisted


@pytest.mark.parametrize(
    ("dtype", "init", "reduced"),
    [
        (
            "cutlass.Float32",
            "cutlass.Float32(float('-inf'))",
            "_helion_vfold_acc_0",
        ),
        (
            "cutlass.Int32",
            "cutlass.Float64(-2147483648)",
            "cutlass.Int32(_helion_vfold_acc_0)",
        ),
        (
            "cutlass.Int64",
            "cutlass.Int64(-9223372036854775808)",
            "_helion_vfold_acc_0",
        ),
    ],
    ids=["float32", "int32", "int64"],
)
def test_the_lane_loop_collapse_knows_an_accumulators_identity(
    dtype: str, init: str, reduced: str
) -> None:
    """The second stage folds a V-fold accumulator across a static lane loop
    and reduces once after it: it recognizes the accumulator by its identity,
    which for an integer maximum is the dtype's lowest value, in the FP64
    accumulator that carries an int32 maximum too."""
    src = """
    for lane in range(2):
        carry_copy = carry
        for vec_lane_1 in cutlass.range_constexpr(4):
            value = values[lane * 4 + vec_lane_1]
            r = cute.arch.warp_reduction_max(value, threads_in_group=32)
            new_carry = cute.math.max(carry_copy, r)
    """
    hoisted = _hoist(
        src, {"value": dtype}, rename_groups={"new_carry": "carry", "carry": "carry"}
    )
    before, _, rest = hoisted.partition("for lane in range(2):")
    assert before.strip() == f"_helion_vfold_acc_0 = {init}"
    lane_loop, _, after = rest.partition("\nr = ")
    assert "warp_reduction_max" not in lane_loop
    assert after.startswith(
        f"cute.arch.warp_reduction_max({reduced}, threads_in_group=32)\n"
    )


def _lane_fold(init: str, reduce_input: str, identity: str) -> str:
    """``fold_lane_reductions`` on two lanes that each fold four elements
    into ``acc`` and reduce it across the threads."""
    body = ast.parse(
        textwrap.dedent(
            f"""
            for lane_1 in cutlass.range_constexpr(2):
                acc = {init}
                for vec_lane_1 in cutlass.range_constexpr(4):
                    acc = cute.math.max(acc, {init.split("(", 1)[0]}(values[lane_1 * 4 + vec_lane_1]), propagate_nan=True)
                red = _cute_grouped_reduce_warp({reduce_input}, 'max', {identity}, lane, pre=1, group_span=64)
                carried = cute.math.max(carried, red)
            """
        )
    ).body
    keys = {
        fold_site_key(node)
        for node in ast.walk(body[0])
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_cute_grouped_reduce_warp"
    }
    counter = iter(range(100))
    folded = fold_lane_reductions(
        body, {"lane_1": keys}, lambda name: f"{name}{next(counter)}"
    )
    return ast.unparse(ast.Module(body=folded, type_ignores=[]))


def test_the_lane_fold_knows_an_accumulator_held_in_a_carrier() -> None:
    """A float accumulator that the reduce converts to the integer whose
    maximum it carries (FP64 for int32, fp32 for int16) starts from that
    integer's bound and is combined in the float.  A conversion at the reduce
    that is not to a carried dtype leaves the checks as they are without
    one; an accumulator that starts from an infinity, or a float that is not
    the integer's carrier, does not start from the fold's identity."""
    lowest = "cutlass.Int32(-2147483648)"
    folded = _lane_fold("cutlass.Float64(-2147483648)", "cutlass.Int32(acc)", lowest)
    assert (
        "_fold_tree_1 = cute.math.max(_fold_parts_0[0], _fold_parts_0[1], "
        "propagate_nan=True)" in folded
    )
    assert folded.count("_cute_grouped_reduce_warp(cutlass.Int32(acc), 'max'") == 1
    short = "cutlass.Int16(-32768)"
    folded = _lane_fold("cutlass.Float32(-32768)", "cutlass.Int16(acc)", short)
    assert "_fold_tree_1 = cute.arch.fmax(_fold_parts_0[0], _fold_parts_0[1]" in folded
    infinity = "cutlass.Float32(float('-inf'))"
    folded = _lane_fold(infinity, "cutlass.Float64(acc)", infinity)
    assert "_fold_tree_1 = cute.arch.fmax(_fold_parts_0[0], _fold_parts_0[1]" in folded
    for init, converted, identity in (
        ("cutlass.Float64(float('-inf'))", "cutlass.Int32(acc)", lowest),
        ("cutlass.Float64(-32768)", "cutlass.Int16(acc)", short),
    ):
        with pytest.raises(exc.BackendUnsupported, match="not the identity of 'max'"):
            _lane_fold(init, converted, identity)


def test_a_grouped_reduce_without_a_typed_identity_keeps_its_loop() -> None:
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        value = values[vec_lane_1]
        shifted = value + bias
        result = _cute_grouped_reduce_warp(shifted, 'sum', zero, lane, pre=2, group_span=32)
        carried = carried + result
    """
    assert _hoist(src) == ast.unparse(ast.parse(textwrap.dedent(src)))
