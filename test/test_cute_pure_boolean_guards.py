from __future__ import annotations

import ast
import itertools
import math
import struct
from types import SimpleNamespace

import pytest

from helion._compiler.cute.affine_vector_io import _conjunction
from helion._compiler.cute.affine_vector_io import _pure_boolean_guard


def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _pair(sources: list[str]) -> tuple[ast.expr, ast.expr]:
    before = ast.BoolOp(op=ast.And(), values=[_expression(part) for part in sources])
    after = _conjunction([_expression(part) for part in sources])
    assert after is not None
    return before, after


def _compile(node: ast.expr):
    return compile(
        ast.fix_missing_locations(ast.Expression(body=node)), "<guard>", "eval"
    )


def _namespace(**values: object) -> dict[str, object]:
    return {
        "cutlass": SimpleNamespace(Int32=int, Float32=float, Boolean=bool),
        **values,
    }


@pytest.mark.parametrize(
    "source",
    [
        "True",
        "False",
        "x < limit",
        "0 <= x < limit",
        "not x",
        "not cutlass.Float32(x)",
        "cutlass.Boolean(x)",
        "x != 0 and 8 // x > 0",
        "x == 0 or 8 // x > 0",
        "cutlass.Int32(row) * cutlass.Int32(x.layout.stride[0]) % 8 == 0",
        "cutlass.Int32(cute.arch.thread_idx()[0]) < limit",
        "cute.arch.block_idx()[2] != 0",
        "(x << 1) & 8 == 0",
    ],
)
def test_supported_guards_are_pure_boolean_scalars(source: str) -> None:
    assert _pure_boolean_guard(_expression(source))


@pytest.mark.parametrize(
    "source",
    [
        "x",
        "1",
        "-0.0",
        "cutlass.Int32(x)",
        "cutlass.Float32(x)",
        "x < limit and 2",
        "x < limit or -0.0",
        "effect() > 0",
        "x.value > 0",
        "x[0] > 0",
        "x.iterator.load() > 0",
        "(x.iterator + cutlass.Int32(0)).load() > 0",
        "cutlass.Int32(x, extra=1) > 0",
        "cutlass.Int32(x, y) > 0",
        "cute.arch.thread_idx(extra=1)[0] > 0",
        "cute.arch.block_idx(0)[0] > 0",
        "cute.arch.thread_idx()[3] > 0",
        "cute.arch.thread_idx()[-1] > 0",
        "x.layout.stride[index] > 0",
        "x.layout.stride[True] > 0",
        "x is y",
        "x in y",
        "(x if enabled else y) > 0",
        "(x := 1) > 0",
    ],
)
def test_numeric_or_unknown_effect_guards_are_not_reassociated(source: str) -> None:
    assert not _pure_boolean_guard(_expression(source))
    before, after = _pair(["first > 0", source, "last > 0"])
    assert ast.dump(before, include_attributes=False) == ast.dump(
        after, include_attributes=False
    )


def test_long_boolean_guards_keep_every_operand_in_order() -> None:
    sources = [f"x{index} != 0" for index in range(14)]
    before, after = _pair(sources)
    leaves = []
    cursor = after
    while isinstance(cursor, ast.BoolOp):
        assert isinstance(cursor.op, ast.And)
        assert len(cursor.values) == 2
        leaves.append(ast.unparse(cursor.values[0]))
        cursor = cursor.values[1]
    leaves.append(ast.unparse(cursor))
    assert leaves == sources
    # Check the complete truth table; no Boolean term is discarded or replaced
    # by a numeric truth value while changing the SDK's expansion structure.
    original, rewritten = _compile(before), _compile(after)
    for bits in itertools.product((0, 1), repeat=len(sources)):
        namespace = {f"x{index}": value for index, value in enumerate(bits)}
        expected = eval(original, namespace)
        actual = eval(rewritten, namespace)
        assert actual is expected
        assert type(actual) is bool


@pytest.mark.parametrize(
    "x", [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -2.0, 2.0]
)
def test_guard_order_preserves_inactive_conversion_and_division(x: float) -> None:
    before, after = _pair(
        [
            "x == x",
            "x < inf",
            "x > negative_inf",
            "cutlass.Int32(x) != 0",
            "8 // cutlass.Int32(x) > 0",
        ]
    )
    namespace = _namespace(x=x, inf=float("inf"), negative_inf=-float("inf"))
    assert eval(_compile(after), namespace) is eval(_compile(before), namespace)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("tail", "x", "exception"),
    [
        ("8 // x > 0", 0, ZeroDivisionError),
        ("cutlass.Int32(x) > 0", float("nan"), ValueError),
        ("cutlass.Int32(x) > 0", float("inf"), OverflowError),
    ],
)
def test_active_arithmetic_errors_and_short_circuit_are_preserved(
    enabled: bool, tail: str, x: float, exception: type[Exception]
) -> None:
    before, after = _pair(["enabled == True", "x != 2", tail])
    namespace = _namespace(enabled=enabled, x=x)
    for node in (before, after):
        if enabled:
            with pytest.raises(exception):
                eval(_compile(node), namespace)
        else:
            assert eval(_compile(node), namespace) is False


@pytest.mark.parametrize("x", [-2, 0, 2])
def test_nested_or_keeps_its_short_circuit_boundary(x: int) -> None:
    before, after = _pair(["x >= 0", "x == 0 or 8 // x > 0", "not x < 0"])
    namespace = _namespace(x=x)
    assert eval(_compile(after), namespace) is eval(_compile(before), namespace)


@pytest.mark.parametrize("value", [2, 2.5, -0.0, 0.0, float("nan")])
def test_numeric_guard_value_and_type_remain_unchanged(value: float) -> None:
    before, after = _pair(["first > 0", "value", "last > 0"])
    assert ast.dump(before) == ast.dump(after)
    namespace = _namespace(first=1, value=value, last=0)
    expected = eval(_compile(before), namespace)
    actual = eval(_compile(after), namespace)
    assert type(actual) is type(expected)
    if isinstance(actual, float):
        assert struct.pack("d", actual) == struct.pack("d", expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("stop", range(4))
def test_unknown_calls_keep_their_original_evaluation_sequence(stop: int) -> None:
    before, after = _pair([f"call({index}) > 0" for index in range(4)])
    assert ast.dump(before) == ast.dump(after)
    traces = []
    for node in (before, after):
        trace: list[int] = []

        def call(index: int, trace: list[int] = trace) -> int:
            trace.append(index)
            return int(index != stop)

        assert eval(_compile(node), {"call": call}) is False
        traces.append(trace)
    assert traces == [list(range(stop + 1))] * 2


@pytest.mark.parametrize("value", [float("nan"), -0.0, 0.0, 2.0])
def test_boolean_conversion_preserves_nonfinite_and_signed_zero(value: float) -> None:
    before, after = _pair(["cutlass.Boolean(value)", "not value == 2", "value != 0"])
    namespace = _namespace(value=value)
    assert eval(_compile(after), namespace) is eval(_compile(before), namespace)
    if math.isnan(value):
        assert eval(_compile(after), namespace) is True


def test_empty_single_and_duplicate_guards_keep_existing_behavior() -> None:
    assert _conjunction([]) is None
    value = _expression("x > 0")
    assert _conjunction([value]) is value
    deduplicated = _conjunction([value, _expression("x > 0"), _expression("y > 0")])
    assert deduplicated is not None
    assert ast.unparse(deduplicated) == "x > 0 and y > 0"
