from __future__ import annotations

import ast
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helion._compiler.cute.full_tile_bounds import FullTilePlan
from helion._compiler.cute.full_tile_bounds import lower_full_tile_bounds
from helion._compiler.cute.full_tile_bounds import prove_full_tile_launch
from helion._compiler.cute.full_tile_bounds import specialize_full_tile_bounds
from helion._compiler.cute.tensor_layout_relations import TensorLayoutRelation


def _plan(
    prefix="n = input_tensor.numel(); flat = input_tensor.view(-1)",
    *,
    grid="((n + BLOCK - 1) // BLOCK,)",
    block="(4, 1, 1)",
    arguments="flat, out, n",
    keyword="block",
    integer_parameters=("n",),
    constants=None,
    rank=1,
):
    body = ast.parse(
        prefix
        + "\nout = torch.empty_like(flat)\nBLOCK = 8\n"
        + f"_launcher(kernel, {grid}, {arguments}, {keyword}={block})\n"
        + "return out\n"
    ).body
    return prove_full_tile_launch(
        body,
        kernel_name="kernel",
        parameter_names=("x", "y", "n"),
        tensor_parameters={"x": rank, "y": rank},
        integer_parameters=integer_parameters,
        tensor_inputs={"input_tensor": 2, "other": 2},
        scalar_inputs={"length", "runtime"},
        constexpr_values={"BLOCK": 8} if constants is None else constants,
    )


@pytest.mark.parametrize(
    "grid",
    [
        "((n + BLOCK - 1) // BLOCK,)",
        "((n + 7) // 8, 1)",
        "(n // (2 * 4), 1, 1)",
        "((7 + n) // BLOCK,)",
        "((n + BLOCK - 1) // BLOCK, BLOCK // 8, 1)",
    ],
)
def test_exact_launch_and_flatten_binding(grid):
    assert _plan(grid=grid) == FullTilePlan(
        TensorLayoutRelation("n", "x", "size", 0), 8, (4, 1, 1), ("x", "y")
    )


def test_actual_argument_positions_and_explicit_metadata_expressions():
    assert _plan(arguments="out, flat, n").relation.tensor == "y"
    assert (
        _plan(arguments="flat, out, flat.size(0)", grid="((flat.size(0) + 7) // 8,)")
        is not None
    )


@pytest.mark.parametrize(
    "grid",
    [
        "((n + BLOCK) // BLOCK,)",
        "((n + BLOCK - 2) // BLOCK,)",
        "((n * 2 + BLOCK - 1) // BLOCK,)",
        "((length + BLOCK - 1) // BLOCK,)",
        "(n / BLOCK,)",
        "((n + BLOCK - 1) // 0,)",
        "((n + BLOCK - 1) // -8,)",
        "((n + BLOCK - 1) // runtime,)",
        "((n + BLOCK - 1) // BLOCK, 2)",
        "((n + BLOCK - 1) // BLOCK, 1, 2)",
        "((n + BLOCK - 1) // BLOCK, runtime)",
        "((n + BLOCK - 1) // BLOCK, 1, 1, 1)",
        "grid_value",
        "[n // BLOCK]",
        "(True,)",
    ],
)
def test_unproved_launch_coverage_declines(grid):
    assert _plan(grid=grid) is None


@pytest.mark.parametrize(
    "block", ["(0, 1, 1)", "(1024, 2, 1)", "(4, 1)", "(4.0, 1, 1)", "threads"]
)
def test_unknown_or_invalid_threads_decline(block):
    assert _plan(block=block) is None


@pytest.mark.parametrize(
    "prefix",
    [
        "n = length; flat = input_tensor.view(-1)",
        "n = other.numel(); flat = input_tensor.view(-1)",
        "n = input_tensor.numel() + 0; flat = input_tensor.view(-1)",
        "n = input_tensor.numel(); flat = input_tensor.view(-1); ignored = effect(flat)",
        "n = input_tensor.numel(); flat = input_tensor.view(-1); input_tensor.resize_(1)",
        "if length:\n    n = input_tensor.numel()\nflat = input_tensor.view(-1)",
    ],
)
def test_independent_length_and_host_effects_decline(prefix):
    assert _plan(prefix) is None


def test_float_or_unknown_extent_and_mismatched_constexpr_decline():
    assert _plan(integer_parameters=()) is None
    assert _plan(constants={"BLOCK": 16}) is None
    assert _plan(constants={"BLOCK": 8.0}) is None
    assert _plan(constants={"BLOCK": -8}) is None
    assert _plan(keyword="cluster") is None
    assert _plan(rank=2) is None


class _Dynamic:
    def __lt__(self, other):
        raise AssertionError("unbaked metadata must short circuit")

    def __gt__(self, other):
        raise AssertionError("unbaked metadata must short circuit")

    def __eq__(self, other):
        raise AssertionError("unbaked metadata must short circuit")


def _tensor(length, stride=1):
    return SimpleNamespace(layout=SimpleNamespace(shape=(length,), stride=(stride,)))


def _sdk(block=0, thread=0):
    return {
        "cutlass": SimpleNamespace(
            Int32=int,
            Int64=int,
            Uint32=int,
            Uint64=int,
            Float32=float,
            const_expr=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            size=lambda tensor, mode: tensor.layout.shape[mode[0]],
            is_static=lambda values: all(type(value) is int for value in values),
            arch=SimpleNamespace(
                block_idx=lambda: (block, 0, 0), thread_idx=lambda: (thread, 0, 0)
            ),
        ),
    }


@pytest.mark.parametrize(
    ("length", "out_length", "stride", "out_stride", "expected"),
    [
        (8, 8, 1, 1, True),
        (1024, 1024, 1, 1, True),
        (((1 << 31) - 1) // 8 * 8, ((1 << 31) - 1) // 8 * 8, 1, 1, True),
        (0, 0, 1, 1, False),
        (1027, 1027, 1, 1, False),
        (1 << 31, 1 << 31, 1, 1, False),
        (-8, -8, 1, 1, False),
        (8, 16, 1, 1, False),
        (16, 8, 1, 1, False),
        (8, 8, 2, 1, False),
        (8, 8, 1, 2, False),
        (8, 8, 0, 1, False),
        (8, 8, -1, 1, False),
        (_Dynamic(), 8, 1, 1, False),
        (8, _Dynamic(), 1, 1, False),
        (8, 8, _Dynamic(), 1, False),
    ],
)
def test_guard_covers_positive_divisible_equal_contiguous_int32_domain(
    length, out_length, stride, out_stride, expected
):
    plan = _plan()
    value = eval(
        compile(
            ast.fix_missing_locations(ast.Expression(plan.predicate())),
            "<guard>",
            "eval",
        ),
        {**_sdk(), "x": _tensor(length, stride), "y": _tensor(out_length, out_stride)},
    )
    assert value is expected


SOURCE = """
pid = cutlass.Int32(cute.arch.block_idx()[0])
thread = cutlass.Int32(cute.arch.thread_idx()[0])
offset = pid * BLOCK
for slot in cutlass.range_constexpr(2):
    index = offset + thread * 2 + slot
    if index < n and index >= 0:
        record(index)
"""


def _rewrite(source, *, renames=None):
    body = ast.parse(source).body
    result = specialize_full_tile_bounds(
        body,
        _plan(),
        argument_names=("x", "y", "n", "runtime", "prefix", "middle"),
        constexpr_values={"BLOCK": 8},
        rename_groups=renames or {},
    )
    return body, result


def _text(body):
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    )


def _execute(body, length, *, block=0, thread=0, **values):
    output = []
    namespace = {
        **_sdk(block, thread),
        "x": _tensor(length),
        "y": _tensor(length),
        "n": length,
        "BLOCK": 8,
        "record": output.append,
        **values,
    }
    exec(compile(_text(body), "<full-tile-oracle>", "exec"), namespace)
    return [(type(value), value, math.copysign(1, value)) for value in output]


@pytest.mark.parametrize("length", [0, 7, 8, 16, 24, 1027])
def test_exact_owned_indices_and_unchanged_tail_fallback(length):
    original, result = _rewrite(SOURCE)
    assert len(result) == 1 and isinstance(result[0], ast.If)
    assert result[0].orelse is original
    assert "if index" not in _text(result[0].body)
    indices = []
    for block in range((length + 7) // 8):
        for thread in range(4):
            expected = _execute(original, length, block=block, thread=thread)
            actual = _execute(result, length, block=block, thread=thread)
            assert actual == expected
            indices.extend(value[1] for value in actual)
    assert sorted(indices) == list(range(length))


@pytest.mark.parametrize("length", [33554432, ((1 << 31) - 1) // 8 * 8])
def test_last_full_tile_coordinates_at_large_int32_extent(length):
    original, result = _rewrite(SOURCE)
    for block in (0, length // 8 - 1):
        for thread in range(4):
            expected = _execute(original, length, block=block, thread=thread)
            actual = _execute(result, length, block=block, thread=thread)
            assert actual == expected
            assert all(0 <= value[1] < length for value in actual)


@pytest.mark.parametrize(
    "expression",
    [
        "cutlass.Float32(offset) + thread",
        "runtime",
        "cutlass.Int32(runtime)",
        "cutlass.Int32(-1)",
        "cutlass.Int64(2147483647) + 1",
        "cutlass.Int32(2147483647) + 1",
        "(offset + 2147483647) - 2147483647",
        "offset << 1",
        "x.iterator.load()",
    ],
)
def test_unknown_float_negative_or_overflowing_intermediates_retain_mask(expression):
    source = (
        SOURCE[: SOURCE.index("for slot")]
        + f"index = {expression}\nrecord(index < n)\n"
    )
    original, result = _rewrite(source)
    assert result is original


@pytest.mark.parametrize(
    "statement",
    [
        "index = runtime",
        "index, unused = runtime, 0",
        "if runtime:\n    index = runtime",
        "for index in unknown_range:\n    pass",
        "for iteration in range(2):\n    index = runtime",
    ],
)
def test_writes_invalidate_previous_integer_facts(statement):
    source = (
        SOURCE[: SOURCE.index("for slot")]
        + "index = offset\n"
        + statement
        + "\nrecord(index < n)\n"
    )
    original, result = _rewrite(source)
    assert result is original


def test_renamed_assignment_and_loop_carried_values_do_not_reuse_stale_facts():
    source = (
        SOURCE[: SOURCE.index("for slot")]
        + "index_old = offset\nindex_new = runtime\nrecord(index_old < n)\n"
    )
    original, result = _rewrite(
        source, renames={"index_old": "index", "index_new": "index"}
    )
    assert result is original
    source = (
        SOURCE[: SOURCE.index("for slot")]
        + "index = offset\nfor iteration in range(2):\n    record(index < n)\n    index = runtime\n"
    )
    original, result = _rewrite(source)
    assert result is original


def test_snapshot_before_reassignment_keeps_its_own_value():
    source = (
        SOURCE[: SOURCE.index("for slot")]
        + "index = offset\nsnapshot = index\nindex = runtime\nrecord(snapshot < n)\n"
    )
    original, result = _rewrite(source)
    assert result is not original
    assert "record(True)" in _text(result[0].body)


@pytest.mark.parametrize(
    "statement",
    [
        "n = 8",
        "x = y",
        "cute = runtime",
        "cutlass = runtime",
        "range = runtime",
        "BLOCK = 8",
        "runtime = 0",
        "index += 1",
        "index: int = 0",
        "while runtime:\n    pass",
        "with runtime:\n    pass",
        "return",
        "index = (runtime := 1)",
        "index = [i for i in range(4)]",
    ],
)
def test_argument_writes_or_unmodelled_control_flow_decline_entire_variant(statement):
    original, result = _rewrite(statement + "\n" + SOURCE)
    assert result is original


@pytest.mark.parametrize(
    "expression",
    [
        "7 and index < n",
        "0 and index < n",
        "prefix and index < n",
        "index < n and prefix",
        "prefix and index < n and True",
        "(prefix or 7) and index < n",
        "(prefix and index < n) or 9",
        "prefix and index < n and middle and index < n",
    ],
)
def test_numeric_short_circuit_values_are_preserved(expression):
    source = SOURCE.replace("record(index)", f"record({expression})")
    original, result = _rewrite(source)
    for prefix in (0, 7, -0.0, 2.5, False, True):
        for middle in (0, 3, False, True):
            assert _execute(original, 8, prefix=prefix, middle=middle) == _execute(
                result, 8, prefix=prefix, middle=middle
            )


def test_disabled_switch_performs_no_analysis_and_returns_same_body():
    body = ast.parse("record(runtime)").body
    function = SimpleNamespace(
        config=SimpleNamespace(config={"cute_proven_bounds": False})
    )
    with patch(
        "helion._compiler.cute.full_tile_bounds.prove_full_tile_launch",
        side_effect=AssertionError("disabled proof called"),
    ):
        assert lower_full_tile_bounds(body, function, (), {}, {}) is body
