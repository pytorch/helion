from __future__ import annotations

import ast
from itertools import combinations
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._compiler import tile_strategy
from helion._compiler.cute.interchanged_store_dce import _instances_never_meet
from helion._compiler.cute.interchanged_store_dce import _serial_span
from helion._compiler.cute.interchanged_store_dce import eliminate_interchanged_stores
from helion._compiler.cute.lane_loops.markers import (
    restore_unprocessed_lane_reduce_markers,
)
from helion._compiler.cute.lane_loops.split import (
    interchange_lane_outside_serial_reductions,
)
from helion._compiler.cute.lane_loops.split import split_lane_loop_reductions
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_fragment_coverage import assert_fragment_norm_coverage
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_PAIRS = {
    frozenset(pair)
    for pair in combinations(("source", "weight", "output", "column"), 2)
}


def _lane_loop(source: str) -> ast.For:
    loop = ast.parse(source).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, "lane")
    return loop


def _program(lanes: str = "range(4)", serial: str = "range(rows)") -> ast.For:
    return _lane_loop(
        f"""
for lane in {lanes}:
    col = lane
    grad_w = 0.0
    for row in {serial}:
        index = row * 4 + col
        mask = row < valid_rows and col < valid_cols
        value = (source.iterator + index).load() if mask else 0.0
        grad_w = grad_w + value
        w = (weight.iterator + col).load()
        product = value * w
        reduced = _helion_lane_reduce(product, 'sum', 0.0, threads_in_group=1)
        scaled = reduced * 2.0
        result = product - scaled
        if mask:
            (output.iterator + index).store(result)
    (column.iterator + col).store(grad_w)
"""
    )


def _interchanged(loop: ast.For) -> tuple[ast.For, ast.For, list[ast.AST], set[int]]:
    """Interchange ``loop`` without eliminating any first-pass store (nor
    declining the nest for an access out of order across the passes, which
    the stores ``eliminate_interchanged_stores`` keeps may be)."""
    with (
        patch(
            "helion._compiler.cute.interchanged_store_dce.eliminate_interchanged_stores",
            return_value=0,
        ),
        patch(
            "helion._compiler.cute.interchanged_store_dce.drop_read_back_partial_stores",
            return_value=None,
        ),
        patch(
            "helion._compiler.cute.interchanged_store_dce.overwrites_later_first_pass_write",
            return_value=False,
        ),
        patch(
            "helion._compiler.cute.interchanged_store_dce.decline_stale_reads",
            return_value=None,
        ),
    ):
        body = interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )
    first = body[0]
    assert isinstance(first, ast.For)
    serial = next(
        statement for statement in first.body if isinstance(statement, ast.For)
    )
    candidates = {
        index
        for index, statement in enumerate(serial.body)
        if "output.iterator" in ast.unparse(statement)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            for node in ast.walk(statement)
        )
    }
    return first, serial, body[1:], candidates


def _prune(
    loop: ast.For,
    pairs: set[frozenset[str]] | None = None,
    protected: set[str] | None = None,
) -> tuple[int, list[ast.AST]]:
    first, serial, final, candidates = _interchanged(loop)
    removed = eliminate_interchanged_stores(
        first,
        serial,
        final,
        candidates,
        _PAIRS if pairs is None else pairs,
        protected or set(),
    )
    return removed, [first, *final]


class _Buffer:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values.copy()
        self.loads = 0
        self.stores = 0

    @property
    def iterator(self) -> _Pointer:
        return _Pointer(self, 0)


class _Pointer:
    def __init__(self, buffer: _Buffer, offset: int) -> None:
        self.buffer = buffer
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.buffer, self.offset + offset)

    def load(self) -> float:
        self.buffer.loads += 1
        return float(self.buffer.values[self.offset])

    def store(self, value: float) -> None:
        self.buffer.stores += 1
        self.buffer.values[self.offset] = value


def _execute(
    body: list[ast.AST],
    rows: int,
    valid_rows: int,
    valid_cols: int,
    inputs: dict[str, np.ndarray] | None = None,
) -> dict[str, _Buffer]:
    """Lower ``body``'s lane reductions and run it on CPU buffers; ``inputs``
    replaces or adds buffers by name."""
    arrays = {
        "source": np.arange(rows * 4, dtype=np.float64) + 0.5,
        "weight": np.array([1.0, 2.0, 4.0, 8.0]),
        "output": np.full(rows * 4, -99.0),
        "column": np.full(4, -99.0),
        **(inputs or {}),
    }
    body = split_lane_loop_reductions(
        body,
        uniform_names={*arrays, "rows", "valid_rows", "valid_cols"},
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    body = restore_unprocessed_lane_reduce_markers(body)
    buffers = {name: _Buffer(values) for name, values in arrays.items()}
    namespace = {
        **buffers,
        "rows": rows,
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
        "cutlass": SimpleNamespace(Float32=float, Int32=int),
        "cute": SimpleNamespace(make_rmem_tensor=lambda shape, dtype: np.zeros(shape)),
    }
    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    exec(compile(ast.unparse(module), "<interchanged>", "exec"), namespace)
    return buffers


def _expected(
    values: np.ndarray,
    rows: int,
    valid_rows: int,
    valid_cols: int,
    output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``_program``'s ``output`` (starting from ``output``) and ``column``
    when it loads ``values``."""
    mask = (np.arange(rows)[:, None] < valid_rows) & (
        np.arange(4)[None, :] < valid_cols
    )
    masked = np.where(mask, values.reshape(rows, 4), 0.0)
    product = masked * np.array([1.0, 2.0, 4.0, 8.0])
    result = product - 2 * product.sum(-1, keepdims=True)
    return np.where(mask, result, output.reshape(rows, 4)), masked.sum(0)


# A reverse scan carried across the lanes visits them in descending order.
@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
@pytest.mark.parametrize(
    ("rows", "valid_rows", "valid_cols"), [(3, 3, 4), (3, 2, 3), (0, 0, 4)]
)
def test_overwritten_stores_and_producers_preserve_cpu_results(
    rows: int, valid_rows: int, valid_cols: int, lanes: str
) -> None:
    original_first, _, original_final, _ = _interchanged(_program(lanes))
    removed, optimized = _prune(_program(lanes))
    assert removed == 1
    before = _execute([original_first, *original_final], rows, valid_rows, valid_cols)
    after = _execute(optimized, rows, valid_rows, valid_cols)
    for name in before:
        np.testing.assert_array_equal(after[name].values, before[name].values)
    assert before["output"].stores == 2 * after["output"].stores
    assert after["output"].stores == valid_rows * valid_cols
    assert before["weight"].loads - after["weight"].loads == rows * 4
    assert before["column"].stores == after["column"].stores == 4
    output, column = _expected(
        after["source"].values, rows, valid_rows, valid_cols, np.full(rows * 4, -99.0)
    )
    np.testing.assert_array_equal(after["output"].values.reshape(rows, 4), output)
    np.testing.assert_array_equal(after["column"].values, column)


def test_disjoint_tensor_names_require_compiler_alias_facts() -> None:
    assert _prune(_program(), pairs=set())[0] == 0


def test_rebound_tensor_name_cannot_reuse_argument_alias_fact() -> None:
    loop = _program()
    loop.body.insert(0, ast.parse("source = output").body[0])
    assert _prune(loop)[0] == 0


def _reading(source: str, serial: str = "range(rows)") -> ast.For:
    """``_program`` with its ``source`` loads reading ``source`` instead."""
    loop = _program(serial=serial)
    loop.body = (
        ast.parse(ast.unparse(loop).replace("source.iterator", f"{source}.iterator"))
        .body[0]
        .body
    )
    return loop


def _first_pass_stores(body: list[ast.AST]) -> int:
    """The stores into ``output[index]`` the interchanged first pass kept."""
    return ast.unparse(body[0]).count("(output.iterator + index).store(")


@pytest.mark.parametrize("source", ["output", "unknown_pointer"])
def test_the_elimination_keeps_a_store_a_load_may_read(source: str) -> None:
    assert _prune(_reading(source))[0] == 0


def test_a_final_pass_reading_the_store_back_drops_it() -> None:
    """The final pass loads the ``output`` elements the first pass stores its
    partial values into, so the interchange drops that store and the final
    pass reads the values the nest started from.  The first pass's own load
    of ``output`` stays: each instance reads the element only it updates."""
    body = interchange_lane_outside_serial_reductions(
        [_reading("output", "range(3)")], proven_disjoint_tensor_pairs=_PAIRS
    )
    assert _first_pass_stores(body) == 0
    values = np.arange(12, dtype=np.float64) + 0.5
    buffers = _execute(body, 3, 2, 3, inputs={"output": values})
    output, column = _expected(values, 3, 2, 3, values)
    np.testing.assert_array_equal(buffers["output"].values.reshape(3, 4), output)
    np.testing.assert_array_equal(buffers["column"].values, column)


def test_a_load_that_only_may_alias_the_store_declines_the_interchange() -> None:
    """``unknown_pointer`` is not proven disjoint from ``output``, so the
    final pass may read the first pass's partial values back: the store is
    dropped, and the first pass's own load of ``unknown_pointer`` may then
    read the value the nest started from where the tile program reads an
    earlier row's update (``unknown_pointer`` a view of ``output`` a row
    behind it)."""
    with pytest.raises(exc.BackendUnsupported, match="dropped partial store"):
        interchange_lane_outside_serial_reductions(
            [_reading("unknown_pointer")], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_an_unresolved_read_of_a_disjoint_tensor_keeps_the_store() -> None:
    """The final pass loads ``weight`` through a local pointer, which names
    no tensor, and the first pass holds a subscript the elimination does not
    model (of a local tuple), so the store cannot be dropped.  That load only
    may read the store: the first pass keeps it, as before, and the final
    pass overwrites it with the right values (``weight`` is not ``output``)."""
    loop = _program()
    loop.body = (
        ast.parse(
            ast.unparse(loop).replace(
                "result = product - scaled",
                "pointer = weight.iterator + col\n"
                "        scales = (1.0, 2.0)\n"
                "        unmodeled = scales[0]\n"
                "        result = product - scaled * pointer.load()",
            )
        )
        .body[0]
        .body
    )
    body = interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    assert _first_pass_stores(body) == 1
    buffers = _execute(body, 3, 2, 3)
    mask = (np.arange(3)[:, None] < 2) & (np.arange(4)[None, :] < 3)
    masked = np.where(mask, buffers["source"].values.reshape(3, 4), 0.0)
    weight = buffers["weight"].values
    product = masked * weight
    result = product - 2 * product.sum(-1, keepdims=True) * weight
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(3, 4), np.where(mask, result, -99.0)
    )
    np.testing.assert_array_equal(buffers["column"].values, masked.sum(0))


def _in_place(lanes: str = "range(4)", after_store: str = "") -> ast.For:
    """``y[i, tile_k, :] = row - row.sum(-1, keepdim=True)`` inside the device
    loop over ``tile_k``, as the lane loop around it renders it."""
    return _lane_loop(
        f"""
for lane in {lanes}:
    seen = 0.0
    for row in range(rows):
        index = row * 4 + lane
        value = (output.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        centered = value - reduced
        (output.iterator + index).store(centered)
        {after_store or "pass"}
    (column.iterator + lane).store(seen)
"""
    )


@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
def test_an_in_place_update_reads_the_values_the_nest_started_from(
    lanes: str,
) -> None:
    """The first pass stored ``row - row`` (its per-lane partial) into the
    rows the final pass then summed, so the update came out as zeros."""
    body = interchange_lane_outside_serial_reductions(
        [_in_place(lanes)], proven_disjoint_tensor_pairs=_PAIRS
    )
    values = np.arange(12, dtype=np.float64) + 0.5
    buffers = _execute(body, 3, 3, 4, inputs={"output": values})
    rows = values.reshape(3, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(3, 4), rows - rows.sum(-1, keepdims=True)
    )
    assert buffers["output"].stores == 12


def test_a_lane_invariant_load_after_the_update_stays_after_it() -> None:
    """``first`` reads the row's first element after the in-place update.
    It reads no lane, so the final nest would run it once per row in its
    serial loop's head, ahead of the lane loop that stores the row, and
    ``source`` would get the element the row started from times ``value``
    instead of the centered one."""
    loop = _in_place(
        after_store="first = (output.iterator + row * 4).load()\n"
        "        (source.iterator + index).store(first * value)"
    )
    with pytest.raises(exc.BackendUnsupported, match="ahead of a write or read"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _centered_then_scaled(
    y: torch.Tensor, w: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    m, n, d = y.shape
    for tile_i in hl.tile(m):
        for tile_k in hl.tile(n):
            row = y[tile_i, tile_k, :].to(torch.float32)
            y[tile_i, tile_k, :] = (row - row.sum(-1, keepdim=True)).to(y.dtype)
            first = w[tile_i, tile_k, 0].to(torch.float32)
            z[tile_i, tile_k, :] = (first[:, :, None] * row).to(z.dtype)
    return z


@pytest.mark.parametrize("aliased", [False, True], ids=["disjoint", "w_is_y"])
def test_a_load_that_may_read_the_update_is_not_hoisted_above_it(
    aliased: bool,
) -> None:
    """The kernel loads ``w[i, k, 0]`` after updating ``y[i, k, :]`` in
    place.  With ``w`` being ``y``, the interchanged nest would load it in
    its serial loop's head, before the lane loop stores the row, so the
    default config is declined; with ``w`` a tensor of its own it renders."""
    y = torch.randn(8, 16, 64)
    w = y if aliased else torch.randn_like(y)
    with cpu_render_target():
        bound = cpu_bind(_centered_then_scaled, (y, w, torch.empty_like(y)))
        if aliased:
            with pytest.raises(
                exc.BackendUnsupported, match="ahead of a write or read"
            ):
                bound.to_code(bound.config_spec.default_config())
        else:
            bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize(
    ("serial", "span"),
    [
        # Casts of known integers they keep are exact.
        ("range(cutlass.Int32(0), cutlass.Int32(512), cutlass.Int32(128))", 511),
        # A 32-bit cast around both bounds of a device loop over its grid tile.
        (
            "range(cutlass.Int32(cutlass.Int32(b) * 384), cutlass.Int32(cutlass.Int32(b) * 384 + 384))",
            383,
        ),
        # The masked tail of a dynamic extent, every value a 32-bit one.
        (
            (
                "range(cutlass.Int32(t), cutlass.Int32(cutlass.Int32(t + 128) if "
                "cutlass.Int32(t + 128) < cutlass.Int32(n) else cutlass.Int32(n)))"
            ),
            127,
        ),
        # The same with an arm that may wrap: at t = 0 and n = -2**32 + 1000
        # the end wraps to 1000, past the arm of 128.
        (
            (
                "range(cutlass.Int32(t), cutlass.Int32(cutlass.Int32(t + 128) if "
                "cutlass.Int32(t + 128) < n else n))"
            ),
            None,
        ),
        # Only one bound narrowed: at n = 2**32 the loop runs at 0 and 2**32.
        ("range(cutlass.Int32(n), n + 1, n)", None),
        # A wrapped descending difference with an ascending step.
        ("range(cutlass.Int32(n + 3), cutlass.Int32(n))", None),
        # A floor division does not commute with narrowing.
        (
            "range(cutlass.Int32(n // 2), cutlass.Int32(cutlass.Int32(n) // 2 + 2))",
            None,
        ),
        # An 8-bit cast inside agrees with the 32-bit one only modulo 2**8.
        ("range(cutlass.Int32(cutlass.Int8(n)), cutlass.Int32(n + 1))", None),
        # A step wrapping to -2**31: at n = 2**31 - 1 the loop visits
        # 2**31 - 1 and -1, which the wrapped bounds do not bound.
        (
            "range(cutlass.Int32(n), cutlass.Int32(n + 3), cutlass.Int32(2147483648))",
            None,
        ),
        # A step wrapping to -1 under a min end: the loop counts down without
        # bound for a negative ``rows``.
        ("range(0, 3 if 3 < rows else rows, cutlass.Int32(4294967295))", None),
    ],
    ids=[
        "constants",
        "both_wrapped",
        "masked_tail",
        "masked_tail_wrapping_arm",
        "begin_only",
        "wrong_way",
        "floordiv",
        "int8",
        "wrapped_step",
        "wrapped_step_min_end",
    ],
)
def test_the_serial_span_accounts_for_narrowing_casts(
    serial: str, span: int | None
) -> None:
    serial_loop = cast("ast.For", ast.parse(f"for row in {serial}:\n    pass").body[0])
    assert _serial_span(serial_loop, [], {"lane"}, set(), None) == span


def test_a_first_pass_rereading_its_update_declines_the_interchange() -> None:
    """A load of the updated element after the store sees the complete value
    in the tile program, and the first pass only ever has the partial one."""
    loop = _in_place(after_store="seen = seen + (output.iterator + index).load()")
    with pytest.raises(exc.BackendUnsupported, match="after the store"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_a_rebound_tensor_name_cannot_hide_a_read_back() -> None:
    """``source = output`` makes the final pass's loads of ``source`` read
    the first pass's partial values of ``output``, whatever the alias facts
    say about the argument ``source``: they may read any tensor, and the
    store they may read back cannot be dropped (the elimination refuses a
    rebound tensor name), so the nest is declined."""
    loop = _in_place()
    loop.body.insert(0, ast.parse("source = output").body[0])
    loop.body = (
        ast.parse(
            ast.unparse(loop).replace(
                "value = (output.iterator + index).load()",
                "value = (source.iterator + index).load()",
            )
        )
        .body[0]
        .body
    )
    with pytest.raises(exc.BackendUnsupported, match="cannot drop"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def _rereading(
    lanes: str, index: str, *, read_back: bool, serial: str = "range(3)"
) -> ast.For:
    """An in-place update of ``output[index]`` whose first pass also adds up,
    per lane, the element it loads ahead of the store.  ``read_back``
    reduces that element (the final pass loads ``output`` too); otherwise
    the update reduces ``source`` and only the first pass loads ``output``."""
    value = "previous" if read_back else "(source.iterator + row * 4 + lane).load()"
    return _lane_loop(
        f"""
for lane in {lanes}:
    seen = 0.0
    for row in {serial}:
        index = {index}
        previous = (output.iterator + index).load()
        seen = seen + previous
        value = {value}
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        centered = value - reduced
        (output.iterator + index).store(centered)
    (column.iterator + lane).store(seen)
"""
    )


# Both lane orders (a reverse scan carried across the lanes visits them
# descending), with the element the first pass loads reduced and read back
# by the final pass, or kept out of it.
_REREADING = [
    pytest.param(lanes, read_back, id=f"{order}-{form}")
    for lanes, order in (("range(4)", "ascending"), ("range(3, -1, -1)", "descending"))
    for read_back, form in ((True, "read_back"), (False, "kept"))
]


@pytest.mark.parametrize(("lanes", "read_back"), _REREADING)
def test_a_first_pass_load_of_its_own_element_ahead_of_the_store_is_exact(
    lanes: str, read_back: bool
) -> None:
    """Each serial iteration updates its own row: the first pass's load
    reads an element no other instance writes, which still holds the value
    the nest started from, whether the partial store goes or stays."""
    body = interchange_lane_outside_serial_reductions(
        [_rereading(lanes, "row * 4 + lane", read_back=read_back)],
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    start = np.arange(12, dtype=np.float64) * 10.0
    buffers = _execute(body, 3, 3, 4, inputs={"output": start})
    previous = start.reshape(3, 4)
    value = previous if read_back else buffers["source"].values.reshape(3, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(3, 4), value - value.sum(-1, keepdims=True)
    )
    np.testing.assert_array_equal(buffers["column"].values, previous.sum(0))


@pytest.mark.parametrize(("lanes", "read_back"), _REREADING)
def test_a_first_pass_load_of_another_iterations_element_declines_the_interchange(
    lanes: str, read_back: bool
) -> None:
    """Every serial iteration updates the same ``output[lane]``: in the tile
    program the first pass's load reads what the previous iteration stored,
    which only the final pass computes.  The first pass would read the
    starting value (the store dropped) or the partial one (the store kept)."""
    with pytest.raises(exc.BackendUnsupported, match="another serial iteration"):
        interchange_lane_outside_serial_reductions(
            [_rereading(lanes, "lane", read_back=read_back)],
            proven_disjoint_tensor_pairs=_PAIRS,
        )


def test_a_first_pass_load_of_its_own_element_in_a_transposed_layout_is_exact() -> None:
    """``output[row + 3 * lane]`` over 3 rows: one lane step outweighs every
    serial difference, so each instance still reads an element only it
    writes, as a serial step outweighing every lane difference does in a
    row-major layout."""
    body = interchange_lane_outside_serial_reductions(
        [_rereading("range(4)", "row + 3 * lane", read_back=False)],
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    start = np.arange(12, dtype=np.float64) * 10.0
    buffers = _execute(body, 3, 3, 4, inputs={"output": start})
    value = buffers["source"].values.reshape(3, 4)
    centered = value - value.sum(-1, keepdims=True)
    np.testing.assert_array_equal(buffers["output"].values.reshape(4, 3), centered.T)
    np.testing.assert_array_equal(buffers["column"].values, start.reshape(4, 3).sum(1))


# A device loop over its grid tile's masked tail ends at ``min(offset +
# block, size)``, generated as either form of a conditional expression, and
# steps by a known positive integer.
@pytest.mark.parametrize(
    "serial",
    [
        "range(0, 3 if 3 < rows else rows)",
        "range(0, 3 if rows > 3 else rows)",
        "range(0, 3 if 3 < rows else rows, cutlass.Int32(1))",
    ],
    ids=["less", "greater", "step"],
)
def test_a_serial_range_ending_at_a_min_spans_its_known_arm(serial: str) -> None:
    """The serial variable stays below either arm of the ``min``, so the
    arm three rows past the start bounds the span (``rows`` a dynamic
    extent), and the first pass's load of its own element is exact."""
    body = interchange_lane_outside_serial_reductions(
        [_rereading("range(4)", "row * 4 + lane", read_back=False, serial=serial)],
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    start = np.arange(12, dtype=np.float64) * 10.0
    buffers = _execute(body, 3, 3, 4, inputs={"output": start})
    value = buffers["source"].values.reshape(3, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(3, 4), value - value.sum(-1, keepdims=True)
    )
    np.testing.assert_array_equal(buffers["column"].values, start.reshape(3, 4).sum(0))


# The serial loop's span unknown (``range(rows)``, one ending at a ``max``,
# or a descending one ending at a ``min``, which it stays above: with
# ``rows`` 0 it visits rows 3, 2 and 1 of ``output[lane]``), a serial range
# moving with the lane (every lane updates ``output[3:6]``), an index whose
# Int32 cast wraps the serial step (2**32) to nothing, and an index rebased
# by a name the serial loop carries (every iteration lands on
# ``output[lane]``).
_UNPROVEN_OWN_ELEMENTS = [
    pytest.param("row * 4 + lane", "range(rows)", "", id="unknown_span"),
    pytest.param(
        "row * 4 + lane", "range(0, 3 if 3 > rows else rows)", "", id="max_end"
    ),
    pytest.param(
        "lane", "range(3, 3 if 3 < rows else rows, -1)", "", id="descending_min_end"
    ),
    pytest.param(
        "row + 3 * lane", "range(3 - 3 * lane, 6 - 3 * lane)", "", id="lane_range"
    ),
    pytest.param("cutlass.Int32(row * 4294967296) + lane", "range(3)", "", id="wrap"),
    pytest.param(
        "row * 4 + lane + shift", "range(3)", "shift = shift - 4", id="carried"
    ),
]


@pytest.mark.parametrize(("index", "serial", "rebase"), _UNPROVEN_OWN_ELEMENTS)
def test_a_first_pass_load_of_an_element_not_proven_its_own_declines_the_interchange(
    index: str, serial: str, rebase: str
) -> None:
    """The proof that the first pass's load reads only its own instance's
    element must hold modulo 2**32 over the serial loop's whole span, the
    same in every lane, with the names it inlines bound once: here it
    cannot, and the nest is declined rather than let the load read another
    instance's update."""
    loop = _rereading("range(4)", index, read_back=False, serial=serial)
    if rebase:
        loop.body.insert(1, ast.parse("shift = 0").body[0])
        serial_loop = cast("ast.For", loop.body[2])
        serial_loop.body.append(ast.parse(rebase).body[0])
    with pytest.raises(exc.BackendUnsupported, match="interchanged lane reduction"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_a_load_in_a_branch_rebinding_its_address_is_not_proven_its_own() -> None:
    """The branch rebinds ``index`` to the previous row's element before its
    load, which the definitions reaching the branch would take for the
    store's own element.  Only a load in a plain assignment counts: this one
    reads what the dropped store would hold, and the nest is declined (here
    the elimination refuses the branch itself)."""
    loop = _lane_loop(
        """
for lane in range(4):
    seen = 0.0
    for row in range(3):
        index = row * 4 + lane
        previous = 0.0
        if row > 0:
            index = (row - 1) * 4 + lane
            previous = (output.iterator + index).load()
        seen = seen + previous
        value = (source.iterator + row * 4 + lane).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        centered = value - reduced
        dest = row * 4 + lane
        (output.iterator + dest).store(centered)
    (column.iterator + lane).store(seen)
"""
    )
    with pytest.raises(exc.BackendUnsupported, match="cannot drop"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_a_serial_range_awaiting_a_rename_leaves_the_own_element_unproven() -> None:
    """``range(start, start + 3)`` spans 3 rows only while ``start`` keeps
    its meaning, which a pending rename may change."""
    loop = _rereading(
        "range(4)", "row * 4 + lane", read_back=False, serial="range(start, start + 3)"
    )
    with pytest.raises(exc.BackendUnsupported, match="interchanged lane reduction"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS, protected_names={"start"}
        )


@pytest.mark.parametrize(
    ("serial_step", "serial_span", "lane_step", "lane_span", "never_meet"),
    [
        (4, 2, 1, 3, True),  # row-major: a row outweighs every lane difference
        (1, 2, 3, 3, True),  # transposed: a lane outweighs every row difference
        (4, 2, 6, 2, True),  # mixed radix: (6, -4) / 2 leaves the serial span
        (2, 2, 1, 3, False),  # row 1, lane 0 meets row 0, lane 2
        (1 << 32, 2, 1, 3, False),  # the serial step wraps to nothing
        (1 << 31, 2, 1, 3, False),  # two serial steps reach 2**32
        (4, 0, 0, 0, True),  # one instance
    ],
)
def test_instances_meet_modulo_two_to_the_32(
    serial_step: int, serial_span: int, lane_step: int, lane_span: int, never_meet: bool
) -> None:
    assert (
        _instances_never_meet(serial_step, serial_span, lane_step, lane_span)
        is never_meet
    )


def _scaled_by_prefix(lanes: str, source: str, *, pointer: bool = False) -> ast.For:
    """An in-place update of ``output`` whose reduction scales each row by a
    per-lane value the lane loop's prefix loads from ``source``, with
    ``pointer`` through a local pointer whose address names no tensor."""
    load = (
        f"pointer = {source}.iterator + lane\n    scale = pointer.load()"
        if pointer
        else f"scale = ({source}.iterator + lane).load()"
    )
    return _lane_loop(
        f"""
for lane in {lanes}:
    {load}
    for row in range(rows):
        index = row * 4 + lane
        value = (output.iterator + index).load()
        product = value * scale
        reduced = _helion_lane_reduce(product, 'sum', 0.0, threads_in_group=1)
        centered = value - reduced
        (output.iterator + index).store(centered)
"""
    )


@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
def test_a_prefix_load_the_final_pass_reruns_reads_an_unwritten_tensor(
    lanes: str,
) -> None:
    """The final pass reloads the prefix's ``weight`` in every serial
    iteration; nothing writes it, so every reload sees what the prefix saw."""
    body = interchange_lane_outside_serial_reductions(
        [_scaled_by_prefix(lanes, "weight")], proven_disjoint_tensor_pairs=_PAIRS
    )
    values = np.arange(12, dtype=np.float64) + 0.5
    buffers = _execute(body, 3, 3, 4, inputs={"output": values})
    rows = values.reshape(3, 4)
    scaled = rows * buffers["weight"].values
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(3, 4),
        rows - scaled.sum(-1, keepdims=True),
    )


@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
def test_a_prefix_load_of_the_updated_tensor_declines_the_interchange(
    lanes: str,
) -> None:
    """The tile program loads row 0 of ``output`` once, before the loop; the
    final pass would reload it in every serial iteration, after the first
    one updated it."""
    with pytest.raises(exc.BackendUnsupported, match="lane loop's prefix"):
        interchange_lane_outside_serial_reductions(
            [_scaled_by_prefix(lanes, "output")], proven_disjoint_tensor_pairs=_PAIRS
        )


@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
@pytest.mark.parametrize("source", ["weight", "output"])
def test_a_prefix_load_through_a_local_pointer_is_resolved(
    lanes: str, source: str
) -> None:
    """The check resolves ``pointer.load()`` through the pointer's
    definition: rerunning it stands for ``weight``, which nothing writes,
    and declines the nest for ``output``, where it reloaded the first
    serial iteration's update."""
    loop = _scaled_by_prefix(lanes, source, pointer=True)
    if source == "weight":
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )
        return
    with pytest.raises(exc.BackendUnsupported, match="lane loop's prefix"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
def test_a_prefix_load_of_another_lanes_element_reads_the_starting_value(
    lanes: str,
) -> None:
    """The prefix loads, for the first pass alone, the element of ``output``
    the mirror lane updates in the first row.  The tile program's prefix
    reads the values the nest started from, where the first pass's later
    lanes read the partial values earlier lanes stored: the store goes."""
    loop = _lane_loop(
        f"""
for lane in {lanes}:
    mirrored = (output.iterator + (3 - lane)).load()
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        centered = value - reduced
        (output.iterator + index).store(centered)
    (column.iterator + lane).store(mirrored)
"""
    )
    body = interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    start = np.arange(12, dtype=np.float64) * 10.0
    buffers = _execute(body, 3, 3, 4, inputs={"output": start})
    value = buffers["source"].values.reshape(3, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(3, 4), value - value.sum(-1, keepdims=True)
    )
    np.testing.assert_array_equal(buffers["column"].values, start[[3, 2, 1, 0]])


def _center_rows_in_place(y: torch.Tensor) -> torch.Tensor:
    for tile_i in hl.tile(y.size(0)):
        for tile_k in hl.tile(y.size(1)):
            row = y[tile_i, tile_k, :]
            y[tile_i, tile_k, :] = row - row.sum(-1, keepdim=True)
    return y


def _softmax_rows_in_place(y: torch.Tensor) -> torch.Tensor:
    for tile_i in hl.tile(y.size(0)):
        for tile_k in hl.tile(y.size(1)):
            y[tile_i, tile_k, :] = torch.softmax(y[tile_i, tile_k, :], -1)
    return y


def _center_rows_and_sum_columns(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rms_norm_bwd``'s shape, in place: each row block's per-column sum
    of the rows it loads (lanes outside the row loop) beside each row
    centered (a reduction over the lanes, which is interchanged)."""
    m_block = hl.register_block_size(x.size(0))
    sums = x.new_empty(
        [(x.size(0) + m_block - 1) // m_block, x.size(1)], dtype=torch.float32
    )
    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros([x.size(1)], dtype=torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            row = x[mb, :].to(torch.float32)
            acc += row.sum(0)
            x[mb, :] = (row - row.mean(-1, keepdim=True)).to(x.dtype)
        sums[mb_cta.id, :] = acc
    return x, sums


def _recenter_rows(x: torch.Tensor, passes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every pass re-centers the same rows in place and adds up what it read."""
    totals = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m, x.size(1)], dtype=torch.float32)
        for _ in hl.grid(passes):
            row = x[tile_m, :]
            acc = acc + row
            x[tile_m, :] = row - row.mean(-1, keepdim=True)
        totals[tile_m, :] = acc
    return x, totals


def _center_other_rows_and_sum_columns(
    x: torch.Tensor, src: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_center_rows_and_sum_columns`` storing ``src``'s rows centered, with
    ``x``'s width specialized for dynamic shapes: only the first nest loads
    ``x``, the rows it sums, ahead of the update."""
    n = hl.specialize(x.size(1))
    m_block = hl.register_block_size(x.size(0))
    sums = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros([n], dtype=torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            row = x[mb, :].to(torch.float32)
            acc += row.sum(0)
            other = src[mb, :].to(torch.float32)
            x[mb, :] = (other - other.mean(-1, keepdim=True)).to(x.dtype)
        sums[mb_cta.id, :] = acc
    return x, sums


def _default_code(
    fn: object,
    args: tuple[object, ...],
    *,
    static_shapes: bool = True,
    **overrides: object,
) -> str:
    """``fn``'s CuTe kernel under its default config with ``overrides``,
    rendered on CPU."""
    with (
        cpu_render_target(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=128 * 1024,
        ),
    ):
        kernel = helion.kernel(
            fn, backend="cute", autotune_effort="none", static_shapes=static_shapes
        )
        bound = kernel.bind(args)
        config = bound.config_spec.default_config()
        return bound.to_code(helion.Config(**{**config.config, **overrides}))


def _top_loops(code: str) -> list[ast.For]:
    """The loops at the top of the generated device kernel."""
    kernel = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    return [node for node in kernel.body if isinstance(node, ast.For)]


def _accesses(node: ast.AST, tensor: str) -> tuple[int, int]:
    """The loads and the stores of ``tensor`` in ``node``."""
    calls = [
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("load", "store")
        and f"{tensor}.iterator" in ast.unparse(call.func.value)
    ]
    return calls.count("load"), calls.count("store")


def test_an_in_place_row_update_keeps_no_first_pass_store() -> None:
    """Under the default config one thread walks a row's 128 features in a
    lane loop around the ``tile_k`` loop, which is interchanged; the first
    nest's partial store of ``row - row`` into ``y`` goes, and with it the
    whole first nest, left with nothing to do."""
    code = _default_code(_center_rows_in_place, (torch.randn(64, 32, 128),))
    assert code.count("for synthetic_lane_2 in range(128):") == 2, code
    (final,) = _top_loops(code)
    # The final nest's accumulate load and its consume load and store.
    assert _accesses(final, "y") == (2, 1), code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_an_in_place_row_update_matches_eager() -> None:
    y = torch.randn(64, 32, 128, device=DEVICE)
    expected = y - y.sum(-1, keepdim=True)
    kernel = helion.kernel(
        _center_rows_in_place, backend="cute", autotune_effort="none"
    )
    code, out = code_and_output(kernel, (y,))
    assert code.count("for synthetic_lane_2 in range(128):") == 2
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_an_in_place_softmax_keeps_no_first_pass_store() -> None:
    """The softmax's max and sum over a row it then stores back in place: as
    for the centering, the first nest's partial store goes, and with it the
    whole first nest."""
    code = _default_code(_softmax_rows_in_place, (torch.randn(64, 32, 128),))
    (final,) = _top_loops(code)
    # The max's, the sum's and the consumer's loads and the one store.
    assert _accesses(final, "y") == (3, 1), code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_an_in_place_softmax_matches_eager() -> None:
    y = torch.randn(64, 32, 128, device=DEVICE)
    expected = torch.softmax(y, -1)
    kernel = helion.kernel(
        _softmax_rows_in_place, backend="cute", autotune_effort="none"
    )
    _, out = code_and_output(kernel, (y,))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_an_in_place_update_beside_a_column_sum_keeps_its_own_rows_load() -> None:
    """The first nest keeps loading the rows it sums ahead of the update:
    each iteration of the row loop loads only the rows it updates, so the
    partial store still goes."""
    code = _default_code(_center_rows_and_sum_columns, (torch.randn(256, 1024),))
    first, final = _top_loops(code)
    assert _accesses(first, "x") == (1, 0), code
    assert _accesses(final, "x") == (2, 1), code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_an_in_place_update_beside_a_column_sum_matches_eager() -> None:
    x = torch.randn(256, 1024, device=DEVICE)
    expected = x - x.mean(-1, keepdim=True)
    kernel = helion.kernel(
        _center_rows_and_sum_columns, backend="cute", autotune_effort="none"
    )
    code, (out, sums) = code_and_output(kernel, (x.clone(),))
    first, _final = _top_loops(code)
    assert _accesses(first, "x") == (1, 0), code
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        sums, x.reshape(sums.size(0), -1, x.size(1)).sum(1), rtol=1e-4, atol=1e-4
    )


def test_an_own_rows_load_under_a_dynamic_row_loop_is_exact() -> None:
    """With dynamic shapes the row loop over a block's masked tail ends at
    ``min(offset + block, size)``, which the block size bounds: the first
    nest's load of the rows it updates is still proven its own, and its
    partial store stays for the final nest to overwrite."""
    code = _default_code(
        _center_other_rows_and_sum_columns,
        (torch.randn(256, 1024), torch.randn(256, 1024)),
        static_shapes=False,
    )
    first, final = _top_loops(code)
    assert _accesses(first, "x") == (1, 1), code
    assert _accesses(final, "x") == (0, 1), code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_an_own_rows_load_under_a_dynamic_row_loop_matches_eager() -> None:
    x = torch.randn(256, 1024, device=DEVICE)
    src = torch.randn(256, 1024, device=DEVICE)
    kernel = helion.kernel(
        _center_other_rows_and_sum_columns,
        backend="cute",
        autotune_effort="none",
        static_shapes=False,
    )
    code, (out, sums) = code_and_output(kernel, (x.clone(), src))
    assert len(_top_loops(code)) == 2, code
    torch.testing.assert_close(
        out, src - src.mean(-1, keepdim=True), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        sums, x.reshape(sums.size(0), -1, x.size(1)).sum(1), rtol=1e-4, atol=1e-4
    )


def test_an_in_place_update_of_the_same_rows_in_every_pass_is_refused() -> None:
    """Every pass of the loop loads the rows the previous one updated: the
    first nest's load would read the values the kernel started from."""
    with pytest.raises(exc.BackendUnsupported, match="another serial iteration"):
        _default_code(_recenter_rows, (torch.randn(64, 128), 3))


@pytest.mark.parametrize("replacement", ["index + reduced", "row + col", "index + 1"])
def test_changed_final_address_rejects_overwrite(replacement: str) -> None:
    first, serial, final, candidates = _interchanged(_program())
    last = cast("ast.For", final[-1])
    lane = cast("ast.For", last.body[-1])
    lane.body = (
        ast.parse(
            ast.unparse(lane).replace(
                "output.iterator + index", f"output.iterator + ({replacement})"
            )
        )
        .body[0]
        .body
    )
    assert (
        eliminate_interchanged_stores(first, serial, final, candidates, _PAIRS, set())
        == 0
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("index = row * 4 + col", "index = row * 4 + col + 1"),
        (
            "mask = row < valid_rows and col < valid_cols",
            "mask = row < valid_rows and col < valid_cols - 1",
        ),
        ("col = lane", "col = lane + 1"),
    ],
)
def test_matching_store_text_requires_matching_address_and_mask_definitions(
    before: str, after: str
) -> None:
    first, serial, final, candidates = _interchanged(_program())
    last = cast("ast.For", final[-1])
    lane = cast("ast.For", last.body[-1])
    lane.body = ast.parse(ast.unparse(lane).replace(before, after)).body[0].body
    assert (
        eliminate_interchanged_stores(first, serial, final, candidates, _PAIRS, set())
        == 0
    )


def test_final_pass_load_that_observes_first_store_rejects_elimination() -> None:
    first, serial, final, candidates = _interchanged(_program())
    last = cast("ast.For", final[-1])
    last.body.insert(
        0, ast.parse("observed = (output.iterator + row * 4).load()").body[0]
    )
    assert (
        eliminate_interchanged_stores(first, serial, final, candidates, _PAIRS, set())
        == 0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "mask = reduced > 0",
        "index = int(reduced)",
        "index = cursor\n        cursor = cursor + 1",
        "opaque(output)",
        "if row == 1:\n            break",
        "result = output[index] + result",
    ],
)
def test_unproved_coverage_or_effect_preserves_store(mutation: str) -> None:
    loop = _program()
    source = ast.unparse(loop).replace("if mask:\n", f"{mutation}\n        if mask:\n")
    loop.body = ast.parse(source).body[0].body
    assert _prune(loop)[0] == 0


def test_lane_dependent_serial_range_preserves_store() -> None:
    loop = _program()
    serial = cast("ast.For", loop.body[2])
    serial.iter = ast.parse("range(lane + 1)", mode="eval").body
    assert _prune(loop)[0] == 0


def test_pending_scalar_alias_keeps_producer_binding() -> None:
    removed, body = _prune(_program(), protected={"scaled", "live_alias"})
    assert removed == 1
    first = cast("ast.For", body[0])
    assert "scaled = reduced * 2.0" in ast.unparse(first)


def test_pending_scalar_alias_in_address_preserves_store() -> None:
    assert _prune(_program(), protected={"index", "other_index"})[0] == 0


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
def test_norm_backward_emits_one_dx_store(name: str) -> None:
    layer_norm_bwd = import_example("layer_norm").layer_norm_bwd
    rms_norm_bwd = import_example("rms_norm").rms_norm_bwd

    x = torch.empty(256, 1024, dtype=torch.float16)
    weight = torch.empty(1024, dtype=torch.float16)
    if name == "rms_norm":
        example = rms_norm_bwd
        inputs = (torch.empty_like(x), x, weight, torch.empty(256, 1))
    else:
        example = layer_norm_bwd
        inputs = (
            torch.empty_like(x),
            x,
            torch.empty(256),
            torch.empty(256),
            weight,
            True,
        )
    bound = helion.kernel(
        example.fn,
        backend="cute",
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ).bind(inputs)
    with patch(
        "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        code = bound.to_code(bound.env.config_spec.default_config())
        with patch(
            "helion._compiler.cute.interchanged_store_dce.eliminate_interchanged_stores",
            return_value=0,
        ):
            original = bound.to_code(bound.env.config_spec.default_config())

    def stores(source: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            and "grad_x.iterator" in ast.unparse(node.func.value)
            for node in ast.walk(ast.parse(source))
        )

    if "fragment_buffer" in code:
        assert name == "layer_norm"
        # The complete fragment owner already emits one write per element;
        # it does not pass through the scalar interchange/DCE rewrite.
        assert stores(original) == stores(code) == 1
        assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(code))
        assert_fragment_norm_coverage(code, 256, 1024)
        return

    assert stores(original) == 2
    assert stores(code) == 1
    first = next(
        node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.For)
    )
    if name == "rms_norm":
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
            and node.value.id == "weight"
            for node in ast.walk(first)
        )
    assert "grad_weight" in ast.unparse(first)


def _atomic_program(
    *, row_total: bool = True, before: str = "", extra: str = ""
) -> ast.For:
    """``_program`` with the reduction added into ``output[row]`` by an atomic
    instead of stored per element; ``row_total=False`` drops the per-lane
    ``column`` accumulator so the reduction is the loop's only effect.
    ``before`` goes ahead of the atomic's ``if``, ``extra`` after the atomic."""
    init = (
        """
    grad_w = 0.0"""
        if row_total
        else ""
    )
    total = (
        """
        grad_w = grad_w + value"""
        if row_total
        else ""
    )
    suffix = (
        """
    (column.iterator + col).store(grad_w)"""
        if row_total
        else ""
    )
    return _lane_loop(
        f"""
for lane in range(4):
    col = lane{init}
    for row in range(rows):
        row_mask = row < valid_rows
        mask = row_mask and col < valid_cols
        value = (source.iterator + row * 4 + col).load() if mask else 0.0{total}
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1){before}
        if row_mask:
            cute.arch.atomic_add((output.iterator + row).llvm_ptr, val=reduced, sem='relaxed'){extra}{suffix}
"""
    )


def _atomics(body: list[ast.AST]) -> list[ast.Call]:
    return [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_add"
    ]


class _AtomicBuffer(_Buffer):
    @property
    def iterator(self) -> _AtomicPointer:
        return _AtomicPointer(self, 0)


class _AtomicPointer(_Pointer):
    def __add__(self, offset: int) -> _AtomicPointer:
        return _AtomicPointer(self.buffer, self.offset + offset)

    @property
    def llvm_ptr(self) -> _AtomicPointer:
        return self

    def atomic_add(self, value: float) -> None:
        self.buffer.stores += 1
        self.buffer.values[self.offset] += value


def _execute_atomics(
    body: list[ast.AST],
    rows: int,
    valid_rows: int,
    valid_cols: int,
    output: np.ndarray | None = None,
) -> dict[str, _AtomicBuffer]:
    body = split_lane_loop_reductions(
        body,
        uniform_names={
            "source",
            "output",
            "column",
            "rows",
            "valid_rows",
            "valid_cols",
        },
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    body = restore_unprocessed_lane_reduce_markers(body)
    buffers = {
        "source": _AtomicBuffer(np.arange(rows * 4, dtype=np.float64) + 0.5),
        "output": _AtomicBuffer(np.zeros(rows) if output is None else output),
        "column": _AtomicBuffer(np.full(4, -99.0)),
    }
    namespace = {
        **buffers,
        "rows": rows,
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
        "cutlass": SimpleNamespace(Float32=float, Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                atomic_add=lambda pointer, val, sem: pointer.atomic_add(val)
            )
        ),
    }
    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    exec(compile(ast.unparse(module), "<interchanged>", "exec"), namespace)
    return buffers


@pytest.mark.parametrize(
    ("rows", "valid_rows", "valid_cols"), [(3, 3, 4), (3, 2, 3), (0, 0, 4)]
)
def test_an_atomic_consumer_of_the_reduction_is_applied_by_the_final_pass_only(
    rows: int, valid_rows: int, valid_cols: int
) -> None:
    """The first pass keeps its per-lane ``column`` totals; its atomic of the
    reduction's per-lane partial is dropped (a store there is overwritten by
    the final pass, an add would double the total) and the final pass adds
    each row's full reduction once.  Its loads of ``source`` and its atomics
    into ``output`` come after the ``column`` stores, which the alias facts
    keep apart."""
    body = interchange_lane_outside_serial_reductions(
        [_atomic_program()], proven_disjoint_tensor_pairs=_PAIRS
    )
    first, *final = body
    assert isinstance(first, ast.For) and first.target.id == "lane"  # pyrefly: ignore [missing-attribute]
    assert not _atomics([first])
    assert len(_atomics(final)) == 1
    assert "reduced" not in ast.unparse(first)
    assert "grad_w = grad_w + value" in ast.unparse(first)
    buffers = _execute_atomics(body, rows, valid_rows, valid_cols)
    values = buffers["source"].values.reshape(rows, 4)
    mask = (np.arange(rows)[:, None] < valid_rows) & (
        np.arange(4)[None, :] < valid_cols
    )
    masked = np.where(mask, values, 0.0)
    np.testing.assert_array_equal(buffers["output"].values, masked.sum(1))
    assert buffers["output"].stores == valid_rows
    np.testing.assert_array_equal(buffers["column"].values, masked.sum(0))


def _atomic_reading(site: str, address: str, destination: str = "row") -> ast.For:
    """``_atomic_program`` adding the reduction into ``output[destination]``,
    whose per-lane ``grad_w`` also adds up ``output[address]``, loaded in the
    lane loop's ``prefix``, in the serial loop's ``body`` ahead of the atomic
    or in the ``suffix`` after it."""
    read = f"grad_w = grad_w + (output.iterator + {address}).load()"
    prefix, body, suffix = (
        f"\n{indent}{read}" if site == name else ""
        for name, indent in (
            ("prefix", " " * 4),
            ("body", " " * 8),
            ("suffix", " " * 4),
        )
    )
    return _lane_loop(
        f"""
for lane in range(4):
    col = lane
    grad_w = 0.0{prefix}
    for row in range(rows):
        row_mask = row < valid_rows
        mask = row_mask and col < valid_cols
        value = (source.iterator + row * 4 + col).load() if mask else 0.0
        grad_w = grad_w + value{body}
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        if row_mask:
            cute.arch.atomic_add((output.iterator + {destination}).llvm_ptr, val=reduced, sem='relaxed'){suffix}
    (column.iterator + col).store(grad_w)
"""
    )


# In the tile program each pass's load reads what the atomics of the passes
# before it added (the same element in every pass, or each lane its own
# column), and a load after the serial loop reads what all of them added.
@pytest.mark.parametrize(
    ("site", "address", "destination"),
    [("body", "0", "0"), ("body", "col", "col"), ("suffix", "0", "row")],
    ids=["running_total", "running_column", "after_the_loop"],
)
def test_a_first_pass_load_of_what_the_atomic_updates_declines_the_interchange(
    site: str, address: str, destination: str
) -> None:
    """The first pass's copy of the atomic is dropped, and the final pass
    applies it only after the whole first pass: the first pass's load would
    read the value the nest started from."""
    with pytest.raises(exc.BackendUnsupported, match="atomic of the final pass"):
        interchange_lane_outside_serial_reductions(
            [_atomic_reading(site, address, destination)],
            proven_disjoint_tensor_pairs=_PAIRS,
        )


def test_a_prefix_load_of_what_the_atomic_updates_reads_the_starting_value() -> None:
    """The lane loop's prefix runs before every serial iteration in the tile
    program too, so its load of ``output`` keeps the value the nest started
    from."""
    body = interchange_lane_outside_serial_reductions(
        [_atomic_reading("prefix", "0")], proven_disjoint_tensor_pairs=_PAIRS
    )
    start = np.array([10.0, 20.0, 30.0])
    buffers = _execute_atomics(body, 3, 3, 4, output=start)
    values = buffers["source"].values.reshape(3, 4)
    np.testing.assert_array_equal(buffers["output"].values, start + values.sum(1))
    np.testing.assert_array_equal(buffers["column"].values, start[0] + values.sum(0))


def test_a_first_pass_whose_only_effect_was_the_atomic_is_dropped() -> None:
    body = interchange_lane_outside_serial_reductions(
        [_atomic_program(row_total=False)]
    )
    assert len(body) == 1 and isinstance(body[0], ast.For)
    assert body[0].target.id == "row"  # pyrefly: ignore [missing-attribute]
    assert len(_atomics(body)) == 1
    buffers = _execute_atomics(body, 3, 2, 3)
    values = buffers["source"].values.reshape(3, 4)
    expected = np.where(np.arange(4)[None, :] < 3, values, 0.0).sum(1)
    expected[2:] = 0.0
    np.testing.assert_array_equal(buffers["output"].values, expected)


def test_an_atomic_statement_with_other_work_declines_the_interchange() -> None:
    """Dropping the statement from the first pass would drop its other
    effect too, and keeping it would double the total: decline the nest."""
    with pytest.raises(exc.BackendUnsupported, match="atomic consumer"):
        interchange_lane_outside_serial_reductions(
            [_atomic_program(extra="\n            seen = reduced")]
        )


_COLUMN_ATOMIC = """
        cute.arch.atomic_add((column.iterator + col).llvm_ptr, val=value, sem='relaxed')"""


def test_an_atomic_both_nests_need_is_applied_once() -> None:
    """``column[col] += value`` reads the reduction's input, not its result:
    the final pass needs it too, and the first pass's copy doubled
    ``column``.  That copy goes, and with it the whole first pass."""
    body = interchange_lane_outside_serial_reductions(
        [_atomic_program(row_total=False, before=_COLUMN_ATOMIC)],
        proven_disjoint_tensor_pairs=_PAIRS,
    )
    assert len(body) == 1
    assert len(_atomics(body)) == 2
    buffers = _execute_atomics(body, 3, 2, 3)
    mask = (np.arange(3)[:, None] < 2) & (np.arange(4)[None, :] < 3)
    masked = np.where(mask, buffers["source"].values.reshape(3, 4), 0.0)
    np.testing.assert_array_equal(buffers["output"].values, masked.sum(1))
    np.testing.assert_array_equal(buffers["column"].values, masked.sum(0) - 99.0)


@pytest.mark.parametrize(
    "access",
    [
        "seen = (column.iterator + col).load()",
        # A subscript of a tensor argument is an access too, also through a
        # local alias, and one of a name the nest never addresses as
        # ``NAME.iterator`` may reach any tensor.
        "seen = column[col]",
        "alias = column\n        seen = alias[col]",
        "seen = other[col]",
    ],
)
def test_a_first_pass_load_of_what_an_atomic_both_nests_need_updates_declines(
    access: str,
) -> None:
    """Only the final pass keeps the atomic both nests need, and it runs
    after the whole first pass: a load of ``column`` the first pass keeps
    would read the value the nest started from, not the rows added so far."""
    loop = _atomic_program(
        row_total=False,
        before=f"""{_COLUMN_ATOMIC}
        {access}
        (weight.iterator + row * 4 + col).store(seen)""",
    )
    with pytest.raises(exc.BackendUnsupported, match="atomic of the final pass"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_a_subscript_of_thread_local_storage_is_no_access() -> None:
    """A subscript of a tuple the nest builds reads no tensor, so the
    interchange goes ahead, the first pass keeping the store it feeds."""
    loop = _atomic_program(
        row_total=False,
        before=f"""{_COLUMN_ATOMIC}
        scales = (1.0, 2.0, 4.0, 8.0)
        (weight.iterator + row * 4 + col).store(scales[col])""",
    )
    body = interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    assert len(body) == 2


def _atomic_add_then_read_back(
    y: torch.Tensor, acc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adds ``y``'s rows into ``acc`` and reads them back beside the rows
    centered: the atomic reads the reduction's input, so both nests need it."""
    z = torch.empty_like(y)
    w = torch.empty_like(y)
    for tile_i in hl.tile(y.size(0)):
        for tile_k in hl.tile(y.size(1)):
            row = y[tile_i, tile_k, :]
            hl.atomic_add(acc, [tile_i, tile_k, slice(None)], row)
            w[tile_i, tile_k, :] = acc[tile_i, tile_k, :]
            z[tile_i, tile_k, :] = row - row.sum(-1, keepdim=True)
    return z, w


def test_a_kernel_reading_back_an_atomic_both_nests_need_is_declined() -> None:
    """Under the default config the ``tile_k`` loop is interchanged: the
    first nest's load of ``acc`` would read it before the final nest adds
    the rows (``w`` all zeros)."""
    y = torch.randn(64, 32, 128)
    acc = torch.zeros(64, 32, 128)
    with mock_cuda_unavailable():
        bound = cpu_bind(
            helion.kernel(
                _atomic_add_then_read_back, backend="cute", autotune_effort="none"
            ),
            (y, acc),
        )
        with pytest.raises(exc.BackendUnsupported, match="atomic of the final pass"):
            bound.to_code(bound.config_spec.default_config())


# A write the final pass needs that the first pass cannot drop: one in the
# lane loop's prefix, which the final pass would rerun in every serial
# iteration, and one statement holding two stores.
_WRITTEN_TWICE = [
    pytest.param(
        """
for lane in range(4):
    if lane < 4:
        scale = (weight.iterator + lane).load()
        (column.iterator + lane).store(scale)
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        product = value * scale
        reduced = _helion_lane_reduce(product, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(product - reduced)
""",
        id="prefix",
    ),
    pytest.param(
        """
for lane in range(4):
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        if row < rows:
            (output.iterator + index).store(value)
            (column.iterator + lane).store(value)
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (weight.iterator + lane).store(value - reduced)
""",
        id="two_stores",
    ),
    # The final pass's copy of the fragment's lane-invariant allocation runs
    # once ahead of its lane loop, so its lanes would share one fragment,
    # which the lane split cannot lower beside the reduction.
    pytest.param(
        """
for lane in range(4):
    frag = cute.make_rmem_tensor((1,), cutlass.Float32)
    for row in range(rows):
        index = row * 4 + lane
        frag[0] = (source.iterator + index).load()
        reduced = _helion_lane_reduce(frag[0], 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(frag[0] - reduced)
""",
        id="thread_local_storage_the_final_pass_needs",
    ),
]


@pytest.mark.parametrize("source", _WRITTEN_TWICE)
def test_a_write_both_nests_would_run_declines_the_interchange(source: str) -> None:
    loop = _lane_loop(source)
    with pytest.raises(exc.BackendUnsupported, match="twice"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


_CENTERED = """
for lane in range(4):
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
"""
# With ``source`` at 0.5, 1.5, ... and ``output`` and ``column`` at -99.
_REORDERED = [
    # The first pass's copy of the consumer store, with the per-lane partial
    # reduction, stores 0 where its mask holds; the complete row sums (8, 24,
    # ...) fail the mask, so the final pass does not store over it, and the
    # tile program leaves ``output`` at -99.
    pytest.param(
        "        if reduced < 5.0:\n"
        "            (output.iterator + index).store(value - reduced)\n",
        "later first-pass write",
        id="reduction_masked_consumer",
    ),
    # The first pass zeroes ``source``, which the final pass reloads:
    # ``output`` 0 where the tile program centers the loaded rows.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        (source.iterator + index).store(0.0)\n",
        "may read what the first pass stores",
        id="final_load_of_a_first_pass_store",
    ),
    # The first pass reads back its copy's partial value: ``column`` 0 where
    # the tile program reads the last row's centered values.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        seen = (output.iterator + index).load()\n"
        "        (column.iterator + lane).store(seen)\n",
        "may read what a dropped partial store writes",
        id="first_pass_load_of_a_consumer_store",
    ),
    # The same read through a subscript.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        seen = output[index]\n"
        "        (column.iterator + lane).store(seen)\n",
        "cannot drop a partial store a load reads back",
        id="first_pass_subscript_of_a_consumer_store",
    ),
    # ``output[index] = 7.0`` follows the consumer store of the same element
    # and only the first pass needs it: the tile program leaves 7.0, but the
    # final pass runs last and would overwrite it with the centered value.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        (output.iterator + index).store(7.0)\n",
        "later first-pass write",
        id="later_first_pass_store",
    ),
    # A subscript write through a name the nest never addresses as
    # ``NAME.iterator`` may reach ``output`` too.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        other[index] = 7.0\n",
        "later first-pass write",
        id="later_subscript_store_through_an_unknown_name",
    ),
]


@pytest.mark.parametrize(("tail", "message"), _REORDERED)
def test_an_access_out_of_order_across_the_passes_declines_the_interchange(
    tail: str, message: str
) -> None:
    loop = _lane_loop(_CENTERED + tail)
    with pytest.raises(exc.BackendUnsupported, match=message):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


# Accesses the passes cannot resolve to tensors, which may therefore access
# any: with ``source`` at 0.5, 1.5, ... and ``output`` and ``column`` at -99.
_UNRESOLVED = [
    # A store in impure inline assembly after the consumer store: the tile
    # program leaves 7.0 in ``output``, the final pass would overwrite it.
    # The interchange refuses impure assembly before it looks at the stores.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        _cute_inline_asm_elementwise(((output.iterator + index).llvm_ptr,),"
        " asm='st.global.f32 [$1], 7.0;', constraints='=r,l', dtype=cutlass.Int32,"
        " is_pure=False)\n",
        "impure inline assembly cannot be interchanged",
        id="impure_inline_asm",
    ),
    # A load helper the passes do not collect, reading the consumer store's
    # partial value back.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        seen = _cute_load_v4b_if((output.iterator + index).llvm_ptr, True)\n"
        "        (column.iterator + lane).store(seen)\n",
        "cannot drop a partial store a load reads back",
        id="unknown_load_helper",
    ),
    # The same read through a subscript whose base is no name.
    pytest.param(
        "        (output.iterator + index).store(value - reduced)\n"
        "        aliases = (output,)\n"
        "        seen = aliases[0][index]\n"
        "        (column.iterator + lane).store(seen)\n",
        "cannot drop a partial store a load reads back",
        id="subscript_of_a_subscript",
    ),
]


@pytest.mark.parametrize(("tail", "message"), _UNRESOLVED)
def test_an_access_the_passes_cannot_resolve_declines_the_interchange(
    tail: str, message: str
) -> None:
    with pytest.raises(exc.BackendUnsupported, match=message):
        interchange_lane_outside_serial_reductions(
            [_lane_loop(_CENTERED + tail)], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_an_unknown_update_after_the_serial_loop_declines_the_interchange() -> None:
    """Only the first pass keeps a call after the serial loop, which may read
    and update any tensor: the final pass would reload the ``column`` it may
    update, and the call may read the consumer store's partial values.  The
    nest is declined (for the read, the first that the checks see)."""
    loop = _lane_loop(
        """
for lane in range(4):
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load() + (column.iterator + lane).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced)
    _cute_opaque_update((column.iterator + lane).llvm_ptr, 1)
"""
    )
    with pytest.raises(exc.BackendUnsupported, match="partial store a load reads back"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


# Element writes into thread-local storage only the first pass keeps: after
# the serial loop, and beside an atomic both nests need.
_LOCAL_STORAGE = [
    pytest.param(
        """
for lane in range(4):
    scratch = cute.make_rmem_tensor((4,), cutlass.Float32)
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced)
    scratch[lane] = 1.0
""",
        id="after_the_loop",
    ),
    pytest.param(
        """
for lane in range(4):
    frag = cute.make_rmem_tensor((1,), cutlass.Float32)
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced)
        cute.arch.atomic_add((column.iterator + lane).llvm_ptr, val=value, sem='relaxed')
        frag[0] = 1.0
        (weight.iterator + index).store(frag[0])
""",
        id="beside_an_atomic",
    ),
]


@pytest.mark.parametrize("source", _LOCAL_STORAGE)
def test_writes_into_thread_local_storage_keep_the_interchange(source: str) -> None:
    """Such a write reaches no tensor: the nest is interchanged, and the rows
    come out centered."""
    body = interchange_lane_outside_serial_reductions(
        [_lane_loop(source)], proven_disjoint_tensor_pairs=_PAIRS
    )
    assert len(body) > 1
    if "atomic_add" in source:
        return
    rows = 3
    buffers = _execute(body, rows, rows, 4, {"weight": np.zeros(rows * 4)})
    values = buffers["source"].values.reshape(rows, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(rows, 4),
        values - values.sum(-1, keepdims=True),
    )


@pytest.mark.parametrize(
    ("prefix", "update", "carried"),
    [
        ("running = 0.0", "running = running + value", "running"),
        (
            "acc = cute.make_rmem_tensor((1,), cutlass.Float32)\n    acc[0] = 0.0",
            "acc[0] = acc[0] + value",
            "acc[0]",
        ),
    ],
    ids=["name", "thread_local_storage"],
)
def test_a_lane_value_carried_across_serial_iterations_declines_the_interchange(
    prefix: str, update: str, carried: str
) -> None:
    """A lane's running total feeding a store the final pass needs: there
    the serial loop runs outside the lane loop, so the total would run on
    through every lane."""
    loop = _lane_loop(
        f"""
for lane in range(4):
    {prefix}
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        {update}
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced)
        (weight.iterator + index).store({carried})
"""
    )
    with pytest.raises(exc.BackendUnsupported, match="carry a lane's value"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def test_a_lane_invariant_value_carried_across_serial_iterations_is_exact() -> None:
    """A count every lane updates the same way runs once per iteration in the
    final pass, as it does in each lane of the tile program."""
    loop = _lane_loop(
        """
for lane in range(4):
    count = 0.0
    for row in range(rows):
        index = row * 4 + lane
        count = count + 1.0
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced + count)
"""
    )
    body = interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    rows = 3
    buffers = _execute(body, rows, rows, 4)
    values = buffers["source"].values.reshape(rows, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(rows, 4),
        values - values.sum(-1, keepdims=True) + np.arange(1, rows + 1)[:, None],
    )


@pytest.mark.parametrize(
    ("update", "rename_groups"),
    [
        ("scale = scale * 2.0", None),
        # As codegen emits it: the loop's carry renames ``doubled`` to
        # ``scale`` once the passes have run.
        ("doubled = scale * 2.0", {"scale": "scale", "doubled": "scale"}),
    ],
    ids=["name", "renamed"],
)
def test_a_lane_value_updated_after_its_use_declines_the_interchange(
    update: str, rename_groups: dict[str, str] | None
) -> None:
    """``scale`` doubles after the store that reads it: the next serial
    iteration reads the update, so the final pass needs it too, and a lane's
    value carried that way would run on through every lane there."""
    loop = _lane_loop(
        f"""
for lane in range(4):
    scale = (weight.iterator + lane).load()
    for row in range(rows):
        index = row * 4 + lane
        value = (source.iterator + index).load()
        scaled = value * scale
        reduced = _helion_lane_reduce(scaled, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(scaled - reduced)
        {update}
"""
    )
    with pytest.raises(exc.BackendUnsupported, match="carry a lane's value"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS, rename_groups=rename_groups
        )


@pytest.mark.parametrize(
    "lanes", ["range(4)", "range(3, -1, -1)"], ids=["ascending", "descending"]
)
def test_a_lane_invariant_value_updated_after_its_use_is_exact(lanes: str) -> None:
    """``k = k + 1.0`` after the statements that read ``k``: the final pass
    keeps the update, in its serial loop's head after the statement it
    follows there, so each row reads its own ``k``."""
    loop = _lane_loop(
        f"""
for lane in {lanes}:
    k = 0.0
    for row in range(rows):
        shift = k * 10.0
        index = row * 4 + lane
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced + shift)
        k = k + 1.0
"""
    )
    body = interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    rows = 3
    buffers = _execute(body, rows, rows, 4)
    values = buffers["source"].values.reshape(rows, 4)
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(rows, 4),
        values - values.sum(-1, keepdims=True) + 10.0 * np.arange(rows)[:, None],
    )


def test_a_lane_invariant_rebinding_after_a_lane_read_declines_the_interchange() -> (
    None
):
    """``k = 0`` after the load of ``k * 4 + lane`` would run in the final
    pass's serial loop head, ahead of that load: every row would load row
    0."""
    loop = _lane_loop(
        """
for lane in range(4):
    for row in range(rows):
        k = row
        value = (source.iterator + k * 4 + lane).load()
        k = 0
        total = value + k
        reduced = _helion_lane_reduce(total, 'sum', 0.0, threads_in_group=1)
        (output.iterator + row * 4 + lane).store(total - reduced)
"""
    )
    with pytest.raises(exc.BackendUnsupported, match="uses the name it binds"):
        interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )


def _rows_scaled_by_a_doubling_row(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Centers ``x[i, k, :] * scale``, ``scale`` doubling from one ``tile_k``
    block to the next."""
    z = torch.empty_like(x)
    for tile_i in hl.tile(x.size(0)):
        scale = w[tile_i, :]
        for tile_k in hl.tile(x.size(1)):
            row = x[tile_i, tile_k, :] * scale[:, None, :]
            z[tile_i, tile_k, :] = row - row.sum(-1, keepdim=True)
            scale = scale * 2.0
    return z


def test_a_kernel_carrying_a_lane_value_across_the_tile_loop_is_declined() -> None:
    """Under the default config one thread walks a row's 128 features in a
    lane loop around the ``tile_k`` loop, and ``scale`` holds one value per
    feature: the interchange's final nest would carry it from lane to lane,
    so it declines the nest, and the computed fragments the config falls
    back to need more shared memory than the budget.  (The final nest used
    to reload ``w`` in every ``tile_k`` block, dropping the doubling.)"""
    with pytest.raises(exc.InvalidConfig, match="carry a lane's value"):
        _default_code(
            _rows_scaled_by_a_doubling_row,
            (torch.randn(64, 32, 128), torch.randn(64, 128)),
        )


def _column_sums_added(x: torch.Tensor) -> torch.Tensor:
    b, m = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            hl.atomic_add(out, [tile_m], x[tile_b, tile_m].sum(dim=0))
    return out


def test_column_sums_over_a_lane_looped_row_tile_are_added_once() -> None:
    """Eight rows of a tile on one thread walk a lane loop around the column
    loop; the interchange puts the column loop outside and the kernel holds
    one atomic of the full column sum, with no first pass left."""
    with mock_cuda_unavailable():
        bound = cpu_bind(
            helion.kernel(_column_sums_added, backend="cute", autotune_effort="none"),
            (torch.randn(37, 100),),
        )
        config = dict(bound.config_spec.default_config().config)
        config.update(
            block_sizes=[8, 32],
            num_threads=[1, 32],
            cute_vector_widths=[1, 1],
            cute_lane_layouts=["strided", "strided"],
        )
        code = bound.to_code(helion.Config(**config))
    kernel = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert len(_atomics(list(kernel.body))) == 1, code
    assert ast.unparse(kernel).count("for lane_0 in range(8):") == 1, code
    (column_loop,) = [node for node in kernel.body if isinstance(node, ast.For)]
    assert column_loop.target.id == "tile_offset_1"  # pyrefly: ignore [missing-attribute]


def _halve_rows_and_center(
    y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Halves ``y`` in place beside ``z``, ``y``'s rows centered: the store
    into ``y`` reads the reduction's input, not its result, and the final
    nest needs it too."""
    for tile_i in hl.tile(y.size(0)):
        for tile_k in hl.tile(y.size(1)):
            row = y[tile_i, tile_k, :]
            y[tile_i, tile_k, :] = row * 0.5
            z[tile_i, tile_k, :] = row - row.sum(-1, keepdim=True)
    return y, z


def _accumulate_rows(x: torch.Tensor, acc: torch.Tensor, passes: int) -> torch.Tensor:
    """Every pass adds up the rows scaled by ``acc`` and adds the row sums into
    ``acc``: each pass scales by the totals the passes before it added."""
    totals = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        t = hl.zeros([tile_m, x.size(1)], dtype=torch.float32)
        for _ in hl.grid(passes):
            row = x[tile_m, :]
            t = t + row * acc[tile_m][:, None]
            hl.atomic_add(acc, [tile_m], row.sum(-1))
        totals[tile_m, :] = t
    return totals


def test_a_running_total_an_atomic_adds_to_is_refused() -> None:
    """One thread per row walks its 128 features in a lane loop around the
    pass loop, which is interchanged: the first nest's loads of ``acc`` would
    read the starting totals in every pass, with the final nest adding the
    row sums only after the whole first nest."""
    with pytest.raises(exc.BackendUnsupported, match="atomic of the final pass"):
        _default_code(
            _accumulate_rows,
            (torch.randn(64, 128), torch.zeros(64), 3),
            num_threads=[32, 1],
        )


def test_a_store_both_nests_need_runs_once() -> None:
    """Under the default config one thread walks a row's 128 features in a
    lane loop around the ``tile_k`` loop, which is interchanged.  The first
    nest halved ``y``, and the final nest then loaded the halved rows,
    summed them and halved them again: ``y`` came out as ``y / 4`` and ``z``
    from ``y / 2``.  The final nest reads that store back, so the first
    nest's copy goes, and with it the whole first nest."""
    code = _default_code(
        _halve_rows_and_center, (torch.randn(64, 32, 128), torch.empty(64, 32, 128))
    )
    (final,) = _top_loops(code)
    assert _accesses(final, "y") == (2, 1), code
    assert _accesses(final, "z") == (0, 1), code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_a_store_both_nests_need_matches_eager() -> None:
    y = torch.randn(64, 32, 128, device=DEVICE)
    kernel = helion.kernel(
        _halve_rows_and_center, backend="cute", autotune_effort="none"
    )
    code, (out_y, out_z) = code_and_output(kernel, (y.clone(), torch.empty_like(y)))
    assert code.count("for synthetic_lane_2 in range(128):") == 2
    torch.testing.assert_close(out_y, y * 0.5)
    torch.testing.assert_close(out_z, y - y.sum(-1, keepdim=True), rtol=1e-4, atol=1e-4)
