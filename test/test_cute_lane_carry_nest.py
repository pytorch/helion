"""A trace-time vector lane loop that carries values through a serial loop.

A mix-order reduction (a LayerNorm backward) reduces every row of a split
over its columns and accumulates per-column partials over the split's rows.
With one vector per thread the column lanes are a trace-time loop around the
serial row loop, which the lane/serial interchange lowers as two nests when
it can: one for the partials, one for the row reductions, each walking the
rows.  Where it declines (a vector row's packets), the lanes carry their
partials through the serial loop in one nest instead
(``_lane_carry_nest``): each row's packets are loaded once, its partials
are updated beside the loads, its reductions combine across the threads and
its consumers follow.

A tile of the row loop on one thread is a lane loop of rows inside the
serial loop, which holds the vector row's reductions: it runs as a serial
loop of rows (``lane_loops/tile_trips.py``), the lanes carrying their
partials through the trips and the tile's sums through its rows, and
``cute_lane_unroll`` issues the loads of a trip's rows first.

The first half runs hand-built lane programs through the interchange and the
split and executes them on CPU buffers; the second renders mix-order kernels
on CPU; the ``cuda`` tests run them against eager.
"""

from __future__ import annotations

import ast
import functools
from itertools import combinations
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._compiler import tile_strategy
from helion._compiler.cute.autotune import search_surface
from helion._compiler.cute.autotune.heuristics.cute import _row_lanes_hold_tile
from helion._compiler.cute.autotune.heuristics.cute_lane_carry_rows import (
    CuteLaneCarryRowHeuristic,
)
from helion._compiler.cute.interchanged_store_dce import _reordered_instances_never_meet
from helion._compiler.cute.interchanged_store_dce import (
    serial_loop_outside_lanes_keeps_memory,
)
from helion._compiler.cute.lane_loops import split
from helion._compiler.cute.lane_loops import tile_trips
from helion._compiler.cute.lane_loops.markers import HELION_LANE_CARRY_ATTR
from helion._compiler.cute.reduction_lanes import sequence_tile_holds_warp_lanes
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_lane_programs import run_lane_program
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._testing.cute_lane_programs import Buffer
    from helion._testing.cute_lane_programs import Pointer
    from helion.autotuner.block_id_sequence import BlockIdSequence
    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_TENSORS = ("source", "weight", "scales", "output", "column", "dots", "marks")
_PAIRS = {frozenset(pair) for pair in combinations(_TENSORS, 2)}
_ROWS = 3
_VEC = 4
_LOAD = "_helion_persistent_branch_vec_load"
_STORE = "_helion_persistent_branch_vec_store"

# One thread's vector row of ``_VEC`` columns around the serial row loop:
# each row is centered on its sum over the columns and scaled, and the
# columns accumulate the loaded values (``total``, updated under the name
# its pending rename merges into it, as a device loop's carry is) and the
# weighted ones (``dot``).
_PROGRAM = f"""
for lane in cutlass.range_constexpr({_VEC}):
    col = base + lane
    total = 0.0
    dot = 0.0
    for row in range({_ROWS}):
        index = row * {_VEC} + col
        total_copy = total
        value = {_LOAD}(1, {_VEC}, 'cutlass.Float32', '', source.iterator + index, (source.iterator + index).load())
        w = {_LOAD}(1, {_VEC}, 'cutlass.Float32', '', weight.iterator + col, (weight.iterator + col).load())
        product = value * w
        reduced = _helion_lane_reduce(product, 'sum', 0.0, threads_in_group=1)
        scale = (scales.iterator + row).load()
        result = scale * (product - reduced)
        {_STORE}(1, {_VEC}, 'cutlass.Float32', output.iterator + index, result, None)
        v_1 = total_copy + value
        dot = dot + product
    {_STORE}(1, {_VEC}, 'cutlass.Float32', column.iterator + col, total, None)
    {_STORE}(1, {_VEC}, 'cutlass.Float32', dots.iterator + col, dot, None)
"""
_RENAMES = {"v_1": "total"}
_SUFFIX = "its lanes cannot carry their values through the serial loop"


def _lane_loop(source: str) -> ast.For:
    loop = ast.parse(source).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, "lane")
    return loop


def _namer() -> Callable[[str], str]:
    taken: list[str] = []

    def new_var(name: str) -> str:
        taken.append(name)
        return f"{name}_{len(taken)}"

    return new_var


def _interchange(
    source: str = _PROGRAM, pairs: set[frozenset[str]] | None = None
) -> list[ast.AST]:
    return split.interchange_lane_outside_serial_reductions(
        [_lane_loop(source)],
        proven_disjoint_tensor_pairs=_PAIRS if pairs is None else pairs,
        protected_names={"total", "v_1"},
        rename_groups=_RENAMES,
        new_var=_namer(),
    )


def _lane_loops(node: ast.AST) -> list[ast.For]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.For)
        and getattr(child, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, None) == "lane"
    ]


def _serial_loop(body: list[ast.AST]) -> ast.For:
    (serial,) = (
        statement
        for statement in body
        if isinstance(statement, ast.For) and ast.unparse(statement.target) == "row"
    )
    return serial


def _arrays(rows: int = _ROWS) -> dict[str, np.ndarray]:
    # Dyadic values: every sum below is exact in any order.
    return {
        "source": np.arange(rows * _VEC, dtype=np.float64) + 0.5,
        "weight": 2.0 ** np.arange(_VEC),
        "scales": np.arange(rows, dtype=np.float64) + 1.0,
        "output": np.full(rows * _VEC, -99.0),
        "column": np.full(_VEC, -99.0),
        "dots": np.full(_VEC, -99.0),
        "marks": np.full(rows, -99.0),
    }


def _execute(
    body: list[ast.AST],
    renames: dict[str, str] = _RENAMES,
    rows: int = _ROWS,
    **constants: int,
) -> dict[str, Buffer]:
    """Lower ``body``'s lane reductions, apply the pending renames and run
    it as one thread on CPU buffers of ``rows`` rows, ``constants`` bound."""

    def store(
        block: int, width: int, dtype: str, pointer: Pointer, value: float, mask: Any
    ) -> None:
        if mask is None or mask:
            pointer.store(value)

    return run_lane_program(
        body,
        _arrays(rows),
        {
            **constants,
            "base": 0,
            "cutlass": SimpleNamespace(Float32=float, Int32=int, range_constexpr=range),
            _LOAD: lambda block, width, dtype, eviction, pointer, value: value,
            _STORE: store,
        },
        uniform_names={*_TENSORS, "base"},
        proven_disjoint_tensor_pairs=_PAIRS,
        renames=renames,
        filename="<lane carry nest>",
    )


def test_the_lanes_carry_their_partials_through_one_nest() -> None:
    loop = _lane_loop(_PROGRAM)
    statements = [*loop.body[:3], *cast("ast.For", loop.body[3]).body, *loop.body[4:]]
    body = split.interchange_lane_outside_serial_reductions(
        [loop],
        proven_disjoint_tensor_pairs=_PAIRS,
        protected_names={"total", "v_1"},
        rename_groups=_RENAMES,
        new_var=_namer(),
    )
    serial = _serial_loop(body)
    # The partials' initial values, the rows, the partials' stores.
    assert [len(_lane_loops(statement)) for statement in body if _lane_loops(statement)]
    assert len(_lane_loops(ast.Module(body=body, type_ignores=[]))) == 3
    (inner,) = _lane_loops(serial)
    # The serial body's statements are the nest's own, once each: the scale,
    # which no lane has a value of its own of, ahead of the lanes.
    placed = [
        position
        for statement in statements
        for position, holder in enumerate((serial.body, inner.body))
        if any(statement is other for other in holder)
    ]
    assert placed == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], ast.unparse(serial)
    handbacks = [
        ast.unparse(statement)
        for statement in inner.body
        if getattr(statement, HELION_LANE_CARRY_ATTR, False)
    ]
    # ``total`` is handed back under the name its update assigns.
    assert handbacks == [
        "dot_lane_next_11.append(dot)",
        "total_lane_next_12.append(v_1)",
    ]
    assert inner.body[-2:] == [
        statement
        for statement in inner.body
        if getattr(statement, HELION_LANE_CARRY_ATTR, False)
    ]


def test_the_split_updates_the_partials_beside_the_loads() -> None:
    """The hand-backs and the partials' updates run in the accumulate pass,
    ahead of the reduction they do not read; the consume pass keeps the
    store and what it reads."""
    body = split.split_lane_loop_reductions(
        _interchange(),
        uniform_names={*_TENSORS, "base"},
        proven_disjoint_tensor_pairs=_PAIRS,
        rename_groups=_RENAMES,
    )
    accumulate, consume = _lane_loops(_serial_loop(body))
    first, second = ast.unparse(accumulate), ast.unparse(consume)
    assert "dot_lane_next_11.append(dot)" in first, first
    assert "total_lane_next_12.append(v_1)" in first, first
    assert "v_1 = total_copy + value" in first, first
    assert "reduced_lane_acc = reduced_lane_acc + product" in first, first
    assert "append" not in second and "total" not in second, second
    assert f"{_STORE}(1, {_VEC}, 'cutlass.Float32', output.iterator" in second, second


def test_a_producer_both_passes_need_is_copied_for_the_consume_pass() -> None:
    """``scaled`` feeds the partial, so it runs in the accumulate pass, and
    the row's store, so the consume pass runs it as well, but not the
    reduction's input: each pass holds a statement of its own, as it does
    for the producers of the reduction's input (later passes key facts by
    the statement and rewrite it in place)."""
    source = (
        _mutated(
            "        product = value * w\n",
            "        product = value * w\n"
            f"        scaled = {_LOAD}(1, {_VEC}, 'cutlass.Float32', '', "
            "weight.iterator + col, (weight.iterator + col).load())\n",
        )
        .replace("scale * (product - reduced)", "scale * (product - reduced) * scaled")
        .replace("dot = dot + product", "dot = dot + product * scaled")
    )
    body = split.split_lane_loop_reductions(
        _interchange(source),
        uniform_names={*_TENSORS, "base"},
        proven_disjoint_tensor_pairs=_PAIRS,
        rename_groups=_RENAMES,
    )
    accumulate, consume = _lane_loops(_serial_loop(body))
    for loop in (accumulate, consume):
        assert f"scaled = {_LOAD}(" in ast.unparse(loop), ast.unparse(loop)
    shared = [
        ast.unparse(statement)
        for statement in accumulate.body
        if any(statement is other for other in consume.body)
    ]
    assert not shared, shared


def test_a_hand_back_that_reads_the_reduction_is_no_accumulate_statement() -> None:
    """The split checks the hand-backs on its own: one whose value is
    computed from the reduction cannot run in the accumulate pass, ahead of
    the reduction, so every hand-back stays among the consumers, where a
    statement that is no store is refused.  (``_lane_carry_nest`` declines
    such a nest; here its nest is given one after the fact.)"""
    body = _interchange()
    (inner,) = _lane_loops(_serial_loop(body))
    (update,) = (
        statement
        for statement in inner.body
        if ast.unparse(statement) == "dot = dot + product"
    )
    cast("ast.Assign", update).value = ast.parse("dot + reduced", mode="eval").body
    with pytest.raises(exc.BackendUnsupported, match="standalone call has side"):
        split.split_lane_loop_reductions(
            body,
            uniform_names={*_TENSORS, "base"},
            proven_disjoint_tensor_pairs=_PAIRS,
            rename_groups=_RENAMES,
        )


@pytest.mark.parametrize(
    ("anchor", "rebinding"),
    [
        # The reduction's input, doubled for the partial behind the marker.
        ("dot = dot + product", "product = product * 2.0"),
        # The partial itself is what the row reduces, ahead of its update.
        ("dot = dot + product", None),
    ],
    ids=["rebound-input", "reduced-carry"],
)
def test_a_carry_that_rebinds_a_reductions_input_is_no_accumulate_statement(
    anchor: str, rebinding: str | None
) -> None:
    """The accumulate pass adds each lane's input to its reduction behind
    its other statements.  A producer of a carried value that binds the
    input again behind the marker would change what is reduced (the row
    came out centered on the doubled products' sum), so the hand-backs stay
    among the consumers, where a statement that is no store is refused.
    (``_lane_carry_nest`` declines such a nest; here its nest is given one
    after the fact.)"""
    body = _interchange()
    (inner,) = _lane_loops(_serial_loop(body))
    if rebinding is None:
        (marker,) = (
            statement
            for statement in inner.body
            if ast.unparse(statement).startswith("reduced = _helion_lane_reduce(")
        )
        call = cast("ast.Call", cast("ast.Assign", marker).value)
        call.args[0] = ast.Name(id="dot", ctx=ast.Load())
    else:
        (position,) = (
            index
            for index, statement in enumerate(inner.body)
            if ast.unparse(statement) == anchor
        )
        inner.body.insert(position, ast.parse(rebinding).body[0])
    with pytest.raises(exc.BackendUnsupported, match="standalone call has side"):
        split.split_lane_loop_reductions(
            body,
            uniform_names={*_TENSORS, "base"},
            proven_disjoint_tensor_pairs=_PAIRS,
            rename_groups=_RENAMES,
        )


def test_a_store_in_the_lane_prefix_fails_the_memory_proof() -> None:
    """The proof of the new order checks the prefix by itself (the nest
    declines a prefix it cannot rerun before asking): a lane's prefix store
    would run once per row there."""
    loop = _lane_loop(
        _mutated(
            "    dot = 0.0\n", "    dot = 0.0\n    (marks.iterator + col).store(0.0)\n"
        )
    )
    (serial,) = (statement for statement in loop.body if isinstance(statement, ast.For))
    assert (
        serial_loop_outside_lanes_keeps_memory(loop, serial, _PAIRS, set(), {})
        == "the lane loop's prefix writes memory"
    )
    plain = _lane_loop(_PROGRAM)
    (serial,) = (
        statement for statement in plain.body if isinstance(statement, ast.For)
    )
    assert (
        serial_loop_outside_lanes_keeps_memory(plain, serial, _PAIRS, set(), {}) is None
    )


def test_a_store_of_a_span_fails_the_memory_proof() -> None:
    """The proof compares the stores' start addresses, so it takes stores of
    one element only: a scalar store or the vectorizer's store marker.  A
    packet of ``_VEC`` elements at ``row * _VEC + lane`` starts at a
    different element in every instance, yet lane 1 of one row and lane 0 of
    the next both write element ``_VEC``, in the order the interchange
    swaps."""
    marker = (
        f"{_STORE}(1, {_VEC}, 'cutlass.Float32', output.iterator + index, result, None)"
    )
    for store, reason in (
        (marker, None),
        ("(output.iterator + index).store(result)", None),
        (
            "_cute_store_u32_vec(output.iterator + index, result)",
            "a store of the serial loop may write more than one element",
        ),
        (
            "cute.arch.store(output.iterator + index, result)",
            "a store of the serial loop may write more than one element",
        ),
    ):
        loop = _lane_loop(_mutated(marker, store))
        (serial,) = (
            statement for statement in loop.body if isinstance(statement, ast.For)
        )
        assert (
            serial_loop_outside_lanes_keeps_memory(loop, serial, _PAIRS, set(), {})
            == reason
        ), store


def test_one_nest_computes_what_the_rows_and_columns_say() -> None:
    buffers = _execute(_interchange())
    arrays = _arrays()
    source = arrays["source"].reshape(_ROWS, _VEC)
    product = source * arrays["weight"]
    expected = arrays["scales"][:, None] * (product - product.sum(-1, keepdims=True))
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(_ROWS, _VEC), expected
    )
    np.testing.assert_array_equal(buffers["column"].values, source.sum(0))
    np.testing.assert_array_equal(buffers["dots"].values, product.sum(0))
    # Each element is stored once; the accumulate pass and the consume pass
    # each load a row's elements (one packet for both, once vectorized).
    assert buffers["output"].stores == _ROWS * _VEC
    assert buffers["source"].loads == 2 * _ROWS * _VEC
    assert buffers["scales"].loads == _ROWS
    assert buffers["column"].stores == buffers["dots"].stores == _VEC


def test_masked_accesses_keep_their_masks() -> None:
    """A masked tail rides along: the load reads zero and the store is
    skipped past the valid rows, in every lane and in both lane passes."""
    source = _mutated(
        "(source.iterator + index).load())",
        f"(source.iterator + index).load() if row < {_ROWS - 1} else 0.0)",
    ).replace("index, result, None)", f"index, result, row < {_ROWS - 1})")
    buffers = _execute(_interchange(source))
    arrays = _arrays()
    values = arrays["source"].reshape(_ROWS, _VEC).copy()
    values[-1] = 0.0
    product = values * arrays["weight"]
    expected = arrays["scales"][:, None] * (product - product.sum(-1, keepdims=True))
    expected[-1] = -99.0
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(_ROWS, _VEC), expected
    )
    np.testing.assert_array_equal(buffers["column"].values, values.sum(0))
    assert buffers["output"].stores == (_ROWS - 1) * _VEC
    assert buffers["source"].loads == 2 * (_ROWS - 1) * _VEC


def test_a_store_no_lane_reads_stays_among_the_lanes() -> None:
    """Only relocatable assignments move ahead of the lanes: a store that
    reads no lane's value keeps its place in the serial body."""
    source = _mutated(
        "dot = dot + product",
        "dot = dot + product\n        (marks.iterator + row).store(scale)",
    )
    body = _interchange(source)
    serial = _serial_loop(body)
    (inner,) = _lane_loops(serial)
    assert "marks.iterator" in ast.unparse(inner), ast.unparse(serial)
    assert "marks.iterator" not in ast.unparse(
        ast.Module(body=serial.body[: serial.body.index(inner)], type_ignores=[])
    )
    buffers = _execute(body)
    np.testing.assert_array_equal(buffers["marks"].values, _arrays()["scales"])
    np.testing.assert_array_equal(
        buffers["column"].values, _arrays()["source"].reshape(_ROWS, _VEC).sum(0)
    )


def test_a_nest_the_two_nests_lower_keeps_them() -> None:
    """The single nest takes only what the two nests decline: scalar accesses
    they resolve keep their two nests."""
    source = f"""
for lane in cutlass.range_constexpr({_VEC}):
    col = base + lane
    total = 0.0
    for row in range({_ROWS}):
        index = row * {_VEC} + col
        value = (source.iterator + index).load()
        reduced = _helion_lane_reduce(value, 'sum', 0.0, threads_in_group=1)
        (output.iterator + index).store(value - reduced)
        total = total + value
    (column.iterator + col).store(total)
"""
    with patch.object(split, "_lane_carry_nest", side_effect=AssertionError):
        first, *final = split.interchange_lane_outside_serial_reductions(
            [_lane_loop(source)], proven_disjoint_tensor_pairs=_PAIRS, new_var=_namer()
        )
    assert isinstance(first, ast.For) and ast.unparse(first.target) == "lane"
    assert ast.unparse(_serial_loop(final).target) == "row"


def _mutated(old: str, new: str) -> str:
    assert old in _PROGRAM
    return _PROGRAM.replace(old, new)


# One row per condition of ``_lane_carry_nest``: each mutation of the program
# breaks that condition only.
@pytest.mark.parametrize(
    ("source", "reason"),
    [
        pytest.param(
            _mutated("dot = dot + product", "dot = dot + reduced"),
            "a carried value reads a lane reduction",
            id="reduction",
        ),
        pytest.param(
            _mutated(
                "dot = dot + product",
                "product = product * 2.0\n        dot = dot + product",
            ),
            "a reduction's input is bound again behind the reduction",
            id="rebound-input",
        ),
        pytest.param(
            _mutated("_helion_lane_reduce(product,", "_helion_lane_reduce(dot,"),
            "a reduction's input is bound again behind the reduction",
            id="reduced-carry",
        ),
        pytest.param(
            _mutated(
                "dot = dot + product",
                "if value > 0.0:\n            dot = dot + product",
            ),
            "a carried value is not updated by plain assignments",
            id="branch",
        ),
        pytest.param(
            _mutated("    dot = 0.0\n", "    dot = 0.0\n    dot = dot + 1.0\n"),
            "dot has no single initializer in the lane loop's prefix",
            id="two-initializers",
        ),
        pytest.param(
            _mutated("    dot = 0.0\n", "    dot = 0.0\n    seed = dot + 1.0\n"),
            "the lane loop's prefix reads a carried value",
            id="prefix-read",
        ),
        pytest.param(
            _mutated("    dot = 0.0\n", "    dot = previous\n")
            + "    previous = dot\n",
            "a value is carried from one lane to the next",
            id="lane-carry",
        ),
        # ``lane_scale`` is rerun among the lanes of every row, behind the
        # rows' ``factor = 3.0``: it would scale by 3 where the prefix
        # scaled by 2.
        pytest.param(
            _mutated(
                "    dot = 0.0\n",
                "    dot = 0.0\n"
                "    factor = 2.0\n"
                "    lane_scale = factor * (lane + 1)\n",
            ).replace(
                "dot = dot + product",
                "factor = 3.0\n        dot = dot + product * lane_scale",
            ),
            "the lane loop's prefix reads a name the serial loop binds",
            id="prefix-rebound",
        ),
        # ``s`` is rerun among the lanes behind the whole of the prefix that
        # holds no lane value, which runs once ahead of the rows: it would
        # scale by the 3 where the prefix scaled by the 2.
        pytest.param(
            _mutated(
                "    col = base + lane\n",
                "    f = 2.0\n    col = base + lane\n    s = f * col\n    f = 3.0\n",
            ).replace("(product - reduced)", "(product - reduced) + s"),
            "the lane loop's prefix binds a name behind a per-lane statement "
            "that reads it",
            id="prefix-rebound-behind",
        ),
        # No lane runs ``f = 2.0`` again: the suffix of every lane after the
        # first would read the 3 that the lane before it bound.
        pytest.param(
            _mutated(
                "    col = base + lane\n", "    f = 2.0\n    col = base + lane\n"
            ).replace("col, total, None)", "col, total * f, None)")
            + "    f = 3.0\n",
            "the suffix binds a name of the lane loop's prefix that it reads first",
            id="suffix-rebound",
        ),
        pytest.param(
            _mutated(
                "    dot = 0.0\n",
                "    dot = 0.0\n    (dots.iterator + col).store(0.0)\n",
            ),
            "the lane loop's prefix cannot be rerun",
            id="prefix-store",
        ),
        pytest.param(
            _mutated(f"range({_ROWS})", "range(col)"),
            "the serial loop's range varies with the lane or the loop",
            id="lane-range",
        ),
        pytest.param(
            _mutated(
                "dot = dot + product",
                "dot = dot + product\n        scratch[lane] = product",
            ),
            "it writes thread-local storage",
            id="local-storage",
        ),
        pytest.param(
            _mutated(
                "dot = dot + product",
                "dot = dot + product\n"
                "        cute.arch.atomic_add(dots.iterator + col, product)",
            ),
            "it holds an atomic, a collective or impure inline assembly",
            id="atomic",
        ),
        pytest.param(
            _mutated(
                "dot = dot + product",
                "dot = dot + product\n        while dot > 9.0:\n            pass",
            ),
            "its lanes are not straight-line code",
            id="while",
        ),
        pytest.param(
            _mutated(
                "output.iterator + index, result", "source.iterator + index, result"
            ),
            "a load may read a tensor the nest writes",
            id="in-place",
        ),
        pytest.param(
            _mutated(
                "output.iterator + index, result", "dots.iterator + index, result"
            ),
            "two stores may reach one tensor in another order",
            id="body-and-suffix-store",
        ),
        pytest.param(
            _mutated(
                "output.iterator + index, result",
                "output.iterator + row + col, result",
            ),
            "a store of the serial loop may write one element twice",
            id="colliding-store",
        ),
        pytest.param(
            _mutated(
                "output.iterator + index, result",
                "output.iterator + int(scale) + col, result",
            ),
            "a store of the serial loop has no known address",
            id="loaded-address",
        ),
        # A packet flush writes a span from its address: the proof compares
        # start addresses, which says nothing about the spans.
        pytest.param(
            _mutated(
                f"{_STORE}(1, {_VEC}, 'cutlass.Float32', output.iterator + index, "
                "result, None)",
                "_cute_store_u32_vec(output.iterator + index, result)",
            ),
            "a store of the serial loop may write more than one element",
            id="packet-store",
        ),
        pytest.param(
            _mutated(f"range({_ROWS})", "range(rows)"),
            "the serial loop's stores have no known addresses",
            id="dynamic-rows",
        ),
    ],
)
def test_a_nest_one_condition_short_is_declined(source: str, reason: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match=_SUFFIX) as declined:
        _interchange(source)
    assert str(declined.value).endswith(f"in one nest either: {reason}")
    # The two nests' refusal leads the message.
    assert "interchanged lane reduction" in str(declined.value)


def test_a_prefix_name_bound_ahead_of_its_readers_keeps_the_nest() -> None:
    """The declines for a prefix name that is bound again go by the order of
    the statements: ``f``, bound once ahead of the per-lane statement that
    reads it, runs once ahead of the rows, and ``g``, which the suffix binds
    before it reads it, starts over in every lane."""
    column = f"    {_STORE}(1, {_VEC}, 'cutlass.Float32', column.iterator + col, total"
    source = (
        _mutated(
            "    col = base + lane\n",
            "    f = 2.0\n    g = 5.0\n    col = base + lane\n    s = f * col\n",
        )
        .replace("(product - reduced)", "(product - reduced) + s + g")
        .replace(f"{column}, None)", f"    g = 3.0\n{column} * g, None)")
    )
    body = _interchange(source)
    ahead = [ast.unparse(statement) for statement in body]
    assert ahead.count("f = 2.0") == ahead.count("g = 5.0") == 1, ahead
    buffers = _execute(body)
    arrays = _arrays()
    values = arrays["source"].reshape(_ROWS, _VEC)
    product = values * arrays["weight"]
    centered = arrays["scales"][:, None] * (product - product.sum(-1, keepdims=True))
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(_ROWS, _VEC),
        centered + 2.0 * np.arange(_VEC) + 5.0,
    )
    np.testing.assert_array_equal(buffers["column"].values, 3.0 * values.sum(0))


def test_tensors_without_an_alias_proof_decline_the_nest() -> None:
    with pytest.raises(
        exc.BackendUnsupported, match="a load may read a tensor the nest writes"
    ):
        _interchange(pairs=set())


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            _mutated(f"cutlass.range_constexpr({_VEC})", f"range({_VEC})"),
            id="rolled-lanes",
        ),
        pytest.param(
            _PROGRAM.replace("v_1 = total_copy + value", "pass")
            .replace("dot = dot + product", "pass")
            .replace("total, None", "0.0, None")
            .replace("dot, None", "0.0, None"),
            id="nothing-carried",
        ),
    ],
)
def test_any_other_shape_keeps_the_two_nests_refusal(source: str) -> None:
    with pytest.raises(exc.BackendUnsupported) as declined:
        _interchange(source)
    assert _SUFFIX not in str(declined.value)


@pytest.mark.parametrize(
    ("serial", "lane", "kept"),
    [
        # A row-major layout: a row's lanes stay below the next row.
        pytest.param((768, 7), (1, 7), True, id="row-major"),
        # A transposed layout: the lane stride outweighs every row.
        pytest.param((1, 7), (8, 7), True, id="transposed"),
        # (row + 1, lane) and (row, lane + 1) share an element.
        pytest.param((1, 7), (1, 7), False, id="diagonal"),
        # A lane-uniform store (a row's sum): two instances on one element
        # are two lanes of one row, which keep their order.
        pytest.param((1, 7), (0, 7), True, id="lane-uniform"),
        pytest.param((0, 7), (4, 7), True, id="row-uniform"),
        # Every instance writes one element.
        pytest.param((0, 7), (0, 7), False, id="one-element"),
        pytest.param((0, 0), (0, 7), True, id="one-row"),
        # Offsets that meet modulo 2**32.
        pytest.param((2**31, 2), (1, 7), False, id="wrapped"),
        # A row step of nothing modulo 2**32: as row-uniform.
        pytest.param((2**32, 7), (1, 7), True, id="row-step-wraps-to-nothing"),
        pytest.param((2**32, 7), (2**32, 7), False, id="both-steps-wrap"),
    ],
)
def test_reordered_store_instances_meet_only_as_the_offsets_say(
    serial: tuple[int, int], lane: tuple[int, int], kept: bool
) -> None:
    assert _reordered_instances_never_meet(*serial, *lane) is kept


_TRIP = 2
_TRIP_ROWS = 4
_FOLD = "threads_in_group=1, owner_lane='row_lane', strided_restore=True"
# The program's rows two to a tile on one thread: the serial loop walks the
# tiles and a lane loop the rows of each.  The row reduction is the vector
# row's (owned by ``lane``); the columns add each tile's sums, one-thread
# folds of the tile, to the partials they carry.
_TRIP_PROGRAM = f"""
for lane in cutlass.range_constexpr({_VEC}):
    col = base + lane
    total = 0.0
    dot = 0.0
    for tile in range(cutlass.Int32(0), cutlass.Int32({_TRIP_ROWS}), cutlass.Int32(_BLOCK_SIZE_1)):
        for row_lane in range({_TRIP}):
            row = tile + cutlass.Int32(row_lane)
            index = row * {_VEC} + col
            total_copy = total
            dot_copy = dot
            value = {_LOAD}(1, {_VEC}, 'cutlass.Float32', '', source.iterator + index, (source.iterator + index).load())
            w = {_LOAD}(1, {_VEC}, 'cutlass.Float32', '', weight.iterator + col, (weight.iterator + col).load())
            product = value * w
            reduced = _helion_lane_reduce(product, 'sum', 0.0, threads_in_group=1, owner_lane='lane')
            scale = (scales.iterator + row).load()
            result = scale * (product - reduced)
            {_STORE}(1, {_VEC}, 'cutlass.Float32', output.iterator + index, result, None)
            fold = _helion_lane_reduce(value, 'sum', 0.0, {_FOLD})
            v_1 = total_copy + fold
            fold_1 = _helion_lane_reduce(product, 'sum', 0.0, {_FOLD})
            v_2 = dot_copy + fold_1
    {_STORE}(1, {_VEC}, 'cutlass.Float32', column.iterator + col, total, None)
    {_STORE}(1, {_VEC}, 'cutlass.Float32', dots.iterator + col, dot, None)
"""
_TRIP_RENAMES = {"v_1": "total", "v_2": "dot"}
_TRIP_REFUSAL = "reduction marker is nested in a different lane owner; "


def _trip_loop(source: str = _TRIP_PROGRAM) -> ast.For:
    """The program with its tile's rows a lane loop, as the tile strategy
    of one thread leaves them."""
    loop = _lane_loop(source)
    (serial,) = (stmt for stmt in loop.body if isinstance(stmt, ast.For))
    setattr(serial.body[0], tile_strategy.HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    return loop


def _serialize(loop: ast.For) -> None:
    tile_trips.serialize_tile_lane_trips(
        [loop],
        constexpr_value={"_BLOCK_SIZE_1": _TRIP}.get,
        rename_groups=_TRIP_RENAMES,
        new_var=_namer(),
    )


def _trips(
    source: str = _TRIP_PROGRAM, pairs: set[frozenset[str]] | None = None
) -> list[ast.AST]:
    loop = _trip_loop(source)
    _serialize(loop)
    return split.interchange_lane_outside_serial_reductions(
        [loop],
        proven_disjoint_tensor_pairs=_PAIRS if pairs is None else pairs,
        protected_names={"total", "v_1", "dot", "v_2"},
        constexpr_value={"_BLOCK_SIZE_1": _TRIP}.get,
        rename_groups=_TRIP_RENAMES,
        new_var=_namer(),
    )


def _named_loop(body: list[ast.stmt] | list[ast.AST], name: str) -> ast.For:
    (loop,) = (
        stmt
        for stmt in body
        if isinstance(stmt, ast.For) and ast.unparse(stmt.target) == name
    )
    return loop


def test_a_tile_of_rows_becomes_a_serial_loop() -> None:
    loop = _trip_loop()
    _serialize(loop)
    serial = _named_loop(loop.body, "tile")
    rows = _named_loop(serial.body, "row_lane")
    assert getattr(rows, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, None) is None
    assert getattr(rows, tile_trips.HELION_TILE_TRIP_ATTR) == "row_lane"
    # What the consumers read moves ahead of the rows, each fold starts at
    # its identity, and the consumers follow the last row, in their order.
    assert [ast.unparse(stmt) for stmt in serial.body if stmt is not rows] == [
        "total_copy = total",
        "dot_copy = dot",
        "fold_tile_acc_1 = 0.0",
        "fold_1_tile_acc_2 = 0.0",
        "fold = fold_tile_acc_1",
        "v_1 = total_copy + fold",
        "fold_1 = fold_1_tile_acc_2",
        "v_2 = dot_copy + fold_1",
    ]
    body = [ast.unparse(stmt) for stmt in rows.body]
    assert "fold_tile_acc_1 = fold_tile_acc_1 + value" in body
    assert "fold_1_tile_acc_2 = fold_1_tile_acc_2 + product" in body
    # The vector row's reduction stays among the rows; no fold's marker does.
    assert sum("_helion_lane_reduce" in stmt for stmt in body) == 1
    assert "owner_lane='row_lane'" not in ast.unparse(rows)
    # The flat nest walks the same rows one per iteration, each fold its
    # row's value.
    flat = getattr(loop, tile_trips.HELION_TILE_TRIP_NEST_ATTR).flat
    text = ast.unparse(_named_loop(flat.body, "tile"))
    assert text.startswith(
        f"for tile in range(cutlass.Int32(0), cutlass.Int32({_TRIP_ROWS})):"
    ), text
    assert "row = tile + cutlass.Int32(0)" in text, text
    assert "fold = value" in text and "v_1 = total_copy + fold" in text, text
    assert "row_lane" not in text, text


def test_the_lanes_carry_through_the_trips_and_the_rows() -> None:
    body = _trips()
    serial = _named_loop(body, "tile")
    rows = _named_loop(serial.body, "row_lane")
    # The partials' initial values, per trip the folds' and the consumers',
    # per row the row's own lanes, and the partials' stores.
    assert len(_lane_loops(ast.Module(body=body, type_ignores=[]))) == 5
    assert [len(_lane_loops(stmt)) for stmt in serial.body if _lane_loops(stmt)] == [
        1,
        1,
        1,
    ]
    (inner,) = _lane_loops(rows)
    head = ast.unparse(
        ast.Module(body=rows.body[: rows.body.index(inner)], type_ignores=[])
    )
    # No lane has a value of its own of the row index or the row's scale.
    assert "row = tile + cutlass.Int32(row_lane)" in head, head
    assert "scale = (scales.iterator + row).load()" in head, head
    # The lanes hand the tile's folds from row to row ...
    assert [
        ast.unparse(stmt)
        for stmt in inner.body
        if getattr(stmt, HELION_LANE_CARRY_ATTR, False)
    ] == [
        "fold_1_tile_acc_2_lane_next_23.append(fold_1_tile_acc_2)",
        "fold_tile_acc_1_lane_next_24.append(fold_tile_acc_1)",
    ]
    # ... and the partials from trip to trip, updated once after its rows.
    consumers = ast.unparse(serial.body[serial.body.index(rows) + 1])
    assert "v_1 = total_copy + fold" in consumers, consumers
    assert "total_lane_next_12.append(v_1)" in consumers, consumers
    assert "v_1" not in ast.unparse(rows), ast.unparse(rows)


def test_rows_per_trip_compute_what_the_rows_and_columns_say() -> None:
    buffers = _execute(_trips(), _TRIP_RENAMES, _TRIP_ROWS, _BLOCK_SIZE_1=_TRIP)
    arrays = _arrays(_TRIP_ROWS)
    source = arrays["source"].reshape(_TRIP_ROWS, _VEC)
    product = source * arrays["weight"]
    expected = arrays["scales"][:, None] * (product - product.sum(-1, keepdims=True))
    np.testing.assert_array_equal(
        buffers["output"].values.reshape(_TRIP_ROWS, _VEC), expected
    )
    np.testing.assert_array_equal(buffers["column"].values, source.sum(0))
    np.testing.assert_array_equal(buffers["dots"].values, product.sum(0))
    # As at one row per trip: each element stored once, each row's elements
    # loaded by its two lane passes, its scale once.
    assert buffers["output"].stores == _TRIP_ROWS * _VEC
    assert buffers["source"].loads == 2 * _TRIP_ROWS * _VEC
    assert buffers["scales"].loads == _TRIP_ROWS
    assert buffers["column"].stores == buffers["dots"].stores == _VEC


def _trip_mutated(old: str, new: str) -> str:
    assert old in _TRIP_PROGRAM
    return _TRIP_PROGRAM.replace(old, new, 1)


# One row per condition of ``serialize_tile_lane_trips``: each mutation of
# the program breaks that condition only.
@pytest.mark.parametrize(
    ("source", "reason"),
    [
        pytest.param(
            _trip_mutated(f"cutlass.Int32({_TRIP_ROWS})", "cutlass.Int32(5)"),
            "its serial loop does not walk whole tiles of static bounds",
            id="partial-tile",
        ),
        pytest.param(
            _trip_mutated(f"cutlass.Int32({_TRIP_ROWS})", "cutlass.Int32(rows)"),
            "its serial loop does not walk whole tiles of static bounds",
            id="dynamic-rows",
        ),
        pytest.param(
            _trip_mutated(f"range({_TRIP})", "range(count)"),
            "the tile's lanes are no loop of a literal count",
            id="dynamic-lanes",
        ),
        pytest.param(
            _trip_mutated(
                "            dot_copy = dot\n",
                "            if tile < 2:\n                dot_copy = dot\n",
            ),
            "the tile's lanes are not straight-line statements",
            id="branch",
        ),
        pytest.param(
            _trip_mutated("total_copy = total", "total_copy = total + row_lane"),
            "the tile's lanes read their offsets beside the row index",
            id="offset-read",
        ),
        pytest.param(
            _trip_mutated(
                f"value, 'sum', 0.0, {_FOLD}",
                "value, 'sum', 0.0, threads_in_group=2, owner_lane='row_lane'",
            ),
            "a reduction of the tile combines threads or another lane",
            id="threaded-fold",
        ),
        pytest.param(
            _trip_mutated(
                "            v_1 = total_copy + fold\n",
                "            v_1 = total_copy + fold\n"
                "            (marks.iterator + 0).store(_helion_lane_reduce("
                f"value, 'sum', 0.0, {_FOLD}))\n",
            ),
            "a reduction of the tile is no plain assignment",
            id="stored-fold",
        ),
        pytest.param(
            _trip_mutated(
                "            v_1 = total_copy + fold\n",
                "            v_1 = total_copy + fold\n"
                "            (marks.iterator + 0).store(fold)\n",
            ),
            "a consumer of the tile's reduction is no plain assignment",
            id="consumer-store",
        ),
        pytest.param(
            _trip_mutated("v_1 = total_copy + fold", "v_1 = total_copy + fold * scale"),
            "a consumer of the tile's reduction reads a value of one row",
            id="row-value",
        ),
        pytest.param(
            _trip_mutated(
                "            dot_copy = dot\n",
                "            dot_copy = dot\n            dot_copy = dot_copy + 0.0\n",
            ),
            "a consumer of the tile's reduction reads a value bound twice",
            id="bound-twice",
        ),
        # ``scaled`` is zero when the partial reads it; run behind the rows,
        # the fold's consumer would bind it last.
        pytest.param(
            _trip_mutated(
                "            v_1 = total_copy + fold\n",
                "            scaled = fold\n"
                "            scaled = 0.0\n"
                "            v_1 = total_copy + scaled\n",
            ),
            "a statement of the rows binds a name the tile's reduction binds",
            id="rebound-consumer",
        ),
    ],
)
def test_a_tile_one_condition_short_keeps_its_lane_loop(
    source: str, reason: str
) -> None:
    loop = _trip_loop(source)
    with pytest.raises(exc.BackendUnsupported) as declined:
        _serialize(loop)
    message = str(declined.value)
    assert _TRIP_REFUSAL in message, message
    assert message.endswith(f"as a serial loop of the lanes either: {reason}"), message
    # Nothing was rewritten.
    assert ast.unparse(loop) == ast.unparse(_trip_loop(source))


# The single nest's own conditions hold for the rows of a tile as for the
# rows of a serial loop: the memory proof reads the flat nest.
@pytest.mark.parametrize(
    ("source", "reason"),
    [
        pytest.param(
            _trip_mutated(
                "source.iterator + index, (source.iterator + index).load())",
                "output.iterator + index, (output.iterator + index).load())",
            ),
            "a load may read a tensor the nest writes",
            id="in-place",
        ),
        pytest.param(
            _trip_mutated(
                f"product, 'sum', 0.0, {_FOLD}", f"result, 'sum', 0.0, {_FOLD}"
            ),
            "a carried value reads a lane reduction",
            id="reduction",
        ),
        pytest.param(
            _trip_mutated(
                "    col = base + lane\n",
                "    f = 2.0\n    col = base + lane\n    s = f * col\n    f = 3.0\n",
            ).replace("(product - reduced)", "(product - reduced) + s"),
            "the lane loop's prefix binds a name behind a per-lane statement "
            "that reads it",
            id="prefix-rebound-behind",
        ),
    ],
)
def test_rows_per_trip_keep_the_single_nests_conditions(
    source: str, reason: str
) -> None:
    with pytest.raises(exc.BackendUnsupported) as declined:
        _trips(source)
    message = str(declined.value)
    assert _TRIP_REFUSAL in message, message
    assert message.endswith(f"of the tile in one nest either: {reason}"), message


def test_a_tiles_candidate_index_minimum_folds_in_the_plain_form() -> None:
    """The minimum over an argmin's candidate indices (``index_candidates``
    on its marker) is a one-thread fold of the tile like any other, and its
    accumulate keeps the plain minimum: the unsigned keys of an Int32 chain
    are for a kernel's values, which a candidate never is."""
    candidates = (
        "fold = _helion_lane_reduce(index, 'min', "
        f"cutlass.Int32(2147483647), {_FOLD}, index_candidates=True)"
    )
    source = _TRIP_PROGRAM.replace(
        f"fold = _helion_lane_reduce(value, 'sum', 0.0, {_FOLD})", candidates
    )
    assert candidates in source
    loop = _trip_loop(source)
    _serialize(loop)
    rows = _named_loop(_named_loop(loop.body, "tile").body, "row_lane")
    assert getattr(rows, tile_trips.HELION_TILE_TRIP_ATTR) == "row_lane"
    body = [ast.unparse(stmt) for stmt in rows.body]
    assert (
        "fold_tile_acc_1 = cute.math.min(fold_tile_acc_1, cutlass.Int32(index), "
        "propagate_nan=True)"
    ) in body, body
    # The same fold of a kernel's Int32 values compares unsigned keys.
    values = _trip_loop(source.replace(", index_candidates=True", ""))
    _serialize(values)
    rows = _named_loop(_named_loop(values.body, "tile").body, "row_lane")
    assert any(
        stmt.startswith("fold_tile_acc_1 = cutlass.Int32(cute.math.min(cutlass.Uint32(")
        for stmt in (ast.unparse(stmt) for stmt in rows.body)
    )


def test_a_tile_on_several_threads_keeps_the_refusal() -> None:
    """Only a tile's lane loop under a vector row is rewritten: without the
    vector row's reduction among its rows there is nothing to serialize."""
    source = _TRIP_PROGRAM.replace("owner_lane='lane'", "owner_lane='row_lane'")
    loop = _trip_loop(source)
    _serialize(loop)
    assert ast.unparse(loop) == ast.unparse(_trip_loop(source))
    assert getattr(loop, tile_trips.HELION_TILE_TRIP_NEST_ATTR, None) is None


def _mix_order(
    grad: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    splits, rows, hidden = grad.shape
    out = torch.empty([splits, rows, hidden], dtype=torch.float32, device=grad.device)
    out_lowp = torch.empty([splits, rows, hidden], dtype=grad.dtype, device=grad.device)
    partials = torch.empty([2, splits, hidden], dtype=torch.float32, device=grad.device)
    for tile_s in hl.tile(splits):
        dot = hl.zeros([tile_s, hidden], dtype=torch.float32)
        total = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            g = grad[tile_s, tile_r, :].to(torch.float32)
            gw = g * weight[None, None, :]
            xh = x_hat[tile_s, tile_r, :]
            row_sum = torch.sum(gw, dim=-1, keepdim=True)
            row_dot = torch.sum(gw * xh, dim=-1, keepdim=True)
            y = residual[tile_s, tile_r, :] + scale[tile_s, tile_r][:, :, None] * (
                (gw - row_sum) - xh * row_dot
            )
            out[tile_s, tile_r, :] = y
            out_lowp[tile_s, tile_r, :] = y.to(out_lowp.dtype)
            dot = dot + torch.sum(g * xh, dim=1)
            total = total + torch.sum(g, dim=1)
        partials[0, tile_s, :] = dot
        partials[1, tile_s, :] = total
    return out, out_lowp, partials


def _row_sums(
    grad_weighted: torch.Tensor, x_hat: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The row reductions are only stored; the partials read another tensor."""
    splits, rows, hidden = grad.shape
    row_sum = torch.empty([splits, rows], dtype=torch.float32, device=grad.device)
    row_dot = torch.empty([splits, rows], dtype=torch.float32, device=grad.device)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=grad.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            gw = grad_weighted[tile_s, tile_r, :]
            xh = x_hat[tile_s, tile_r, :]
            row_sum[tile_s, tile_r] = torch.sum(gw, dim=-1)
            row_dot[tile_s, tile_r] = torch.sum(gw * xh, dim=-1)
            g = grad[tile_s, tile_r, :].to(torch.float32)
            acc = acc + torch.sum(g * xh, dim=1)
        partial[tile_s, :] = acc
    return row_sum, row_dot, partial


def _in_place(x: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
    """The rows are centered where they are loaded from."""
    splits, rows, hidden = x.shape
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            x[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            acc = acc + torch.sum(row, dim=1)
        partial[tile_s, :] = acc
    return partial


def _reduction_fed_partials(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The column partials accumulate the centered rows."""
    splits, rows, hidden = x.shape
    out = torch.empty_like(x)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            centered = row - torch.sum(row, dim=-1, keepdim=True)
            out[tile_s, tile_r, :] = centered
            acc = acc + torch.sum(centered, dim=1)
        partial[tile_s, :] = acc
    return out, partial


def _dropout_rows(
    grad: torch.Tensor, x_hat: torch.Tensor, scale: torch.Tensor, keep: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A bool mask scales the rows the reductions center."""
    splits, rows, hidden = grad.shape
    out = torch.empty([splits, rows, hidden], dtype=grad.dtype, device=grad.device)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=grad.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            g = grad[tile_s, tile_r, :].to(torch.float32)
            xh = x_hat[tile_s, tile_r, :]
            row_dot = torch.sum(g * xh, dim=-1, keepdim=True)
            total = scale[tile_s, tile_r][:, :, None] * (g - xh * row_dot)
            kept = keep[tile_s, tile_r, :].to(torch.float32) * 1.25
            out[tile_s, tile_r, :] = (total * kept).to(out.dtype)
            acc = acc + torch.sum(g * xh, dim=1)
        partial[tile_s, :] = acc
    return out, partial


def _into_argument(x: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The centered rows go to a tensor the caller passes."""
    splits, rows, hidden = x.shape
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            out[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            acc = acc + torch.sum(row, dim=1)
        partial[tile_s, :] = acc
    return out, partial


def _stored_tile_sums(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Each tile's column sums are stored as well as carried."""
    splits, rows, hidden = x.shape
    out = torch.empty_like(x)
    last = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            out[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            sums = torch.sum(row, dim=1)
            last[tile_s, :] = sums
            acc = acc + sums
        partial[tile_s, :] = acc
    return out, last, partial


def _scaled_rows(
    grad: torch.Tensor, x_hat: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One row store, the row's scale times what its two reductions center."""
    splits, rows, hidden = grad.shape
    out = torch.empty_like(grad)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=grad.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            g = grad[tile_s, tile_r, :]
            xh = x_hat[tile_s, tile_r, :]
            row_sum = torch.sum(g, dim=-1, keepdim=True)
            row_dot = torch.sum(g * xh, dim=-1, keepdim=True)
            out[tile_s, tile_r, :] = scale[tile_s, tile_r][:, :, None] * (
                (g - row_sum) - xh * row_dot
            )
            acc = acc + torch.sum(g * xh, dim=1)
        partial[tile_s, :] = acc
    return out, partial


def _two_folds(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A tile's column sums and maxima feed two partials, stored into the
    halves of one tensor."""
    splits, rows, hidden = x.shape
    out = torch.empty_like(x)
    partials = torch.empty([2, splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        weighted = hl.zeros([tile_s, hidden], dtype=torch.float32)
        largest = hl.full([tile_s, hidden], -1.0e30, dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            out[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            sums = torch.sum(row, dim=1)
            weighted = weighted + sums * torch.amax(row, dim=1)
            largest = torch.maximum(largest, sums)
        partials[0, tile_s, :] = weighted
        partials[1, tile_s, :] = largest
    return out, partials


def _row_norms(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The one row store writes a row's reduction: no element of the row."""
    splits, rows, hidden = x.shape
    norm = torch.empty([splits, rows], dtype=torch.float32, device=x.device)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            norm[tile_s, tile_r] = torch.sum(row * row, dim=-1)
            acc = acc + torch.sum(row, dim=1)
        partial[tile_s, :] = acc
    return norm, partial


def _running_rows(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """A second row store reads the partials ahead of their update."""
    splits, rows, hidden = x.shape
    out = torch.empty_like(x)
    run = torch.empty_like(x)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            out[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            run[tile_s, tile_r, :] = row + acc[:, None, :]
            acc = acc + torch.sum(row, dim=1)
        partial[tile_s, :] = acc
    return out, run, partial


def _decaying_partials(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The partials halve once per tile, ahead of the tile's sums."""
    splits, rows, hidden = x.shape
    out = torch.empty_like(x)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            out[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            acc = acc * 0.5
            acc = acc + torch.sum(row, dim=1)
        partial[tile_s, :] = acc
    return out, partial


def _reduced_behind(x: torch.Tensor) -> torch.Tensor:
    """The values carried over the row loop are reduced behind it."""
    splits, rows, hidden = x.shape
    out = torch.empty([splits], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            acc = acc + torch.sum(x[tile_s, tile_r, :], dim=1)
        out[tile_s] = torch.sum(acc, dim=-1)
    return out


def _kernel(fn: Callable[..., Any]) -> Any:
    return helion.kernel(fn, backend="cute", static_shapes=True, autotune_effort="none")


_KERNELS = {
    fn.__name__: _kernel(fn)
    for fn in (
        _mix_order,
        _row_sums,
        _in_place,
        _reduction_fed_partials,
        _dropout_rows,
        _into_argument,
        _stored_tile_sums,
        _scaled_rows,
        _two_folds,
        _row_norms,
        _running_rows,
        _decaying_partials,
        _reduced_behind,
    )
}
_SPLITS, _SERIAL_ROWS = 5, 3
# Rows a tile of two or four divides.
_TILED_ROWS = 4
# The shared memory the reductions of the modelled B200 may take.
_SMEM = 232448


def _args(
    name: str,
    hidden: int,
    device: torch.device | str = "cpu",
    rows: int = _SERIAL_ROWS,
) -> tuple[Any, ...]:
    generator = torch.Generator().manual_seed(hidden)

    def rand(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(dtype).to(device)

    shape = (_SPLITS, rows, hidden)
    if name == "_mix_order":
        return (
            rand(*shape, dtype=torch.bfloat16),
            rand(hidden),
            rand(*shape),
            rand(_SPLITS, rows),
            rand(*shape),
        )
    if name == "_row_sums":
        return rand(*shape), rand(*shape), rand(*shape, dtype=torch.bfloat16)
    if name == "_in_place":
        return rand(*shape), torch.empty(_SPLITS, hidden, device=device)
    if name == "_dropout_rows":
        return (
            rand(*shape, dtype=torch.bfloat16),
            rand(*shape),
            rand(_SPLITS, rows),
            (rand(*shape) > 0.0),
        )
    if name == "_into_argument":
        return rand(*shape), torch.empty(shape, device=device)
    if name == "_scaled_rows":
        return rand(*shape), rand(*shape), rand(_SPLITS, rows)
    return (rand(*shape),)


@functools.cache
def _bind(name: str, hidden: int, rows: int = _SERIAL_ROWS) -> BoundKernel:
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        return _KERNELS[name].bind(_args(name, hidden, rows=rows))


def _per_block(
    sequence: BlockIdSequence[Any], values: dict[int, object], default: object
) -> list[object]:
    """A config list of ``sequence`` in its own slot order."""
    return [values.get(item.block_ids[0], default) for item in sequence]


def _vector_row(
    bound: BoundKernel,
    threads: int,
    vec: int,
    trip: int = 1,
    *,
    grid: int = 1,
    tile_vec: int = 1,
    **knobs: object,
) -> helion.Config:
    """``trip`` rows per trip of the row loop, on one thread and in vectors
    of ``tile_vec`` rows, each row on ``threads`` threads of ``vec`` columns
    each, ``grid`` splits per CTA on one thread, with ``knobs``."""
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids
    _grid_tile, row_tile = (item.block_ids[0] for item in spec.block_sizes)
    hidden = threads * vec
    config: dict[str, object] = {
        "block_sizes": [grid, trip],
        "num_threads": _per_block(spec.num_threads, {reduction: threads}, 1),
        "cute_vector_widths": _per_block(
            spec.cute_vector_widths, {reduction: vec, row_tile: tile_vec}, 1
        ),
        "cute_reduction_sequence": "scalar",
        **knobs,
    }
    if hidden & (hidden - 1):
        config["cute_unpadded_rows"] = True
    if spec.cute_resident_reduction_blocks:
        config["cute_reduction_schedule"] = "scalar"
    return spec.normalized_config(helion.Config(**config))  # type: ignore[arg-type]


def _render(
    name: str,
    threads: int,
    vec: int,
    rows: int = _SERIAL_ROWS,
    trip: int = 1,
    **knobs: object,
) -> str:
    bound = _bind(name, threads * vec, rows)
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        return bound.to_code(_vector_row(bound, threads, vec, trip, **knobs))


def _device_function(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _row_loops(code: str) -> list[ast.For]:
    """The device loops over the rows of a split in rendered ``code``."""
    return [
        node
        for node in ast.walk(_device_function(code))
        if isinstance(node, ast.For)
        and ast.unparse(node.iter).startswith("range(cutlass.Int32(0)")
    ]


# The thread counts that tile a 768-wide row in whole warps, powers of two or
# not, at every vector width, a 256-wide row on one warp and on two, and a
# 2048-wide one.
@pytest.mark.parametrize(
    ("threads", "vec"),
    [(96, 8), (192, 4), (384, 2), (32, 8), (64, 4), (256, 8), (512, 4)],
)
def test_a_vector_row_runs_the_rows_once(threads: int, vec: int) -> None:
    code = _render("_mix_order", threads, vec)
    assert f"block=({threads}, 1, 1)" in code, code
    (rows,) = _row_loops(code)
    text = ast.unparse(rows)
    # One packet per operand and row, shared by the two lane passes; the
    # residual's follows the reductions with its consumer.
    for tensor in ("grad", "weight", "x_hat", "residual"):
        assert text.count(f"cute.arch.load({tensor}.iterator") == 1, code
    assert ".load()" not in text.replace("scale.layout.stride[1])).load()", ""), code
    # The partials are updated beside the loads, ahead of the reductions.
    update = text.index("total_lane_next.append(")
    reduce = text.index("_lane_acc_reduced = " if threads > 32 else "warp_reduction")
    assert text.index("cute.arch.load(x_hat.iterator") < update < reduce, code
    assert reduce < text.index("cute.arch.load(residual.iterator"), code
    # They are stored once per split, after the rows.
    assert "workspace" not in text and "partials.iterator" not in text, code
    assert code.count("partials.iterator") == 2, code


def test_row_reductions_that_are_only_stored_keep_one_nest() -> None:
    code = _render("_row_sums", 96, 8)
    (rows,) = _row_loops(code)
    text = ast.unparse(rows)
    for tensor in ("grad_weighted", "x_hat", "grad"):
        assert text.count(f"cute.arch.load({tensor}.iterator") == 1, code
    # Every packet is loaded before the reductions; one thread stores them.
    reduce = text.index("_lane_acc_reduced = ")
    assert text.rindex("cute.arch.load(") < reduce, code
    assert text.count("if cutlass.Int32(cute.arch.thread_idx()[0]) % 96 < 1:") == 2
    assert text.count(".store(") == 2, code


def test_configs_the_two_nests_lower_never_reach_the_single_nest() -> None:
    """The default config and a row on one warp of scalar lanes compile as
    before, to the two nests: the single nest is tried only where they
    decline."""
    bound = _bind("_mix_order", 256)
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids
    one_warp = spec.normalized_config(
        helion.Config(
            block_sizes=[1, 1],
            num_threads=[
                32 if item.block_ids[0] == reduction else 1 for item in spec.num_threads
            ],
            cute_reduction_sequence="scalar",
        )
    )
    with (
        cpu_render_target(reduction_smem_budget_bytes=_SMEM),
        patch.object(split, "_lane_carry_nest", side_effect=AssertionError),
    ):
        for config in (spec.default_config(), one_warp):
            code = bound.to_code(config)
            assert "for synthetic_lane_2 in range(8):" in code, code
            assert len(_row_loops(code)) == 2, code


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("_in_place", "a load may read a tensor the nest writes"),
        ("_reduction_fed_partials", "a carried value reads a lane reduction"),
    ],
)
def test_a_kernel_outside_the_proofs_keeps_its_refusal(name: str, reason: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match=reason):
        _render(name, 32, 8)


def test_a_row_the_vectors_do_not_tile_never_forms_the_nest() -> None:
    """A row that is no whole number of vectors keeps its masked scalar
    lanes, which are no trace-time loop: nothing reaches the single nest."""
    bound = _bind("_mix_order", 100)
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids
    config = spec.normalized_config(
        helion.Config(
            block_sizes=[1, 1],
            cute_vector_widths=[
                8 if item.block_ids[0] == reduction else 1
                for item in spec.cute_vector_widths
            ],
            cute_reduction_sequence="scalar",
        )
    )
    with (
        cpu_render_target(reduction_smem_budget_bytes=_SMEM),
        patch.object(split, "_lane_carry_nest", side_effect=AssertionError),
    ):
        code = bound.to_code(config)
    assert "range_constexpr" not in code, code
    assert "< 100" in code, code


# One packed integer per thread: 8, 4 and 2 bytes.
@pytest.mark.parametrize(
    ("threads", "vec", "packet"),
    [(96, 8, "cutlass.Uint64"), (192, 4, "cutlass.Uint32"), (384, 2, "cutlass.Uint16")],
)
def test_a_bool_row_loads_as_byte_packets_under_its_knob(
    threads: int, vec: int, packet: str
) -> None:
    code = _render("_dropout_rows", threads, vec, cute_bool_byte_packets=True)
    (rows,) = _row_loops(code)
    text = ast.unparse(rows)
    assert text.count("cute.arch.load(keep.iterator") == 1, code
    assert f"{packet})" in text, code
    # Each lane keeps the scalar load's compare of its byte against zero.
    assert "& 255) != cutlass.Uint8(0)" in text, code
    assert "keep.layout.stride[2])).load()" not in text, code
    # Off by default: one byte load per element.
    plain = ast.unparse(_row_loops(_render("_dropout_rows", threads, vec))[0])
    assert "cute.arch.load(keep.iterator" not in plain, plain
    assert "keep.layout.stride[2])).load() != cutlass.Uint8(0)" in plain, plain


def test_the_byte_packet_knob_follows_the_vector_row() -> None:
    """A persistent row without a ``reduction_loops`` slot opens the knob on
    its structure; a config whose row runs in scalar lanes turns it off."""
    for hidden in (256, 768):
        bound = _bind("_dropout_rows", hidden)
        spec = bound.config_spec
        assert not spec.reduction_loops.valid_block_ids()
        assert spec.cute_bool_byte_packet_blocks == set(spec.reduction_block_ids)
        assert "cute_bool_byte_packets" in spec._flat_fields()
        on = _vector_row(bound, hidden // 8, 8, cute_bool_byte_packets=True)
        assert on.config["cute_bool_byte_packets"] is True
        assert spec.normalized_config(on) == on
        scalar = spec.normalized_config(
            helion.Config(
                **{**spec.default_config().config, "cute_bool_byte_packets": True}
            )
        )
        assert scalar.config["cute_bool_byte_packets"] is False
    # Without a bool row there is no knob.
    spec = _bind("_mix_order", 256).config_spec
    assert not spec.cute_bool_byte_packet_blocks
    with pytest.raises(exc.InvalidConfig, match="cute_bool_byte_packets requires"):
        spec.normalized_config(
            helion.Config(
                **{**spec.default_config().config, "cute_bool_byte_packets": True}
            )
        )


def _lane_loop_of_rows(
    rows: ast.For, trip: int, constexpr: bool = False
) -> list[ast.For]:
    """The loops over the ``trip`` rows of a tile in the row loop ``rows``."""
    spelled = "cutlass.range_constexpr" if constexpr else "range"
    return [
        node
        for node in ast.walk(rows)
        if isinstance(node, ast.For)
        and ast.unparse(node.target).startswith("lane_")
        and ast.unparse(node.iter) == f"{spelled}({trip})"
    ]


# Two and four rows per trip, on three warps of a row that is no power of
# two, on six and on one.
@pytest.mark.parametrize(
    ("threads", "vec", "trip"), [(96, 8, 2), (192, 4, 4), (32, 8, 2)]
)
def test_a_tile_of_rows_runs_in_the_one_nest(threads: int, vec: int, trip: int) -> None:
    code = _render("_mix_order", threads, vec, rows=_TILED_ROWS, trip=trip)
    assert f"block=({threads}, 1, 1)" in code, code
    (rows,) = _row_loops(code)
    (tile,) = _lane_loop_of_rows(rows, trip)
    text, row = ast.unparse(rows), ast.unparse(tile)
    # One packet per operand and row, as at one row per trip.
    for tensor in ("grad", "weight", "x_hat", "residual"):
        assert row.count(f"cute.arch.load({tensor}.iterator") == 1, code
        assert text.count(f"cute.arch.load({tensor}.iterator") == 1, code
    # The tile's column sums accumulate in each row's load pass ...
    reduce = row.index("_lane_acc_reduced = " if threads > 32 else "warp_reduction")
    assert row.index("_tile_acc_lane_next.append(") < reduce, code
    # ... and reach the partials once per trip, after its rows.
    after = ast.unparse(
        ast.Module(body=rows.body[rows.body.index(tile) + 1 :], type_ignores=[])
    )
    assert "total_lane_next.append(" in after and "_tile_acc" in after, code
    assert "total_lane_next.append(" not in row, code
    assert code.count("partials.iterator") == 2, code


def test_the_lane_unroll_issues_the_loads_of_a_trips_rows_first() -> None:
    code = _render(
        "_mix_order", 96, 8, rows=_TILED_ROWS, trip=_TILED_ROWS, cute_lane_unroll=4
    )
    (rows,) = _row_loops(code)
    loads, compute = (
        ast.unparse(loop)
        for loop in _lane_loop_of_rows(rows, _TILED_ROWS, constexpr=True)
    )
    # The packets the load pass reads, and each row's scale, for all four
    # rows; then the rows, one after another.
    for tensor in ("grad", "weight", "x_hat"):
        assert loads.count(f"cute.arch.load({tensor}.iterator") == 1, code
        assert f"cute.arch.load({tensor}.iterator" not in compute, code
    assert "scale.layout.stride[1])).load()" in loads, code
    assert "_lane_acc" not in loads and ".append(_persistent_branch" in loads, code
    assert "_lane_acc_reduced = " in compute, code
    # The residual's packet follows its row's reductions, with its consumer.
    assert "cute.arch.load(residual.iterator" in compute, code
    # Without the knob the rows stay a rolled loop.
    plain = _render("_mix_order", 96, 8, rows=_TILED_ROWS, trip=_TILED_ROWS)
    assert len(_lane_loop_of_rows(_row_loops(plain)[0], _TILED_ROWS)) == 1, plain
    assert "_lane_unroll_loads" not in plain, plain


def test_row_sums_that_are_only_stored_run_several_rows_per_trip() -> None:
    code = _render("_row_sums", 384, 2, rows=_TILED_ROWS, trip=2, cute_lane_unroll=2)
    (rows,) = _row_loops(code)
    loads, compute = (
        ast.unparse(loop) for loop in _lane_loop_of_rows(rows, 2, constexpr=True)
    )
    assert loads.count("cute.arch.load(") == 3 and "cute.arch.load(" not in compute
    assert compute.count(".store(") == 2, code


@pytest.mark.parametrize(
    ("name", "rows", "reason"),
    [
        # A partial last tile masks its lanes.
        (
            "_mix_order",
            _SERIAL_ROWS,
            "its serial loop does not walk whole tiles of static bounds",
        ),
        ("_stored_tile_sums", _TILED_ROWS, "is no plain assignment"),
        ("_in_place", _TILED_ROWS, "a load may read a tensor the nest writes"),
        (
            "_reduction_fed_partials",
            _TILED_ROWS,
            "a carried value reads a lane reduction",
        ),
    ],
)
def test_a_tile_outside_the_proofs_keeps_its_refusal(
    name: str, rows: int, reason: str
) -> None:
    with pytest.raises(exc.BackendUnsupported, match=reason) as declined:
        _render(name, 32, 8, rows=rows, trip=2)
    assert "reduction marker is nested in a different lane owner" in str(declined.value)


def _seeds(bound: BoundKernel) -> list[helion.Config] | None:
    with bound.env:
        return CuteLaneCarryRowHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def _layout(bound: BoundKernel, seed: helion.Config) -> tuple[int, int, int, int]:
    """``(threads, vec, rows per trip, lane unroll)`` of a seed."""
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids
    return (
        spec.num_threads.config_get(seed.config["num_threads"], reduction),
        spec.cute_vector_widths.config_get(
            seed.config["cute_vector_widths"], reduction
        ),
        seed.config["block_sizes"][1],
        seed.config.get("cute_lane_unroll", 1),
    )


# Rows whole warps of vectors tile, a power of two or not, at every vector
# width one packet holds (2048 columns stop at 16 warps), at one row per trip
# and at each trip the loop's rows divide by.
@pytest.mark.parametrize(
    ("hidden", "rows", "layouts", "trips"),
    [
        (768, _TILED_ROWS, [(96, 8), (192, 4), (384, 2)], [1, 4, 2]),
        (768, _SERIAL_ROWS, [(96, 8), (192, 4), (384, 2)], [1]),
        (256, 2, [(32, 8), (64, 4), (128, 2)], [1, 2]),
        (2048, 8, [(256, 8), (512, 4)], [1, 4, 2]),
    ],
)
def test_the_seeds_plant_the_vector_rows(
    hidden: int, rows: int, layouts: list[tuple[int, int]], trips: list[int]
) -> None:
    bound = _bind("_mix_order", hidden, rows)
    spec = bound.config_spec
    seeds = _seeds(bound)
    assert seeds is not None
    assert [_layout(bound, seed) for seed in seeds] == [
        (threads, vec, trip, trip) for trip in trips for threads, vec in layouts
    ]
    planted = [dict(seed.config) for seed in spec.compiler_seed_configs]
    for seed in seeds:
        assert seed.config["block_sizes"][0] == 1
        assert seed.config.get("cute_unpadded_rows", False) is (hidden == 768)
        assert seed.config["cute_reduction_sequence"] == "scalar"
        assert dict(seed.config) in planted
        # No seed loses a knob to normalization.
        normalized = spec.normalized_config(seed).config
        assert all(normalized[key] == value for key, value in seed.config.items())
    # One seed of each kind renders the single nest: the primary one and the
    # last, a trip of rows where the loop has one.
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        for seed in (seeds[0], seeds[-1]):
            code = bound.to_code(spec.normalized_config(seed))
            threads, _vec, trip, _unroll = _layout(bound, seed)
            assert len(_row_loops(code)) == 1, code
            assert f"block=({threads}, 1, 1)" in code, code
            assert ("_lane_unroll_loads" in code) is (trip > 1), code


@pytest.mark.parametrize(
    ("name", "hidden", "seeded"),
    [
        # No whole warps of vectors tile the row.
        ("_mix_order", 100, 0),
        # The lowering declines these kernels at every config.
        ("_in_place", 768, 0),
        ("_reduction_fed_partials", 768, 0),
        # The tile's sums are stored: one row per trip only.
        ("_stored_tile_sums", 768, 3),
        # The rows go to an argument, proven apart from the input.
        ("_into_argument", 768, 9),
    ],
)
def test_the_seeds_follow_what_the_lowering_proves(
    name: str, hidden: int, seeded: int
) -> None:
    bound = _bind(name, hidden, _TILED_ROWS)
    seeds = _seeds(bound) or []
    assert len(seeds) == seeded
    assert {seed.config["block_sizes"][1] for seed in seeds} == (
        {1, 2, 4} if seeded == 9 else {1} if seeded else set()
    )
    if name in ("_stored_tile_sums", "_into_argument"):
        with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
            for seed in seeds:
                bound.to_code(bound.config_spec.normalized_config(seed))


def test_an_output_that_may_alias_an_input_gets_no_seed() -> None:
    x = torch.randn(_SPLITS, _TILED_ROWS, 768)
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        bound = _KERNELS["_into_argument"].bind((x, x))
    assert _seeds(bound) is None


def test_an_output_that_is_the_input_is_refused() -> None:
    """Storage the kernel both loads and stores has no proof: the vector-row
    config is refused instead of reordering the accesses, at one row per
    trip and at several."""
    y = torch.randn(_SPLITS, _TILED_ROWS, 256)
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        bound = _KERNELS["_into_argument"].bind((y, y))
        for trip in (1, 2):
            with pytest.raises(
                exc.BackendUnsupported, match="a load may read a tensor the nest writes"
            ):
                bound.to_code(_vector_row(bound, 32, 8, trip))


def test_the_rows_knobs_open_on_the_structure() -> None:
    """``cute_unpadded_rows`` and ``cute_lane_unroll`` are searched on the
    row's structure, whatever its length and the loop's: the search space is
    the same at every extent."""
    fingerprints = set()
    for hidden, rows in ((768, _SERIAL_ROWS), (768, _TILED_ROWS), (100, 6), (1536, 2)):
        spec = _bind("_mix_order", hidden, rows).config_spec
        fields = spec._flat_fields()
        assert fields["cute_unpadded_rows"].search_choices == (False, True)
        assert len(fields["cute_lane_unroll"].search_choices) > 1
        fingerprints.add(spec.structural_fingerprint())
    assert len(fingerprints) == 1
    # A power-of-two row has no unpadded layout: the knob is searched there
    # too, and normalizes off.
    bound = _bind("_mix_order", 256, _TILED_ROWS)
    spec = bound.config_spec
    assert spec._flat_fields()["cute_unpadded_rows"].search_choices == (False, True)
    config = _vector_row(bound, 32, 8, cute_unpadded_rows=True)
    assert config.config["cute_unpadded_rows"] is False


def test_a_row_reduced_behind_the_loop_is_no_carry_row() -> None:
    """The structure is a row reduced inside the device loop.  A kernel of
    the same blocks whose loop only carries values, reduced behind it, is
    none: no knob opens on it, no seed is planted and its loop tile keeps a
    vector width under a vector row (a power-of-two row has no
    ``reduction_loops`` slot there either, so only the loop that holds the
    reduction tells the two apart)."""
    for hidden in (256, 768):
        inside = _bind("_mix_order", hidden, _TILED_ROWS).config_spec
        (reduction,) = inside.reduction_block_ids
        assert reduction in inside.cute_innermost_loop_reduction_blocks
        assert search_surface.cute_lane_carry_row(inside) is not None
        behind = _bind("_reduced_behind", hidden, _TILED_ROWS).config_spec
        (reduction,) = behind.reduction_block_ids
        assert reduction not in behind.cute_innermost_loop_reduction_blocks
        assert search_surface.cute_lane_carry_row(behind) is None
        fields = behind._flat_fields()
        assert fields["cute_unpadded_rows"].search_choices == (False,)
        assert fields["cute_lane_unroll"].search_choices == (1,)
        assert "cute_lane_carry_row" not in behind.autotuner_heuristics
    spec = _bind("_reduced_behind", 256, _TILED_ROWS).config_spec
    (reduction,) = spec.reduction_block_ids
    _grid_tile, row_tile = (item.block_ids[0] for item in spec.block_sizes)
    widths = _per_block(spec.cute_vector_widths, {reduction: 8, row_tile: 4}, 1)
    config = spec.normalized_config(
        helion.Config(
            block_sizes=[1, 4],
            num_threads=_per_block(spec.num_threads, {reduction: 32}, 1),
            cute_vector_widths=widths,
        )
    )
    assert config.config["cute_vector_widths"] == widths


def test_a_threaded_sequence_runs_scalar_where_no_tile_holds_its_lanes() -> None:
    """``reload`` and ``resident`` need a tile of two warps' elements: on a
    loop of a few rows no config compiles them, and they normalize to
    ``scalar`` (in a config, and in a flat row after a shrink)."""
    spec = _bind("_mix_order", 768, _TILED_ROWS).config_spec
    for sequence in ("reload", "resident"):
        config = spec.normalized_config(
            helion.Config(
                **{**spec.default_config().config, "cute_reduction_sequence": sequence}
            )
        )
        assert config.config["cute_reduction_sequence"] == "scalar"
    bounded = spec.normalized_config(
        helion.Config(
            **{**spec.default_config().config, "cute_reduction_sequence": "bounded"}
        )
    )
    assert bounded.config["cute_reduction_sequence"] == "bounded"
    assert sequence_tile_holds_warp_lanes([4, 64])
    assert sequence_tile_holds_warp_lanes([None])
    assert not sequence_tile_holds_warp_lanes([4, 32])
    assert not sequence_tile_holds_warp_lanes([])
    with _bind("_mix_order", 768, _TILED_ROWS).env:
        generation = ConfigGeneration(spec)
        flat = generation.default_flat()
        (index,), _ = generation._key_to_flat_indices["cute_reduction_sequence"]
        flat[index] = "resident"
        generation.extension.repair_flat_config(flat)
        assert flat[index] == "scalar"


def test_seeds_that_never_compile_on_the_nest_are_not_planted() -> None:
    """A vector of the row loop's tile on several threads puts the row's
    lanes around the tile's, and a resident schedule rematerializes one
    row store: neither family seeds a nest where it cannot compile."""
    bound = _bind("_mix_order", 2048, 8)
    spec = bound.config_spec
    # The resident family's knobs stay (a config tuned before still loads) ...
    assert spec.cute_resident_reduction_blocks == set(spec.reduction_block_ids)
    assert "cute_reduction_schedule" in spec._flat_fields()
    # ... but the row loop stores two tensors per row: no resident seed.
    heuristics = spec.autotuner_heuristics
    assert "cute_resident_reduction" not in heuristics
    assert "cute_tile_vec" not in heuristics
    assert heuristics == ["cute_lane_carry_row"]
    schedules = {
        seed.config.get("cute_reduction_schedule")
        for seed in spec.compiler_seed_configs
    }
    assert schedules == {"scalar"}
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        for seed in spec.compiler_seed_configs:
            bound.to_code(spec.normalized_config(seed))
    # A row longer than a CTA's threads always walks lanes; a shorter one
    # only beside enough threads of the tile.
    assert _row_lanes_hold_tile(spec, 1)
    wide = _bind("_mix_order", 768, _TILED_ROWS).config_spec
    assert _row_lanes_hold_tile(wide, 2) and not _row_lanes_hold_tile(wide, 1)
    assert not _row_lanes_hold_tile(_bind("_mix_order", 256, 2).config_spec, 4)


# What ``resident_reductions._plan`` needs of the kernel whatever the config,
# one kernel per condition, at a row the resident plans tile (a power of two):
# one row store that writes the row's elements, stores into tensors of their
# own that no load reads, and carries that read no row reduction.
@pytest.mark.parametrize(
    ("name", "seeded"),
    [
        ("_scaled_rows", True),
        # Two row stores.
        ("_running_rows", False),
        # The one row store writes a reduction, no element of the row.
        ("_row_norms", False),
        # Two partials stored into the halves of one tensor.
        ("_two_folds", False),
        # The rows are stored where they are loaded from.
        ("_in_place", False),
        ("_reduction_fed_partials", False),
    ],
)
def test_the_resident_families_seed_what_their_lowering_proves(
    name: str, seeded: bool
) -> None:
    bound = _bind(name, 1024, _TILED_ROWS)
    spec = bound.config_spec
    # The knobs follow the structure, seeded or not: a config tuned before
    # still loads.
    assert spec.cute_resident_reduction_blocks == set(spec.reduction_block_ids)
    assert "cute_reduction_schedule" in spec._flat_fields()
    assert ("cute_resident_reduction" in spec.autotuner_heuristics) is seeded
    resident = {
        (
            seed.config["cute_reduction_schedule"],
            str(seed.config["num_threads"]),
            str(seed.config["cute_vector_widths"]),
            seed.config.get("cute_reduction_pack_output", False),
        ): seed
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_reduction_schedule") in ("resident", "pipelined")
    }
    assert bool(resident) is seeded
    # One seed of each schedule and layout renders, the packed row output
    # included (the row's store is a product).
    assert len(resident) == (6 if seeded else 0)
    with cpu_render_target(reduction_smem_budget_bytes=_SMEM):
        for seed in resident.values():
            assert "_cute_resident" in bound.to_code(spec.normalized_config(seed))


def test_a_split_of_as_many_rows_as_columns_keeps_its_seeds() -> None:
    """A tensor's static shape may repeat the row's length (256 rows of 256
    columns): its loads are no reductions along the row, and the seed gates
    read the carries as they are."""
    bound = _bind("_scaled_rows", 256, 256)
    assert {"cute_resident_reduction", "cute_lane_carry_row"} <= set(
        bound.config_spec.autotuner_heuristics
    )
    seeds = _seeds(bound)
    assert seeds is not None and len(seeds) == 9


def test_the_tile_vector_gives_way_to_rows_per_trip() -> None:
    """At four rows per split the tile-vector family would put a vector of
    the four rows on one thread, which the interchange proves for some row
    bodies only: where the lane-carry family seeds the nest, rows per trip
    included, it plants nothing."""
    bound = _bind("_row_sums", 768, _TILED_ROWS)
    spec = bound.config_spec
    assert spec.autotuner_heuristics == ["cute_lane_carry_row"]
    _grid_tile, row_tile = (item.block_ids[0] for item in spec.block_sizes)
    assert all(
        spec.cute_vector_widths.config_get(seed.config["cute_vector_widths"], row_tile)
        == 1
        for seed in spec.compiler_seed_configs
    )
    assert {seed.config["block_sizes"][1] for seed in spec.compiler_seed_configs} == {
        1,
        2,
        4,
    }


def test_knob_values_the_rows_per_trip_do_not_take_normalize_away() -> None:
    """``cute_lane_unroll`` past the rows of a trip compiles the code of the
    largest factor that divides them, and a vector width on the one-thread
    tile either makes its rows a trace-time loop no pass runs under the
    vector row or changes nothing: both normalize to the value that
    compiles the same code."""
    bound = _bind("_mix_order", 768, _TILED_ROWS)
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids
    _grid_tile, row_tile = (item.block_ids[0] for item in spec.block_sizes)
    for trip, asked, kept in ((2, 16, 2), (4, 8, 4), (4, 2, 2), (1, 4, 1)):
        config = _vector_row(bound, 96, 8, trip, cute_lane_unroll=asked)
        assert config.config["cute_lane_unroll"] == kept, (trip, asked)
    # Scalar lanes of the row are a lane loop the knob may take as it is.
    scalar = spec.normalized_config(
        helion.Config(
            block_sizes=[1, 2],
            num_threads=_per_block(spec.num_threads, {reduction: 32}, 1),
            cute_reduction_sequence="scalar",
            cute_lane_unroll=16,
        )
    )
    assert scalar.config["cute_lane_unroll"] == 16
    plain = _vector_row(bound, 96, 8, 4, cute_lane_unroll=4)
    for width in (2, 4, 16):
        assert _vector_row(bound, 96, 8, 4, tile_vec=width, cute_lane_unroll=4) == plain
    # A tile on threads of its own keeps its width, and so does one under a
    # row that is no vector row.
    threaded = spec.normalized_config(
        helion.Config(
            **{
                **plain.config,
                "num_threads": _per_block(
                    spec.num_threads, {reduction: 96, row_tile: 2}, 1
                ),
                "cute_vector_widths": _per_block(
                    spec.cute_vector_widths, {reduction: 8, row_tile: 2}, 1
                ),
            }
        )
    )
    assert (
        spec.cute_vector_widths.config_get(
            threaded.config["cute_vector_widths"], row_tile
        )
        == 2
    )
    lone = spec.normalized_config(
        helion.Config(
            block_sizes=[1, 4],
            num_threads=_per_block(spec.num_threads, {row_tile: 1}, 0),
            cute_vector_widths=_per_block(spec.cute_vector_widths, {row_tile: 4}, 1),
            cute_reduction_sequence="scalar",
        )
    )
    assert (
        spec.cute_vector_widths.config_get(lone.config["cute_vector_widths"], row_tile)
        == 4
    )
    # What the width used to do under the vector row.
    with (
        cpu_render_target(reduction_smem_budget_bytes=_SMEM),
        patch.object(search_surface, "_normalize_cute_carry_row_tile_vector"),
        pytest.raises(exc.BackendUnsupported, match="a different lane owner"),
    ):
        bound.to_code(_vector_row(bound, 96, 8, 4, tile_vec=4))


def _expected(
    name: str, args: tuple[Any, ...], trip: int = 1
) -> tuple[torch.Tensor, ...]:
    """What eager computes at ``trip`` rows per tile of the row loop (a
    statement runs for the whole tile before the next one)."""
    if name in ("_running_rows", "_decaying_partials", "_two_folds"):
        (x,) = args
        centered = x - x.sum(-1, keepdim=True)
        tiles = x.view(x.shape[0], -1, trip, x.shape[-1])
        sums = tiles.sum(2)
        if name == "_running_rows":
            before = (sums.cumsum(1) - sums).repeat_interleave(trip, dim=1)
            return centered, x + before, x.sum(1)
        if name == "_two_folds":
            weighted = (sums * tiles.amax(2)).sum(1)
            return centered, torch.stack([weighted, sums.amax(1)])
        acc = torch.zeros_like(sums[:, 0])
        for tile in sums.unbind(1):
            acc = acc * 0.5 + tile
        return centered, acc
    if name == "_into_argument":
        (x, _out) = args
        return x - x.sum(-1, keepdim=True), x.sum(1)
    if name == "_mix_order":
        grad, weight, x_hat, scale, residual = args
        g = grad.float()
        gw = g * weight
        y = residual + scale[:, :, None] * (
            (gw - gw.sum(-1, keepdim=True)) - x_hat * (gw * x_hat).sum(-1, keepdim=True)
        )
        return y, y.to(grad.dtype), torch.stack([(g * x_hat).sum(1), g.sum(1)])
    if name == "_dropout_rows":
        grad, x_hat, scale, keep = args
        g = grad.float()
        total = scale[:, :, None] * (g - x_hat * (g * x_hat).sum(-1, keepdim=True))
        return (total * (keep.float() * 1.25)).to(grad.dtype), (g * x_hat).sum(1)
    grad_weighted, x_hat, grad = args
    return (
        grad_weighted.sum(-1),
        (grad_weighted * x_hat).sum(-1),
        (grad.float() * x_hat).sum(1),
    )


@functools.cache
def _compiled(
    name: str, threads: int, vec: int, rows: int = _SERIAL_ROWS
) -> tuple[Any, tuple[Any, ...]]:
    args = _args(name, threads * vec, DEVICE, rows)
    bound = _KERNELS[name].bind(args)
    return bound, args


def _check(
    outputs: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    equal_nan: bool = False,
) -> None:
    for out, ref in zip(outputs, expected, strict=True):
        assert out.dtype == ref.dtype and out.shape == ref.shape
        # A bf16 output may round the other way where the sums' order moves
        # the fp32 value across a tie.
        lowp = out.dtype == torch.bfloat16
        torch.testing.assert_close(
            out.float(),
            ref.float(),
            rtol=2e-2 if lowp else 2e-4,
            atol=2e-2 if lowp else 2e-3,
            equal_nan=equal_nan,
        )


# The geometries of ``test_a_vector_row_runs_the_rows_once`` that differ in
# how the row's sums combine (one warp, two, three whole warps of a row that
# is no power of two) and in the packets (2, 4 and 8 lanes); the row sums
# that are only stored; a second row store that reads the partials ahead of
# their update; and partials of the columns' sums and maxima.
@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("name", "threads", "vec"),
    [
        ("_mix_order", 96, 8),
        ("_mix_order", 192, 4),
        ("_mix_order", 384, 2),
        ("_mix_order", 32, 8),
        ("_mix_order", 64, 4),
        ("_row_sums", 96, 8),
        ("_row_sums", 64, 4),
        ("_running_rows", 96, 8),
        ("_two_folds", 32, 8),
    ],
)
def test_one_nest_matches_eager(name: str, threads: int, vec: int) -> None:
    bound, args = _compiled(name, threads, vec)
    config = _vector_row(bound, threads, vec)
    assert len(_row_loops(bound.to_code(config))) == 1
    _check(bound.compile_config(config)(*args), _expected(name, args))


# Several rows per trip at the geometries above: two and four rows, rolled
# and with a trip's loads issued first (all four rows', and pairs of them),
# the row sums that are only stored, and the bool row's byte packets; then
# the kernels whose result depends on the tile: a row store that reads the
# partials of the tiles before its own, sums and maxima of a tile's rows
# that feed two partials, and partials that decay once per tile.
@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("name", "threads", "vec", "trip", "knobs"),
    [
        ("_mix_order", 96, 8, 2, {}),
        ("_mix_order", 96, 8, 4, {"cute_lane_unroll": 4}),
        ("_mix_order", 192, 4, 4, {"cute_lane_unroll": 2}),
        ("_mix_order", 384, 2, 2, {"cute_lane_unroll": 2}),
        ("_mix_order", 32, 8, 4, {"cute_lane_unroll": 4}),
        ("_row_sums", 384, 2, 2, {"cute_lane_unroll": 2}),
        (
            "_dropout_rows",
            96,
            8,
            4,
            {"cute_lane_unroll": 4, "cute_bool_byte_packets": True},
        ),
        ("_running_rows", 96, 8, 2, {"cute_lane_unroll": 2}),
        ("_two_folds", 32, 8, 2, {}),
        ("_decaying_partials", 64, 4, 4, {"cute_lane_unroll": 4}),
    ],
)
def test_rows_per_trip_match_eager(
    name: str, threads: int, vec: int, trip: int, knobs: dict[str, object]
) -> None:
    bound, args = _compiled(name, threads, vec, _TILED_ROWS)
    config = _vector_row(bound, threads, vec, trip, **knobs)
    code = bound.to_code(config)
    (rows,) = _row_loops(code)
    unroll = knobs.get("cute_lane_unroll", 1)
    assert len(_lane_loop_of_rows(rows, unroll, constexpr=True)) == (
        2 if unroll > 1 else 0
    ), code
    _check(bound.compile_config(config)(*args), _expected(name, args, trip))


@pytest.mark.cuda
@skipIfNotCUDA()
def test_one_nest_matches_eager_at_its_edges() -> None:
    """Two splits per CTA, walked by its threads one after the other, with
    a partial last pair; rows of NaN and of infinities; an output the caller
    passes as a permuted view; and a bool row that starts eight and three
    bytes into its storage under byte packets."""
    bound, args = _compiled("_mix_order", 96, 8)
    config = _vector_row(bound, 96, 8, grid=2)
    assert "for lane_0 in range(2):" in (code := bound.to_code(config)), code
    assert len(_row_loops(code)) == 1, code
    _check(bound.compile_config(config)(*args), _expected("_mix_order", args))
    grad, weight, x_hat, scale, residual = args
    x_hat, residual = x_hat.clone(), residual.clone()
    x_hat[1, 0, 5] = float("nan")
    x_hat[2, 1, 700] = float("inf")
    residual[3, 2] = float("-inf")
    odd = (grad, weight, x_hat, scale, residual)
    _check(
        bound.compile_config(_vector_row(bound, 96, 8))(*odd),
        _expected("_mix_order", odd),
        equal_nan=True,
    )
    hidden = 768
    x = torch.randn(_SPLITS, _SERIAL_ROWS, hidden, device=DEVICE)
    turned = torch.full((hidden, _SERIAL_ROWS, _SPLITS), -7.0, device=DEVICE)
    out_args = (x, turned.permute(2, 1, 0))
    bound = _KERNELS["_into_argument"].bind(out_args)
    config = _vector_row(bound, 96, 8)
    assert len(_row_loops(bound.to_code(config))) == 1
    _check(
        bound.compile_config(config)(*out_args), _expected("_into_argument", out_args)
    )
    grad, x_hat, scale, keep = _args("_dropout_rows", hidden, DEVICE)
    # A packet's width into the storage the rows load as packets; three
    # bytes in no packet is aligned, and the bytes load one by one.
    for offset, packets in ((8, True), (3, False)):
        shifted = torch.zeros(keep.numel() + offset, dtype=torch.bool, device=DEVICE)
        shifted[offset:] = keep.flatten()
        mask_args = (grad, x_hat, scale, shifted[offset:].view(keep.shape))
        bound = _KERNELS["_dropout_rows"].bind(mask_args)
        config = _vector_row(bound, 96, 8, cute_bool_byte_packets=True)
        assert ("cutlass.Uint64)" in bound.to_code(config)) is packets
        _check(
            bound.compile_config(config)(*mask_args),
            _expected("_dropout_rows", (grad, x_hat, scale, keep)),
        )


@pytest.mark.cuda
@skipIfNotCUDA()
def test_byte_packets_change_no_bit() -> None:
    bound, args = _compiled("_dropout_rows", 96, 8)
    plain = bound.compile_config(_vector_row(bound, 96, 8))(*args)
    config = _vector_row(bound, 96, 8, cute_bool_byte_packets=True)
    assert "cutlass.Uint64)" in bound.to_code(config)
    packed = bound.compile_config(config)(*args)
    for out, ref in zip(packed, plain, strict=True):
        assert torch.equal(out, ref)
    _check(packed, _expected("_dropout_rows", args))


@pytest.mark.cuda
@skipIfNotCUDA()
def test_one_nest_matches_eager_on_views() -> None:
    """Offset, strided and permuted views of larger buffers: the operands
    whose rows are not contiguous packets fall back to their scalar loads,
    and a split that starts past the buffer's base keeps its rows."""
    hidden = 768
    grad, weight, x_hat, scale, residual = _args("_mix_order", hidden, DEVICE)
    padded = torch.zeros(_SPLITS + 2, _SERIAL_ROWS, hidden, device=DEVICE)
    padded[2:] = x_hat
    wide = torch.zeros(_SPLITS, 2 * _SERIAL_ROWS, hidden, device=DEVICE)
    wide[:, ::2] = residual
    turned = grad.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    # Rows that start one element into rows of 769: no packet is aligned.
    skewed = torch.zeros(_SPLITS, _SERIAL_ROWS, hidden + 1, device=DEVICE)
    skewed[..., 1:] = x_hat
    expected = _expected("_mix_order", (grad, weight, x_hat, scale, residual))
    for args in (
        (turned, weight, padded[2:], scale, wide[:, ::2]),
        (grad, weight, skewed[..., 1:], scale, residual),
    ):
        bound = _KERNELS["_mix_order"].bind(args)
        config = _vector_row(bound, 96, 8)
        assert len(_row_loops(bound.to_code(config))) == 1
        _check(bound.compile_config(config)(*args), expected)


@pytest.mark.cuda
@skipIfNotCUDA()
def test_rows_per_trip_match_eager_on_views() -> None:
    """The views of ``test_one_nest_matches_eager_on_views`` at two rows per
    trip with the trip's loads first, and an output the caller passes as a
    strided view of a larger buffer."""
    hidden, rows = 768, _TILED_ROWS
    grad, weight, x_hat, scale, residual = _args("_mix_order", hidden, DEVICE, rows)
    padded = torch.zeros(_SPLITS + 2, rows, hidden, device=DEVICE)
    padded[2:] = x_hat
    wide = torch.zeros(_SPLITS, 2 * rows, hidden, device=DEVICE)
    wide[:, ::2] = residual
    turned = grad.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    skewed = torch.zeros(_SPLITS, rows, hidden + 1, device=DEVICE)
    skewed[..., 1:] = x_hat
    expected = _expected("_mix_order", (grad, weight, x_hat, scale, residual))
    for args in (
        (turned, weight, padded[2:], scale, wide[:, ::2]),
        (grad, weight, skewed[..., 1:], scale, residual),
    ):
        bound = _KERNELS["_mix_order"].bind(args)
        config = _vector_row(bound, 96, 8, 2, cute_lane_unroll=2)
        (loop,) = _row_loops(bound.to_code(config))
        assert len(_lane_loop_of_rows(loop, 2, constexpr=True)) == 2
        _check(bound.compile_config(config)(*args), expected)
    x = torch.randn(_SPLITS, rows, hidden, device=DEVICE)
    buffer = torch.full((_SPLITS, 2 * rows, hidden), -7.0, device=DEVICE)
    bound = _KERNELS["_into_argument"].bind((x, buffer[:, 1::2]))
    config = _vector_row(bound, 96, 8, 2, cute_lane_unroll=2)
    out, partial = bound.compile_config(config)(x, buffer[:, 1::2])
    torch.testing.assert_close(out, x - x.sum(-1, keepdim=True), rtol=2e-4, atol=2e-3)
    torch.testing.assert_close(partial, x.sum(1), rtol=2e-4, atol=2e-3)
    # The rows between the output's are untouched.
    assert torch.all(buffer[:, 0::2] == -7.0)
