"""CPU unit tests of ``cute/jam_grid_lanes.py``'s proofs on hand-built ASTs.

The codegen tests (``test_cute_jam_grid_lanes_codegen.py``) render kernels;
the forms here are the ones no kernel makes on purpose, where a proof must
refuse what it cannot show.
"""

from __future__ import annotations

import ast
import itertools
import textwrap
from typing import Any
from typing import cast

import pytest
import torch

from helion._compiler.cute.jam_grid_lanes import _Decline
from helion._compiler.cute.jam_grid_lanes import _ElideResidueBounds
from helion._compiler.cute.jam_grid_lanes import _Jam
from helion._compiler.cute.jam_grid_lanes import _LoadReuse

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("before", "body", "proven"),
    [
        ("", "    v = base + lane", True),
        ("if c:\n    base = base + 1\n", "    v = base + lane", False),
        ("", "    base = base + 1", False),
    ],
    ids=["defined_once", "rebound_before_the_loop", "rebound_in_the_loop"],
)
def test_a_lane_base_rebound_is_no_proven_multiple(
    before: str, body: str, proven: bool
) -> None:
    # The packets need the lane base a multiple of the lane count; a name
    # bound again, in a branch before the loop or in the loop, may hold
    # anything.
    *preceding, loop = ast.parse(
        f"base = pid * 8\n{before}for lane in range(8):\n{body}\n"
    ).body
    jam = _Jam(
        cast("ast.For", loop),
        preceding=preceding,
        following=[],
        rename_groups={},
        tensors={},
        proven_disjoint_tensor_pairs=(),
        constexpr_values={},
        env=cast("Any", None),
        new_var=str,
    )
    assert jam._multiple_of(ast.Name(id="base"), 8) == proven


def _jam_on(tensors: dict[str, torch.Tensor]) -> _Jam:
    """A jam of an eight-lane loop over ``tensors`` whose lane index is ``i``."""
    (loop,) = ast.parse("for lane in range(8):\n    pass\n").body
    jam = _Jam(
        cast("ast.For", loop),
        preceding=[],
        following=[],
        rename_groups={},
        tensors=tensors,
        proven_disjoint_tensor_pairs=(),
        constexpr_values={},
        env=cast("Any", None),
        new_var=str,
    )
    jam.lane_index = "i"
    return jam


def _store(first: str, second: str) -> str:
    return (
        f"(out.iterator + cutlass.Int32({first}) * cutlass.Int32(out.layout.stride[0])"
        f" + cutlass.Int32({second}) * cutlass.Int32(out.layout.stride[1])).store(v)"
    )


@pytest.mark.parametrize(
    ("second", "declined"),
    [(("i", "j"), False), (("j", "i"), True)],
    ids=["same_lane_dimension", "transposed"],
)
def test_stores_into_one_tensor_share_a_lane_dimension(
    second: tuple[str, str], declined: bool
) -> None:
    # ``out[i, j]`` and ``out[j, i]``: lanes ``i`` and ``j`` both store to
    # elements of the other's row and column, in another order once jammed.
    body = ast.parse(f"{_store('i', 'j')}\n{_store(*second)}\n").body
    jam = _jam_on({"out": torch.empty(16, 16)})
    if declined:
        with pytest.raises(_Decline, match="do not all address one dimension"):
            jam._check_memory(body)
    else:
        jam._check_memory(body)


def _load(first: str, second: str) -> str:
    return (
        f"v = (out.iterator + cutlass.Int32({first}) * cutlass.Int32(out.layout.stride[0])"
        f" + cutlass.Int32({second}) * cutlass.Int32(out.layout.stride[1])).load()"
    )


@pytest.mark.parametrize(
    ("load", "declined"),
    [(("i", "k"), False), (("k", "i"), True), (("0", "k"), True)],
    ids=["own_row", "another_lanes_row", "a_fixed_row"],
)
def test_a_stored_tensor_is_loaded_only_from_the_lanes_own_elements(
    load: tuple[str, str], declined: bool
) -> None:
    # Each lane stores its row ``out[i, :]``; ``out[k, i]`` and ``out[0, k]``
    # read rows other lanes store, in another order once jammed.
    body = ast.parse(f"{_store('i', 'k')}\n{_load(*load)}\n").body
    jam = _jam_on({"out": torch.empty(16, 16)})
    if declined:
        with pytest.raises(_Decline, match="do not all address one dimension"):
            jam._check_memory(body)
    else:
        jam._check_memory(body)


@pytest.mark.parametrize(
    ("out", "load", "reason"),
    [
        # Rows 8 elements apart: ``out[i, 8]`` is ``out[i + 1, 0]``.
        (
            torch.empty(136).as_strided((16, 16), (8, 1)),
            _load("i", "k"),
            "two of whose index tuples may address one element",
        ),
        # An address with a term that is no index times a stride.
        (
            torch.empty(16, 16),
            (
                "v = (out.iterator + cutlass.Int32(i) * "
                "cutlass.Int32(out.layout.stride[0]) + 1).load()"
            ),
            "an access whose indices the jam does not parse",
        ),
    ],
    ids=["overlapping_rows", "unparsed_load"],
)
def test_a_stored_tensor_the_lanes_may_share_declines_with_its_cause(
    out: torch.Tensor, load: str, reason: str
) -> None:
    body = ast.parse(f"{_store('i', 'k')}\n{load}\n").body
    with pytest.raises(_Decline, match=reason):
        _jam_on({"out": out})._check_memory(body)


def test_a_load_the_jam_does_not_parse_declines() -> None:
    # Its tensor would escape the stored-and-loaded checks.
    with pytest.raises(_Decline, match="not a load, store"):
        _jam_on({"x": torch.empty(16)})._check_memory(ast.parse("v = x.load(i)").body)


def test_a_tensor_accessed_by_subscript_declines() -> None:
    # CuTe's ``torch.gather`` reads ``x[g, k]`` straight from the tensor.
    # Beside a store into ``x``, one lane may read the row another stores,
    # before the store in the rolled loop and after it once jammed.
    body = ast.parse(
        "v = x[g, k]\n"
        "(x.iterator + cutlass.Int32(i) * cutlass.Int32(x.layout.stride[0])"
        " + cutlass.Int32(k) * cutlass.Int32(x.layout.stride[1])).store(v)\n"
    ).body
    with pytest.raises(_Decline, match="accessed by subscript"):
        _jam_on({"x": torch.empty(16, 16)})._check_memory(body)


def test_a_reduce_that_fills_a_buffer_in_place_declines() -> None:
    # The fragment reduce writes ``res``, which no name the jam renames
    # records: each lane's copy would refill one buffer before any lane read
    # its results.
    body = ast.parse(
        "_cute_grouped_reduce_shared_two_stage_fragment(vals, res, 'sum',"
        " cutlass.Float32(0.0), t, g, p, count=2, pre=1, group_span=64,"
        " group_count=1)\n"
    ).body
    with pytest.raises(_Decline, match="not a load, store"):
        _jam_on({})._check_memory(body)


def _lane_jam(
    source: str,
    tensors: dict[str, torch.Tensor],
    constexpr_values: dict[str, int] | None = None,
) -> _Jam:
    """The jam of ``source``'s last statement, an eight-lane loop, after the
    statements before it."""
    *preceding, loop = ast.parse(textwrap.dedent(source)).body
    return _Jam(
        cast("ast.For", loop),
        preceding=preceding,
        following=[],
        rename_groups={},
        tensors=tensors,
        proven_disjoint_tensor_pairs=(),
        constexpr_values=constexpr_values or {},
        env=cast("Any", None),
        new_var=str,
    )


@pytest.mark.parametrize(
    ("statements", "reason"),
    [
        ("k = i\ni = 0\ni = k", "lane index is rebound"),
        ("lane = 0", "rebinds the lane variable"),
    ],
    ids=["lane_index", "lane_variable"],
)
def test_a_rebound_lane_index_or_variable_declines(
    statements: str, reason: str
) -> None:
    # A store indexed by the lane index reaches only the lane's own elements
    # while the index holds the lane's tile index, and each lane's copy
    # replaces the lane variable by its literal.
    jam = _lane_jam(
        "for lane in range(8):\n    i = base + cutlass.Int32(lane)\n"
        + textwrap.indent(statements, "    "),
        {},
    )
    with pytest.raises(_Decline, match=reason):
        jam._find_lane_index(jam.loop.body)


@pytest.mark.parametrize(
    "statements",
    [
        "if lane == 1:\n    b = base - 1\nelse:\n    b = base\ni = b + cutlass.Int32(lane)",
        "i = base + cutlass.Int32(lane)\nfor k in range(n):\n    v = cutlass.Int32(lane)",
        "i = base + cutlass.Int32(lane)\nv, w = lane, lane",
    ],
    ids=["branch_test", "nested_loop", "tuple_assignment"],
)
def test_only_the_lane_index_reads_the_lane_variable(statements: str) -> None:
    # The store proof takes ``base + lane`` for the only part of an address
    # that differs per lane: the branch gives lanes 0 and 1 one index.
    jam = _lane_jam("for lane in range(8):\n" + textwrap.indent(statements, "    "), {})
    with pytest.raises(_Decline, match="another statement reads the lane variable"):
        jam._find_lane_index(jam.loop.body)


def test_a_lane_base_that_differs_per_lane_declines() -> None:
    # Every lane runs the reduce, so ``b`` is each lane's own name, and the
    # lanes' indices ``b + lane`` need one ``b`` to differ.
    jam = _lane_jam(
        """
        for lane in range(8):
            b = cutlass.Int32(cute.arch.warp_reduction_sum(t, threads_in_group=32))
            i = b + cutlass.Int32(lane)
            (out.iterator + cutlass.Int32(i) * cutlass.Int32(out.layout.stride[0])).store(t)
        """,
        {"out": torch.empty(64)},
    )
    with pytest.raises(_Decline, match="base differs per lane"):
        jam.run()


def test_the_lane_variable_read_after_the_loop_declines() -> None:
    # The rolled loop leaves its last lane in ``lane``; the jammed lanes bind
    # no lane variable.
    loop, after = ast.parse(
        "for lane in range(8):\n    i = base + cutlass.Int32(lane)\nv = lane\n"
    ).body
    jam = _Jam(
        cast("ast.For", loop),
        preceding=[],
        following=[after],
        rename_groups={},
        tensors={},
        proven_disjoint_tensor_pairs=(),
        constexpr_values={},
        env=cast("Any", None),
        new_var=str,
    )
    with pytest.raises(_Decline, match="read after it"):
        jam.run()


@pytest.mark.parametrize(
    "statement",
    ["x[i][j] = v", "a, x[i] = (v, w)", "x.attr = v", "x.attr += v"],
    ids=["nested_subscript", "subscript_in_a_tuple", "attribute", "augmented"],
)
def test_an_update_in_place_through_another_target_declines(statement: str) -> None:
    # ``_mutated_receivers`` records ``x[i] = v`` as an update of ``x``; each
    # of these updates ``x`` too, which would stay lane-invariant.
    jam = _lane_jam(f"for lane in range(8):\n    {statement}\n", {})
    with pytest.raises(_Decline, match="target the jam does not follow"):
        jam._check_forms(jam.loop.body)


@pytest.mark.parametrize(
    "statements",
    [
        "a = y\na[0] = v",
        "row = x[i]\nrow[j] = v",
        "a = y\na += [v]",
        "a = y\na.append(v)",
        "a, b = (y, z)\na[0] = v",
        "a, unused = y, 0\na[0] = v",
        "a = y if c else z\na[0] = v",
        "a = y[0][0]\na[0] = v",
        "a = []\nb = a\na.append(v)",
        "a = []\nb, c = a, 0\na.append(v)",
        "a = []\nb = [a]\na.append(v)",
    ],
    ids=[
        "alias",
        "element",
        "augmented",
        "append",
        "tuple",
        "tuple_with_a_constant",
        "conditional",
        "nested_element",
        "copy_of_an_updated_list",
        "tuple_holding_an_updated_list",
        "display_holding_an_updated_list",
    ],
)
def test_an_update_in_place_through_an_alias_declines(statements: str) -> None:
    # The update changes ``y`` (or ``x``'s row) too, which the register proof
    # would leave lane-invariant: one copy of a later ``w = y[0]`` would read
    # the last lane's update.  A list the body updates may not be taken
    # whole by another name either.
    jam = _lane_jam("for lane in range(8):\n" + textwrap.indent(statements, "    "), {})
    with pytest.raises(_Decline, match="update in place through an alias"):
        jam._check_aliases(jam.loop.body)


@pytest.mark.parametrize(
    "alias",
    ["a = y", "a, unused = (y, 0)", "a = y if i < 0 else y", "a = [y][0]"],
    ids=["copy", "tuple", "conditional", "element"],
)
def test_an_aliased_list_updated_per_lane_declines_the_jam(alias: str) -> None:
    # Jammed, ``y = [...]`` would run once and every lane's ``a`` would name
    # it, so all eight stores would write the last lane's ``v``.
    jam = _lane_jam(
        f"""
        for lane in range(8):
            i = base + cutlass.Int32(lane)
            y = [cutlass.Float32(0.0)]
            v = (x.iterator + cutlass.Int32(i) * cutlass.Int32(x.layout.stride[0])).load()
            {alias}
            a[0] = v
            w = y[0]
            (out.iterator + cutlass.Int32(i) * cutlass.Int32(out.layout.stride[0])).store(w)
        """,
        {"x": torch.empty(64), "out": torch.empty(64)},
    )
    jam.proven_disjoint_tensor_pairs = (frozenset(("x", "out")),)
    with pytest.raises(_Decline, match="update in place through an alias"):
        jam.run()


@pytest.mark.parametrize(
    ("preceding", "definition"),
    [
        # Another member of the base's rename group binds it again.
        ("base = 8 * k\nbase_1 = cutlass.Int32(3)", None),
        # The group's member bound before the loop is the base the body reads.
        ("base_1 = 8 * k", "8 * k"),
    ],
    ids=["rebound_through_a_member", "bound_through_a_member"],
)
def test_a_lane_base_definition_compares_names_through_their_rename_group(
    preceding: str, definition: str | None
) -> None:
    *before, loop = ast.parse(
        textwrap.dedent(preceding) + "\nfor lane in range(8):\n    i = base + lane\n"
    ).body
    jam = _Jam(
        cast("ast.For", loop),
        preceding=before,
        following=[],
        rename_groups={"base_1": "base"},
        tensors={},
        proven_disjoint_tensor_pairs=(),
        constexpr_values={},
        env=cast("Any", None),
        new_var=str,
    )
    value = jam._definition("base")
    assert (None if value is None else ast.unparse(value)) == definition


@pytest.mark.parametrize(
    "binding",
    ["t = x", "t, u = (x, y)", "t = x if c else y", "t = [x][0]"],
    ids=["copy", "tuple", "conditional", "element"],
)
def test_a_tensor_bound_to_another_name_declines(binding: str) -> None:
    # The memory proof follows a tensor by its own name only: ``t[g, k]``
    # would read ``x`` past the subscript check.
    jam = _lane_jam(f"for lane in range(8):\n    {binding}\n    v = t[g, k]\n", {})
    jam.tensors = {"x": torch.empty(16, 16), "y": torch.empty(16, 16)}
    with pytest.raises(_Decline, match="tensor bound to another name"):
        jam._check_aliases(jam.loop.body)


@pytest.mark.parametrize(
    "statements",
    [
        "a = y\nw = a[0]",
        "a = []\na.append(v)\nw = a[0]",
        "acc = cutlass.Float32(0.0)\nacc += v\ntotal = acc",
    ],
    ids=["copy_only_read", "fresh_list", "augmented_scalar"],
)
def test_names_that_share_no_updated_object_are_no_alias(statements: str) -> None:
    # Reading through a copy changes nothing another name holds; a list the
    # body makes is each lane's own; a scalar's ``+=`` makes a new value.
    jam = _lane_jam("for lane in range(8):\n" + textwrap.indent(statements, "    "), {})
    jam._check_aliases(jam.loop.body)


def test_a_reduce_of_a_lane_invariant_value_updates_each_lanes_own_copy() -> None:
    # Every lane runs the reduce, which is no proven pure computation: eight
    # copies updating one ``t`` would reduce it eight times per trip, where
    # each lane of the rolled loop reduces its own ``t`` once.
    jam = _lane_jam(
        """
        for lane in range(8):
            i = base + cutlass.Int32(lane)
            t = cutlass.Float32(0.0)
            for k in range(n):
                t = cutlass.Float32(cute.arch.warp_reduction_sum(t, threads_in_group=32)) + cutlass.Float32(1.0)
            (out.iterator + cutlass.Int32(i) * cutlass.Int32(out.layout.stride[0])).store(t)
        """,
        {"out": torch.empty(64)},
    )
    code = ast.unparse(ast.Module(body=jam.run(), type_ignores=[]))
    for lane in range(8):
        assert f"t_lane{lane} = cutlass.Float32(0.0)" in code
        assert (
            f"t_lane{lane} = cutlass.Float32(cute.arch.warp_reduction_sum(t_lane{lane},"
            in code
        )


def test_a_loop_whose_variable_differs_per_lane_runs_per_lane() -> None:
    # The branch sets ``k`` per lane, and the store reads the ``k`` the loop
    # leaves: each lane's copy of the loop binds a ``k`` of its own, so the
    # count its body updates is the lane's own too (one ``total`` that every
    # lane's copy adds to stores 26 instead of 5 at ``n = 3``).
    jam = _lane_jam(
        """
        for lane in range(8):
            i = base + cutlass.Int32(lane)
            total = cutlass.Int32(0)
            k = cutlass.Int32(0)
            row = i * cutlass.Int32(2)
            if row > c:
                k = cutlass.Int32(1)
            for k in range(n):
                total = total + cutlass.Int32(1)
            (out.iterator + cutlass.Int32(i) * cutlass.Int32(out.layout.stride[0])).store(total + k)
        """,
        {"out": torch.empty(64)},
    )
    code = ast.unparse(ast.Module(body=jam.run(), type_ignores=[]))
    for lane in range(8):
        assert f"total_lane{lane} = total_lane{lane} + cutlass.Int32(1)" in code


@pytest.mark.parametrize(("pre", "batched"), [(1, True), (0, False)])
def test_reduces_batch_only_with_siblings(pre: int, batched: bool) -> None:
    # ``pre == 0`` passes the power-of-two test ``pre & (pre - 1) == 0``.
    group = [
        ast.parse(
            f"r_{lane} = _cute_grouped_reduce_shared_two_stage(v_{lane}, 'sum',"
            f" cutlass.Float32(0.0), t, g, p, pre={pre}, group_span=64,"
            " group_count=1)"
        ).body[0]
        for lane in range(8)
    ]
    assert (_jam_on({})._batched_reduce(group) is not None) == batched


@pytest.mark.parametrize(
    ("branch", "own"),
    [
        (
            "if d:\n    a = cutlass.Float32(1.0)\nelse:\n    a = cutlass.Float32(2.0)",
            True,
        ),
        (
            (
                "if d:\n    if e:\n        a = cutlass.Float32(1.0)\n"
                "else:\n    a = cutlass.Float32(2.0)"
            ),
            False,
        ),
        ("for k in range(m):\n    a = cutlass.Float32(1.0)", False),
    ],
    ids=["every_path", "nested_branch", "loop"],
)
def test_a_branch_owns_only_names_it_writes_on_every_path(
    branch: str, own: bool
) -> None:
    # A branch whose test differs per lane renames the names it writes before
    # reading them.  ``b = a`` reads the ``a`` of before the branch when no
    # nested arm or loop trip wrote it, which a name of the branch's own would
    # leave unbound: such names vary instead.
    source = textwrap.dedent(
        """
        for lane in range(8):
            i = base + cutlass.Int32(lane)
            a = cutlass.Float32(7.0)
            v = (x.iterator + cutlass.Int32(i) * cutlass.Int32(x.layout.stride[0])).load()
            if v > 0:
        BRANCH
                b = a
                (out.iterator + cutlass.Int32(i) * cutlass.Int32(out.layout.stride[0])).store(b)
        """
    ).replace("BRANCH", textwrap.indent(branch, " " * 8))
    jam = _lane_jam(source, {"x": torch.empty(64), "out": torch.empty(64)})
    varying = jam._varying_names(jam.loop.body)
    assert ("a" in varying) != own
    assert bool(jam.region_names) == own


def test_packets_take_no_load_from_a_loop_or_branch_body() -> None:
    # A lane's copy of a branch or loop may not run its body (the test or the
    # trip count may keep an invalid address from being loaded), or may store
    # an element and load it back: a packet goes ahead of all the copies.
    for guarded in (
        "if v > 0:\n    w = (x.iterator + i).load()\n",
        "for k in range(v):\n    w = (x.iterator + i).load()\n",
        "if c:\n    (x.iterator + i).store(2.0)\n    w = (x.iterator + i).load()\n",
    ):
        (stmt,) = ast.parse(guarded).body
        assert _Jam._scalar_loads(stmt) == []
    (load,) = ast.parse("w = (x.iterator + i).load()").body
    assert len(_Jam._scalar_loads(load)) == 1


def _reuse(source: str, tensors: dict[str, torch.Tensor] | None = None) -> str:
    """``source``, statements of a jammed body over ``tensors``, after
    ``_LoadReuse``."""
    counter = itertools.count()
    reuse = _LoadReuse(
        {}, lambda name: f"{name}_{next(counter)}", tensors or {}, frozenset()
    )
    body = reuse.run(ast.parse(textwrap.dedent(source)).body)
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_load_reuse_serves_loads_of_the_same_address() -> None:
    # A device loop that runs once becomes straight-line code; a constexpr
    # loop's loads serve a later constexpr loop of the same extent.
    code = _reuse(
        """
        for t in range(0, 1024, 1024):
            a = (x.iterator + t).load()
        for u in range(0, 1024, 1024):
            b = (x.iterator + u).load()
        for i in cutlass.range_constexpr(4):
            c = (y.iterator + i).load()
        for j in cutlass.range_constexpr(4):
            d = (y.iterator + j).load()
        """
    )
    assert "b = a" in code
    assert "_jam_stash_0.append(c)" in code
    assert "d = _jam_stash_0[j]" in code


# Each case changes an address or the earlier load's value between the two
# loads, or keeps them in scopes the reuse does not connect.
@pytest.mark.parametrize(
    ("source", "kept"),
    [
        (
            # ``s`` carries from one iteration to the next: the loop's
            # first load reads x[0], its later ones x[1], x[2], x[3].
            """
            s = cutlass.Int32(0)
            first = (x.iterator + s).load()
            for i in cutlass.range_constexpr(4):
                v = (x.iterator + s).load()
                s = s + 1
            """,
            "v = (x.iterator + s).load()",
        ),
        (
            # The first loop reads x[i], the second x[0] in every iteration.
            """
            s = cutlass.Int32(0)
            t = cutlass.Int32(0)
            for i in cutlass.range_constexpr(4):
                a = (x.iterator + s).load()
                s = s + 1
            for j in cutlass.range_constexpr(4):
                b = (x.iterator + t).load()
            """,
            "b = (x.iterator + t).load()",
        ),
        (
            # ``j`` is served the first loop's ``i`` of the same iteration,
            # not ``i``'s value after the loop that ``z`` gathers by.
            """
            for v in cutlass.range_constexpr(4):
                i = (idx.iterator + v).load()
            z = (x.iterator + i).load()
            for w in cutlass.range_constexpr(4):
                j = (idx.iterator + w).load()
                y = (x.iterator + j).load()
            """,
            "y = (x.iterator + j).load()",
        ),
        (
            # ``b`` is served what ``a`` loaded, not the zero ``a`` holds
            # by then.
            """
            for i in cutlass.range_constexpr(2):
                a = (idx.iterator + i).load()
            a = cutlass.Int32(0)
            z = (x.iterator + a).load()
            for j in cutlass.range_constexpr(2):
                b = (idx.iterator + j).load()
                w = (x.iterator + b).load()
            """,
            "w = (x.iterator + b).load()",
        ),
        (
            """
            a = (x.iterator + p).load()
            if c:
                p = p + 1
            b = (x.iterator + p).load()
            """,
            "b = (x.iterator + p).load()",
        ),
        (
            """
            a = (x.iterator + p).load()
            for k in range(n):
                p = p + 1
            b = (x.iterator + p).load()
            """,
            "b = (x.iterator + p).load()",
        ),
        (
            # One address in loops of two extents: a stash of two values
            # cannot serve four iterations.
            """
            for i in cutlass.range_constexpr(2):
                a = (x.iterator + p).load()
            for j in cutlass.range_constexpr(4):
                b = (x.iterator + p).load()
            """,
            "b = (x.iterator + p).load()",
        ),
        (
            # ``a`` no longer holds the loaded value.
            """
            a = (x.iterator + p).load()
            a = a * 2.0
            b = (x.iterator + p).load()
            """,
            "b = (x.iterator + p).load()",
        ),
        (
            # Loops that run twice stay rolled.
            """
            for t in range(0, 2048, 1024):
                a = (x.iterator + t).load()
            for u in range(0, 2048, 1024):
                b = (x.iterator + u).load()
            """,
            "b = (x.iterator + u).load()",
        ),
        (
            # The stash fills in iteration order, but iteration ``j`` of
            # the second loop loads element ``3 - j``.
            """
            for i in cutlass.range_constexpr(4):
                i = 3 - i
                a = (x.iterator + i).load()
            for j in cutlass.range_constexpr(4):
                j = 3 - j
                b = (x.iterator + j).load()
            """,
            "b = (x.iterator + j).load()",
        ),
        (
            # Iteration ``j`` loads element ``j + 1``, which the first loop
            # loaded one iteration later (and the last one never).
            """
            for i in cutlass.range_constexpr(4):
                k = i + 1
                a = (x.iterator + k).load()
            for j in cutlass.range_constexpr(4):
                j = j + 1
                b = (x.iterator + j).load()
            """,
            "b = (x.iterator + j).load()",
        ),
        (
            # ``_frag`` updates ``res`` in place: by the second loop
            # ``res[0]`` holds the first loop's last result.
            """
            for i in cutlass.range_constexpr(2):
                _frag(vals, res)
                k = res[0]
                a = (x.iterator + k).load()
            for j in cutlass.range_constexpr(2):
                k2 = res[0]
                b = (x.iterator + k2).load()
            """,
            "b = (x.iterator + k2).load()",
        ),
        (
            # ``a[0] = 1`` changes ``b[0]`` too: ``b`` holds the list ``a``
            # names.
            """
            a = [cutlass.Int32(0)]
            b = a
            u = (x.iterator + b[0]).load()
            a[0] = cutlass.Int32(1)
            w = (x.iterator + b[0]).load()
            """,
            "w = (x.iterator + b[0]).load()",
        ),
        (
            # ``len(b)`` reads the list ``a`` names, one longer by then.
            """
            a = []
            b = a
            u = (x.iterator + len(b)).load()
            a.append(cutlass.Int32(1))
            w = (x.iterator + len(b)).load()
            """,
            "w = (x.iterator + len(b)).load()",
        ),
        (
            """
            a = [cutlass.Int32(0)]
            b = a
            u = (x.iterator + operator.getitem(b, 0)).load()
            a[0] = cutlass.Int32(1)
            w = (x.iterator + operator.getitem(b, 0)).load()
            """,
            "w = (x.iterator + operator.getitem(b, 0)).load()",
        ),
        (
            # ``b`` names the list ``a`` holds, from before the body.
            """
            b = a
            u = (x.iterator + len(a)).load()
            b.append(cutlass.Int32(1))
            w = (x.iterator + len(a)).load()
            """,
            "w = (x.iterator + len(a)).load()",
        ),
        (
            # A vector type the matcher does not parse may differ between
            # two loads of one address.
            """
            a = cute.arch.load(x.iterator + p, ir.VectorType.get([V1], cutlass.Uint16.mlir_type))
            b = cute.arch.load(x.iterator + p, ir.VectorType.get([V2], cutlass.Uint16.mlir_type))
            """,
            "b = cute.arch.load(x.iterator + p, ir.VectorType.get([V2], cutlass.Uint16.mlir_type))",
        ),
    ],
    ids=[
        "carried_in_its_loop",
        "carried_in_the_earlier_loop",
        "served_value_after_its_loop",
        "served_value_reassigned",
        "branch_between",
        "rolled_loop_between",
        "different_extents",
        "earlier_name_reassigned",
        "loops_running_twice",
        "loop_target_rebound",
        "loop_target_rebound_before_the_load",
        "argument_updated_by_a_call",
        "container_updated_through_another_name",
        "container_length_through_another_name",
        "container_getitem_through_another_name",
        "container_updated_through_its_copy",
        "unparsed_vector_types",
    ],
)
def test_load_reuse_keeps_a_load_it_cannot_prove_served(source: str, kept: str) -> None:
    assert kept in _reuse(source)


def test_load_reuse_follows_a_copy_of_a_name() -> None:
    # ``q`` holds ``p``'s value, and ``k`` the second loop's iteration.
    code = _reuse(
        """
        q = p
        a = (x.iterator + p).load()
        b = (x.iterator + q).load()
        for i in cutlass.range_constexpr(4):
            c = (y.iterator + i).load()
        for j in cutlass.range_constexpr(4):
            k = j
            d = (y.iterator + k).load()
        """
    )
    assert "b = a" in code
    assert "d = _jam_stash_0[j]" in code


def test_load_reuse_keeps_served_values_within_the_register_budget() -> None:
    # Sixteen 16-byte packets per loop fill the 64 registers: the loads of
    # one more element each stay loads.
    code = _reuse(
        """
        for i in cutlass.range_constexpr(16):
            a = cute.arch.load(x.iterator + i, ir.VectorType.get([8], cutlass.Uint16.mlir_type))
            c = (y.iterator + i).load()
        for j in cutlass.range_constexpr(16):
            b = cute.arch.load(x.iterator + j, ir.VectorType.get([8], cutlass.Uint16.mlir_type))
            d = (y.iterator + j).load()
        """
    )
    assert "b = _jam_stash_0[j]" in code
    assert "d = (y.iterator + j).load()" in code


@pytest.mark.parametrize(
    ("dtype", "served"),
    [(torch.float32, True), (torch.int64, False)],
    ids=["fp32", "int64"],
)
def test_load_reuse_counts_a_64_bit_scalar_as_two_registers(
    dtype: torch.dtype, served: bool
) -> None:
    # Forty scalars stashed for the second loop: forty registers, or eighty,
    # past the budget, at eight bytes each.
    code = _reuse(
        """
        for i in cutlass.range_constexpr(40):
            a = (x.iterator + i).load()
        for j in cutlass.range_constexpr(40):
            b = (x.iterator + j).load()
        """,
        {"x": torch.empty(64, dtype=dtype)},
    )
    assert ("b = _jam_stash_0[j]" in code) == served


def test_a_residue_bound_below_its_period_keeps_its_select() -> None:
    # ``r`` is ``index % 8``: ``r < 8`` always holds and its select folds,
    # ``r < 4`` fails for half the residues.
    def folded(bound: int) -> str:
        (stmt,) = ast.parse(f"v = load() if r < {bound} else 0.0").body
        _ElideResidueBounds(
            {"r": 8},
            lambda node: node.value if isinstance(node, ast.Constant) else None,
        ).visit(stmt)
        return ast.unparse(stmt)

    assert folded(8) == "v = load()"
    assert folded(4) == "v = load() if r < 4 else 0.0"


@pytest.mark.parametrize(
    ("preceding", "bound", "constexpr_values"),
    [
        ("", "cutlass.Int32(4294967296)", {}),
        ("", "cutlass.Int32(C)", {"C": 4294967296}),
        ("C = 4294967296\n", "cutlass.Int32(C)", {}),
    ],
    ids=["literal", "constexpr", "assigned"],
)
def test_an_int32_bound_past_the_types_range_keeps_its_select(
    preceding: str, bound: str, constexpr_values: dict[str, int]
) -> None:
    # The device wraps ``cutlass.Int32(4294967296)`` to 0, so ``r < 0``
    # never holds: the bound is no constant to fold the select with, as a
    # literal, a constexpr or a name bound before the loop.
    jam = _lane_jam(
        preceding + "for lane in range(8):\n    i = base + cutlass.Int32(lane)\n"
        "    r = _cute_python_mod(i, 8)\n"
        f"    v = 1 if r < {bound} else 0\n",
        {},
        constexpr_values,
    )
    code = ast.unparse(ast.Module(body=jam.run(), type_ignores=[]))
    assert f"v_lane7 = 1 if r_lane7 < {bound} else 0" in code


def test_an_int32_modulus_past_the_types_range_is_no_residue() -> None:
    # The device wraps ``cutlass.Int32(4294967304)`` to 8, so the residue's
    # period is not the literal: its bounds select stays.
    jam = _lane_jam(
        "for lane in range(8):\n    i = base + cutlass.Int32(lane)\n"
        "    r = _cute_python_mod(i, cutlass.Int32(4294967304))\n"
        "    v = 1 if r < 4294967304 else 0\n",
        {},
    )
    code = ast.unparse(ast.Module(body=jam.run(), type_ignores=[]))
    assert "v_lane7 = 1 if r_lane7 < 4294967304 else 0" in code


@pytest.mark.parametrize(
    ("statements", "folded"),
    [
        ("r = _cute_python_mod(i, 8)\nv = 1 if r < 8 else 0", True),
        ("v = 1 if r < 8 else 0\nr = _cute_python_mod(i, 8)", False),
    ],
    ids=["after_its_binding", "before_its_binding"],
)
def test_a_residue_select_folds_only_after_the_residue_is_bound(
    statements: str, folded: bool
) -> None:
    # Before its binding, ``r`` holds the previous lane's residue (lane 0's
    # comes from before the loop, which may fail ``r < 8``): folding the
    # select there would hide that carried read from the register check.
    jam = _lane_jam(
        "for lane in range(8):\n    i = base + cutlass.Int32(lane)\n"
        + textwrap.indent(statements, "    "),
        {},
    )
    if folded:
        code = ast.unparse(ast.Module(body=jam.run(), type_ignores=[]))
        assert "r < 8" not in code
        assert "v = 1" in code
    else:
        with pytest.raises(_Decline, match="carried from one lane"):
            jam.run()


@pytest.mark.parametrize(
    ("update", "private"),
    [
        ("_fuse_cache_0[k] = x", True),
        ("_fuse_cache_0[k] = _fuse_cache_0[k] + x", False),
        ("_fuse_cache_0[k] += x", False),
        ("_fuse_cache_0[_fuse_cache_0[k]] = x", False),
    ],
    ids=["written", "read_in_the_value", "augmented", "read_in_the_index"],
)
def test_a_register_cache_is_each_lanes_own_when_written_before_read(
    update: str, private: bool
) -> None:
    # Rolled, a lane that reads the cache before writing it reads the slots
    # the previous lane wrote; a copy of its own would hold nothing yet.
    jam = _lane_jam(
        f"""
        _fuse_cache_0 = cute.make_rmem_tensor(8, cutlass.Float32)
        for lane in range(8):
            {update}
            y = _fuse_cache_0[k]
        """,
        {},
    )
    jam._find_private_caches(jam.loop.body)
    assert bool(jam.private_caches) == private


@pytest.mark.parametrize(
    ("before", "after", "private", "dead"),
    [
        ("pass", "pass", True, True),
        ("z = _fuse_cache_0[0]", "pass", True, False),
        ("pass", "z = _fuse_cache_0[0]", False, False),
    ],
    ids=["own", "named_before_the_loop", "read_after_the_loop"],
)
def test_a_register_cache_named_outside_the_loop(
    before: str, after: str, private: bool, dead: bool
) -> None:
    # The lanes' copies replace a cache's allocation only when no other
    # statement before the loop names it; a cache the code after the loop
    # reads gets no copies.
    *preceding, loop = ast.parse(
        "_fuse_cache_0 = cute.make_rmem_tensor(8, cutlass.Float32)\n"
        f"{before}\n"
        "for lane in range(8):\n"
        "    _fuse_cache_0[k] = x\n"
        "    y = _fuse_cache_0[k]\n"
    ).body
    jam = _Jam(
        cast("ast.For", loop),
        preceding=preceding,
        following=ast.parse(after).body,
        rename_groups={},
        tensors={},
        proven_disjoint_tensor_pairs=(),
        constexpr_values={},
        env=cast("Any", None),
        new_var=str,
    )
    jam._find_private_caches(jam.loop.body)
    assert bool(jam.private_caches) == private
    assert bool(jam.dead_allocations) == dead


@pytest.mark.parametrize(
    ("body", "private"),
    [
        (
            (
                "for k in range(4):\n"
                "    _fuse_cache_0[k] = x\n"
                "for k in range(4):\n"
                "    y = _fuse_cache_0[k]"
            ),
            True,
        ),
        # The loop that writes slot ``k`` reads slot ``k + 1`` first.
        (
            (
                "for k in range(4):\n"
                "    _fuse_cache_0[k] = x\n"
                "    y = _fuse_cache_0[k + 1]"
            ),
            False,
        ),
        # A lane taking the other arm reads what it has not written.
        (
            "if c:\n    _fuse_cache_0[k] = x\nelse:\n    y = _fuse_cache_0[k]",
            False,
        ),
        ("if c:\n    _fuse_cache_0[k] = x\ny = _fuse_cache_0[k]", False),
        (
            (
                "if c:\n"
                "    _fuse_cache_0[k] = x\n"
                "else:\n"
                "    _fuse_cache_0[k] = z\n"
                "y = _fuse_cache_0[k]"
            ),
            True,
        ),
        # A lane that skips the branch does not access the cache.
        ("if c:\n    _fuse_cache_0[k] = x\n    y = _fuse_cache_0[k]", True),
        # A trip that skips the branch leaves its slot to the next lane.
        (
            (
                "for k in range(4):\n"
                "    if c:\n"
                "        _fuse_cache_0[k] = x\n"
                "for k in range(4):\n"
                "    y = _fuse_cache_0[k]"
            ),
            False,
        ),
        # So does an arm whose inner branch skips the write.
        (
            (
                "if c:\n"
                "    if d:\n"
                "        _fuse_cache_0[k] = x\n"
                "else:\n"
                "    _fuse_cache_0[k] = z\n"
                "y = _fuse_cache_0[k]"
            ),
            False,
        ),
    ],
    ids=[
        "sweeps",
        "read_ahead_in_the_writing_loop",
        "read_in_the_other_arm",
        "written_in_one_arm",
        "written_in_both_arms",
        "every_access_in_one_arm",
        "written_in_one_arm_of_the_writing_loop",
        "written_in_an_inner_arm",
    ],
)
def test_a_register_cache_is_written_before_read_on_every_path(
    body: str, private: bool
) -> None:
    jam = _lane_jam(
        "_fuse_cache_0 = cute.make_rmem_tensor(8, cutlass.Float32)\n"
        "for lane in range(8):\n" + textwrap.indent(body, "    "),
        {},
    )
    jam._find_private_caches(jam.loop.body)
    assert bool(jam.private_caches) == private


def test_the_lanes_copies_of_a_register_cache_fit_its_budget() -> None:
    # Eight lanes' copies of a 16-element cache would hold 128 elements, twice
    # the fragment ``fuse_two_pass_loads`` admits for one thread.
    jam = _lane_jam(
        """
        _fuse_cache_0 = cute.make_rmem_tensor(16, cutlass.Float32)
        for lane in range(8):
            _fuse_cache_0[k] = x
            y = _fuse_cache_0[k]
        """,
        {},
    )
    with pytest.raises(_Decline, match="register caches would not fit"):
        jam._find_private_caches(jam.loop.body)
