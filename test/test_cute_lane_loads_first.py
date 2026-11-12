"""Which statements the loads-first lane unroll moves into its load phase.

``unroll_lane_loads`` issues every unrolled lane's loads before the first lane
computes, together with the pure index arithmetic those loads read, which the
compute phase derives again.  A name a lane reads before it assigns it
carries the previous lane's value: its assignment stays in place, and so do
the loads that read it, since a copy in the load phase would leave the last
lane's value where the compute phase reads the carried one.
"""

from __future__ import annotations

import ast
import textwrap
import types

import pytest

from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.ast_read_writes import HELION_VEC_LANE_OF_ATTR
from helion._compiler.cute.unroll_lane_loads import unroll_lane_loads

pytestmark = pytest.mark.cpu_only


def _unroll(source: str, *, vloops: bool = False) -> tuple[str, bool]:
    """Unroll the top-level lane loops of ``source`` 4 ways; with ``vloops``,
    the ``for`` loops directly in their bodies are tile V-loops of the lane
    loop (``HELION_VEC_LANE_OF_ATTR``, as a ``VecLaneWrapper`` tags them)."""
    body = ast.parse(textwrap.dedent(source)).body
    for stmt in body:
        if isinstance(stmt, ast.For):
            assert isinstance(stmt.target, ast.Name)
            setattr(stmt, HELION_LANE_LOOP_VAR_ATTR, stmt.target.id)
            for inner in stmt.body if vloops else ():
                if isinstance(inner, ast.For):
                    setattr(inner, HELION_VEC_LANE_OF_ATTR, stmt.target.id)
    new_body, fired = unroll_lane_loads(body, 4, lambda name: name)
    return "\n".join(ast.unparse(stmt) for stmt in new_body), fired


class _Pointer:
    """``tensor.iterator + offset`` over a Python list (``_stores``)."""

    def __init__(self, data: list[int], offset: int = 0) -> None:
        self.data = data
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.data, self.offset + offset)

    def load(self) -> int:
        return self.data[self.offset]

    def store(self, value: int) -> None:
        self.data[self.offset] = value


def _stores(code: str, **values: int) -> list[int]:
    """What ``code`` stores into ``out`` run as Python, with ``x`` holding
    1, 2, 3, ... and ``values`` bound."""
    out = [0] * 64
    exec(
        code,
        {
            "cutlass": types.SimpleNamespace(range_constexpr=range, Int32=int),
            "x": types.SimpleNamespace(iterator=_Pointer(list(range(1, 65)))),
            "out": types.SimpleNamespace(iterator=_Pointer(out)),
            **values,
        },
    )
    return out


def test_a_load_of_a_loop_carried_value_stays_in_place() -> None:
    # Hoisted with the load that reads it, ``acc = acc + 1`` also stayed in
    # the compute phase: eight increments for four lanes.
    _code, fired = _unroll(
        """
        acc = cutlass.Int32(0)
        for lane in range(4):
            acc = acc + 1
            v = (x.iterator + acc).load()
            (out.iterator + lane).store(v)
        """
    )
    assert not fired


def test_a_v_loop_load_of_a_loop_carried_value_stays_in_place() -> None:
    # The same through the trips of a tile V-loop, a level of its own: copied
    # into the load phase, ``offset = offset + 1`` also ran there, and the
    # compute phase loaded and stored past the 16 elements.
    source = textwrap.dedent(
        """
        offset = base
        for lane in range(4):
            for vec_lane_0 in cutlass.range_constexpr(4):
                offset = offset + 1
                v = (x.iterator + offset).load()
                (out.iterator + offset).store(v)
        """
    )
    code, _fired = _unroll(source, vloops=True)
    assert _stores(code, base=0) == _stores(source, base=0)


def test_a_load_of_lane_arithmetic_moves_with_it() -> None:
    # Over 4 lanes, and over 8 in rolled trips of 4: the index is derived in
    # both phases, the load moves to the first, and the stores stay the
    # original's.
    for lanes in (4, 8):
        source = textwrap.dedent(
            f"""
            for lane in range({lanes}):
                index = base + lane
                v = (x.iterator + index).load()
                (out.iterator + lane).store(v)
            """
        )
        code, fired = _unroll(source)
        assert fired
        assert code.count("index = base + lane") == 2
        assert code.count(".load()") == 1
        assert _stores(code, base=3) == _stores(source, base=3)


_LOAD_READ_BEFORE_IT = """
    for lane in range(LANES):
        acc = acc + v
        v = (x.iterator + lane).load()
        (out.iterator + lane).store(acc)
"""
_INDEX_READ_BEFORE_IT = """
    idx = base
    for lane in range(LANES):
        w = idx * 2
        idx = base + lane + 1
        v = (x.iterator + idx).load()
        (out.iterator + w).store(v)
"""


@pytest.mark.parametrize("lanes", [4, 8])
@pytest.mark.parametrize(
    ("source", "values"),
    [
        (_LOAD_READ_BEFORE_IT, {"acc": 0, "v": 100}),
        (_INDEX_READ_BEFORE_IT, {"base": 0}),
    ],
    ids=["load", "index"],
)
def test_an_assignment_read_before_it_keeps_the_carried_value(
    source: str, values: dict[str, int], lanes: int
) -> None:
    # Lane 0 reads the ``v`` or ``idx`` from before the loop, each later lane
    # the previous lane's.  Moved into the load phase, the assignment left the
    # last lane's value for lane 0 (stores 4, 5, 7, 10 for 100, 101, 103, 106)
    # and, over 8 lanes in two rolled trips of 4, for lane 4.
    source = textwrap.dedent(source).replace("LANES", str(lanes))
    code, _fired = _unroll(source)
    assert _stores(code, **values) == _stores(source, **values)
