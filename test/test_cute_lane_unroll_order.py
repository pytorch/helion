"""The loads-first lane unroll (``cute_lane_unroll``,
``helion/_compiler/cute/unroll_lane_loads.py``) emits the same code in every
process: the pure statements a nested lane loop's loads pull out of the outer
lane body enter the load phase in the order the loads read them, not in the
iteration order of a set of their names, which string hashing changes from one
process to the next."""

from __future__ import annotations

import ast

import pytest

from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.unroll_lane_loads import unroll_lane_loads

pytestmark = pytest.mark.cpu_only


def test_outer_dependencies_hoist_in_the_order_the_loads_read_them() -> None:
    # Eight offsets of the outer lane body, read by the nested lane loop's
    # load in reverse order: a set of their names would match that order in
    # one of 8! cases.
    names = [f"offset_{i}" for i in range(8)]
    source = "\n".join(
        [
            "for lane_1 in range(2):",
            *(f"    {name} = lane_1 + {i}" for i, name in enumerate(names)),
            "    for lane_0 in range(2):",
            f"        value = (ptr + lane_0 + {' + '.join(reversed(names))}).load()",
            "        (out + lane_0).store(value)",
        ]
    )
    (outer,) = ast.parse(source).body
    assert isinstance(outer, ast.For)
    setattr(outer, HELION_LANE_LOOP_VAR_ATTR, "lane_1")
    setattr(outer.body[-1], HELION_LANE_LOOP_VAR_ATTR, "lane_0")
    body, fired = unroll_lane_loads([outer], 2, lambda name: name)
    assert fired
    load_phase = next(
        stmt
        for stmt in body
        if isinstance(stmt, ast.For)
        and "_lane_unroll_loads_0.append" in ast.unparse(stmt)
    )
    hoisted = [
        stmt.targets[0].id  # type: ignore[union-attr]
        for stmt in load_phase.body
        if isinstance(stmt, ast.Assign)
    ]
    assert hoisted == names[::-1]
