from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import sympy
import torch

from .test_cute_chained_coordinate_bounds import _context
from .test_cute_chained_warp_loop_carry import fixture
from .test_cute_chained_warp_loop_carry import select
from helion._compiler.cute import chained_coordinate_bounds as bounds
from helion._compiler.cute import chained_matmul as chain


def _loop_context(begin=0, end=3, *, step=1):
    cg, _ = _context()
    plan, _, sink = fixture()
    assert plan.loop is not None
    plan = replace(
        plan,
        loop=replace(plan.loop, begin=begin, end=end),
        axes=((0, 3, 1), (1, 65, 32)),
    )
    selected = select(plan)
    assert selected is not None
    plan = replace(plan, warp_loop_carry=selected)
    cg.device_function.cute_state.chained_matmul_plan = plan
    cg.device_function.resolved_block_size = lambda block_id: step
    return cg, plan, sink


def _source(begin, end):
    return f"""
thread = cutlass.Int32(cute.arch.thread_idx()[0])
block = cutlass.Int32(cute.arch.block_idx()[0])
total = 0.0
for index in range({begin}, {end}):
    if thread < 128 and block < 9 and index < {end}:
        total = (total * 0.5) + (index * 0.25)
    if pointer.toint() % 16 == 0 and index * 32 + thread % 32 < length:
        store(block, thread, index, total)
finish(total)
"""


def _execute(code, block, thread, address):
    effects = []
    namespace = {
        "cutlass": SimpleNamespace(Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (thread, 0, 0),
                block_idx=lambda: (block, 0, 0),
            )
        ),
        "pointer": SimpleNamespace(toint=lambda: address),
        "length": 49,
        "store": lambda *values: effects.append(values),
        "finish": lambda value: effects.append((value,)),
    }
    exec(code, namespace)
    return effects


@pytest.mark.parametrize("begin,end", [(0, 0), (0, 3), (2, 7)])
def test_checked_warp_carry_keeps_loops_math_and_runtime_guards(begin, end):
    cg, plan, _ = _loop_context(begin, end)
    assert bounds._launch(cg, plan) == ((128, 1, 1), (9, 1, 1))
    original = ast.parse(_source(begin, end)).body
    before = ast.dump(ast.Module(body=original, type_ignores=[]))
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    if begin != end:
        assert receipt is not None
        rendered = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))
        assert "thread < 128" not in rendered and "block < 9" not in rendered
    assert ast.dump(ast.Module(body=original, type_ignores=[])) == before
    old_loop = next(node for node in original if isinstance(node, ast.For))
    new_loop = next(node for node in rewritten if isinstance(node, ast.For))
    assert ast.dump(old_loop.iter) == ast.dump(new_loop.iter)
    assert ast.dump(old_loop.body[-1]) == ast.dump(new_loop.body[-1])
    old_math = [
        ast.dump(node) for node in ast.walk(old_loop) if isinstance(node, ast.Assign)
    ]
    new_math = [
        ast.dump(node) for node in ast.walk(new_loop) if isinstance(node, ast.Assign)
    ]
    assert old_math == new_math
    old_code = compile(ast.Module(body=original, type_ignores=[]), "old", "exec")
    new_code = compile(ast.Module(body=rewritten, type_ignores=[]), "new", "exec")
    for block in range(9):
        for thread in range(128):
            for address in (256, 258):
                assert _execute(old_code, block, thread, address) == _execute(
                    new_code, block, thread, address
                )


@pytest.mark.parametrize(
    "begin,end",
    [
        (False, 3),
        (0, True),
        (-1, 3),
        (2, 1),
        (0, 1 << 31),
        (sympy.Symbol("begin", integer=True), 3),
        (0, sympy.Symbol("end", integer=True)),
    ],
)
def test_dynamic_negative_or_overflow_loop_bounds_decline(begin, end):
    cg, plan, _ = _loop_context(begin, end)
    original = ast.parse(_source(0, 3)).body
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert rewritten is original and receipt is None


@pytest.mark.parametrize("step", [0, 2, True, sympy.Symbol("step", integer=True)])
def test_only_exact_scalar_step_is_admitted(step):
    cg, plan, _ = _loop_context(step=step)
    assert bounds._launch(cg, plan) is None


@pytest.mark.parametrize("change", ["unowned", "dynamic_grid", "other_launch"])
def test_loop_still_requires_original_owner_and_exact_launch(change):
    cg, plan, _ = _loop_context()
    if change == "unowned":
        plan = replace(plan, warp_loop_carry=None)
        cg.device_function.cute_state.chained_matmul_plan = plan
    elif change == "dynamic_grid":
        plan = replace(plan, axes=((0, cast("int", sympy.Symbol("extent")), 1),))
        cg.device_function.cute_state.chained_matmul_plan = plan
    else:
        cg.device_function.cute_state.chained_matmul_plan = replace(plan)
    assert bounds._launch(cg, plan) is None


@pytest.mark.parametrize("change", ["graph", "storage"])
def test_stale_carry_proof_cannot_authorize_coordinate_rewrite(change):
    cg, plan, sink = _loop_context()
    assert plan.warp_loop_carry is not None
    if change == "graph":
        seed = plan.warp_loop_carry.seed
        seed.args = (seed.args[0], 0.5)
    else:
        sink.meta["val"] = torch.empty_like(sink.meta["val"])
    with pytest.raises(chain._UnsupportedChain, match="ownership changed"):
        bounds._launch(cg, plan)
