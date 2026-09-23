from __future__ import annotations

import ast
from copy import deepcopy

import pytest

from test.test_cute_interchanged_store_dce import _PAIRS
from test.test_cute_interchanged_store_dce import _interchanged
from test.test_cute_interchanged_store_dce import _program

from helion._compiler import tile_strategy
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.interchanged_store_dce import eliminate_interchanged_stores
from helion._compiler.cute.persistent_branch_vec import _definition_snapshots
from helion._compiler.cute.persistent_branch_vec import _freeze_definition


def _doubling_assignments(depth):
    source = ["budget_0 = base"]
    source.extend(
        f"budget_{i} = budget_{i - 1} + budget_{i - 1}" for i in range(1, depth + 1)
    )
    return ast.parse("\n".join(source)).body


def _balanced_sum(name, depth):
    result = ast.Name(id=name, ctx=ast.Load())
    for _ in range(depth):
        result = ast.BinOp(left=result, op=ast.Add(), right=deepcopy(result))
    return ast.fix_missing_locations(result)


@pytest.mark.parametrize("site", ["prefix", "body", "bound", "address"])
@pytest.mark.parametrize("depth", [3, 12])
def test_interchanged_store_budget_preserves_unproved_passes(site, depth):
    first, serial, final, candidates = _interchanged(_program())
    final_serial = final[-1]
    final_lane = final_serial.body[-1]
    if site == "prefix":
        additions = _doubling_assignments(depth)
        first.body[:0] = additions
        final[:0] = deepcopy(additions)
        assert (_definition_snapshots(additions) is None) == (depth == 12)
    elif site == "body":
        additions = _doubling_assignments(depth)
        serial.body[:0] = additions
        final_lane.body[:0] = deepcopy(additions)
        candidates = {index + len(additions) for index in candidates}
        assert (_definition_snapshots(additions) is None) == (depth == 12)
    elif site == "bound":
        expression = _balanced_sum("rows", depth)
        serial.iter.args[0] = expression
        final_serial.iter.args[0] = deepcopy(expression)
        assert (_freeze_definition(expression, {}) is None) == (depth == 12)
    else:
        expression = _balanced_sum("index", depth)
        for loop in (serial, final_lane):
            for node in ast.walk(loop):
                if (
                    isinstance(node, ast.BinOp)
                    and ast.unparse(node.left) == "output.iterator"
                ):
                    node.right = deepcopy(expression)
        assert (_freeze_definition(expression, {}) is None) == (depth == 12)
    before = ast.dump(ast.Module(body=[first, *final], type_ignores=[]))
    removed = eliminate_interchanged_stores(
        first, serial, final, candidates, _PAIRS, set()
    )
    assert removed == (1 if depth == 3 else 0)
    if depth == 12:
        assert ast.dump(ast.Module(body=[first, *final], type_ignores=[])) == before


@pytest.mark.parametrize("depth", [3, 12])
def test_two_pass_snapshot_budget_keeps_original_global_loads(depth, monkeypatch):
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    before = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    after = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
    updated = after + delta
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', x.iterator + lane_index, updated, None)
"""
    ).body
    additions = _doubling_assignments(depth)
    body[1].body[:0] = additions
    assert (_definition_snapshots(additions) is None) == (depth == 12)
    before = ast.dump(ast.Module(body=body, type_ignores=[]))
    result = fuse_two_pass_loads(
        body, tensor_dtypes={"x": "cutlass.BFloat16"}, reload_modes={7: "register"}
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert ("_fuse_cache_" in code) == (depth == 3)
    assert code.count("_helion_persistent_branch_vec_load") == (1 if depth == 3 else 2)
    if depth == 12:
        assert ast.dump(ast.Module(body=result, type_ignores=[])) == before


@pytest.mark.parametrize("depth", [3, 12])
@pytest.mark.parametrize("address_dependent", [False, True])
def test_lane_split_budget_retains_aliasing_memory_order(depth, address_dependent):
    offset = f"lane + budget_{depth}" if address_dependent else "lane"
    body = (
        _doubling_assignments(depth)
        + ast.parse(
            f"""
value = (x.iterator + {offset}).load()
reduced = _helion_lane_reduce(value, 'sum', 0.0, 1, 1, 0, '', 1)
(x.iterator + {offset}).store(reduced + budget_{depth})
"""
        ).body
    )
    index = len(body) - 2
    marker = tile_strategy._is_lane_reduce_marker_assign(body[index])
    assert marker is not None
    assert tile_strategy._lane_split_reorders_aliasing_memory(
        body, [(index, marker)], set(), "lane", 8, {}
    ) == (address_dependent and depth == 12)
