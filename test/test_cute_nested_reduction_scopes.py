from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import sympy

from helion import exc
from helion._compiler import reduction_strategy as reductions
from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import ast_rename
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.nested_lane_reductions import (
    normalize_nested_lane_reductions,
)
from helion._compiler.cute.nested_lane_reductions import resolve_pruned_lane_owners
from helion._compiler.generate_ast import GenerateAST


def _body(source: str) -> list[ast.AST]:
    return list(ast.parse(source).body)


def _marker(value: str, owner: str) -> str:
    return lanes._lane_reduce_marker_expr(
        value, "sum", "cutlass.Float64(0)", 1, owner_lane=owner
    )


def _nested_body(outer_extent: int, inner_extent: int) -> list[ast.AST]:
    inner = lanes._create_lane_loop(
        "inner",
        inner_extent,
        _body(
            "row = tile_i + outer\n"
            "column = tile_j + inner\n"
            "valid_row = row < end0\n"
            "valid_column = column < end1\n"
            "snapshot = carried\n"
            "snapshot2 = snapshot\n"
            "value = (x.iterator + row * width + column).load() if valid_row and valid_column else cutlass.Float64(0)\n"
            f"inner_sum = {_marker('value', 'inner')}\n"
            "masked_inner = inner_sum if valid_row else cutlass.Float64(0)\n"
            f"outer_sum = {_marker('masked_inner', 'outer')}\n"
            "next_carried = snapshot2 + outer_sum\n"
        ),
    )
    outer = lanes._create_lane_loop("outer", outer_extent, [inner])
    inner_tiles = ast.parse(
        f"for tile_j in range(0, end1, {inner_extent}):\n    pass"
    ).body[0]
    assert isinstance(inner_tiles, ast.For)
    inner_tiles.body = [outer]
    outer_tiles = ast.parse(
        f"for tile_i in range(0, end0, {outer_extent}):\n    pass"
    ).body[0]
    assert isinstance(outer_tiles, ast.For)
    outer_tiles.body = [inner_tiles]
    return [
        *_body("carried = cutlass.Float64(initial)"),
        outer_tiles,
        *_body("out.store(carried)"),
    ]


def _normalize(body: list[ast.AST]) -> list[ast.AST]:
    return normalize_nested_lane_reductions(
        body,
        uniform_names={"x", "width", "end0", "end1", "initial", "out"},
        proven_disjoint_tensor_pairs={frozenset({"x", "out"})},
        proven_tensor_stride_values={},
        rename_groups={"next_carried": "carried", "carried": "carried"},
    )


class _Memory:
    def __init__(self, values: np.ndarray, offset: int = 0) -> None:
        self.values = values
        self.offset = offset

    @property
    def iterator(self) -> _Memory:
        return self

    def __add__(self, offset: int) -> _Memory:
        return _Memory(self.values, self.offset + offset)

    def load(self) -> float:
        return float(self.values[self.offset])

    def store(self, value: float) -> None:
        self.values[self.offset] = value


@pytest.mark.parametrize("outer_extent,inner_extent", [(2, 3), (4, 8), (8, 4)])
@pytest.mark.parametrize(
    "end0,end1", [(0, 5), (5, 0), (1, 1), (3, 5), (8, 8), (11, 13)]
)
@pytest.mark.parametrize("seed", [7, 19])
def test_complete_nested_sums_with_tails_and_carried_output(
    outer_extent: int, inner_extent: int, end0: int, end1: int, seed: int
) -> None:
    values = np.random.default_rng(seed).integers(-31, 32, (16, 16)).astype(np.float64)
    output = np.full((1,), np.nan)
    body = _normalize(_nested_body(outer_extent, inner_extent))
    lanes.validate_lane_reduce_owners(body)
    body = lanes.split_lane_loop_reductions(
        body,
        proven_disjoint_tensor_pairs={frozenset({"x", "out"})},
        rename_groups={"next_carried": "carried", "carried": "carried"},
    )
    body = lanes.restore_unprocessed_lane_reduce_markers(body)
    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    ast_rename(module, {"next_carried": "carried"})
    assert "_helion_lane_reduce" not in ast.unparse(module)
    exec(
        compile(ast.fix_missing_locations(module), "<nested-reduction>", "exec"),
        {
            "cutlass": SimpleNamespace(Float64=float, Int32=int, range=range),
            "x": _Memory(values.reshape(-1)),
            "out": _Memory(output),
            "width": 16,
            "end0": end0,
            "end1": end1,
            "initial": 37,
        },
    )
    assert output[0] == 37 + values[:end0, :end1].sum()


@pytest.mark.parametrize(
    "change",
    [
        "unknown_owner",
        "varying_suffix",
        "foreign_first",
        "missing_inner",
        "suffix_load",
        "suffix_call",
        "suffix_store",
        "conditional_suffix",
        "repeated_binding",
        "lane_write",
        "loop_else",
        "dynamic_extent",
        "raw_carry",
        "unknown_prefix_call",
        "constant_outer_input",
        "shadow_global",
        "shadow_range",
        "shadow_math",
        "shadow_operator",
        "shadow_marker",
        "shadow_ancestor",
        "own_wrapper_lane",
        "own_wrapper_helper",
    ],
)
def test_unproved_nested_schedules_remain_rejected(change: str) -> None:
    program = _nested_body(4, 8)
    loops = [
        node
        for top in program
        for node in ast.walk(top)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "inner"
    ]
    assert len(loops) == 1
    inner = loops[0]
    if change == "unknown_owner":
        inner.body[-2] = _body(f"outer_sum = {_marker('masked_inner', 'absent')}")[0]
    elif change == "varying_suffix":
        inner.body[-3] = _body("masked_inner = inner_sum + inner")[0]
    elif change == "foreign_first":
        inner.body[-4], inner.body[-2] = inner.body[-2], inner.body[-4]
    elif change == "missing_inner":
        inner.body[-4] = _body("inner_sum = value")[0]
    elif change == "suffix_load":
        inner.body[-1] = _body("next_carried = x.load() + outer_sum")[0]
    elif change == "suffix_call":
        inner.body[-1] = _body("next_carried = unknown(outer_sum)")[0]
    elif change == "suffix_store":
        inner.body.append(_body("out.store(outer_sum)")[0])
    elif change == "conditional_suffix":
        inner.body[-1] = _body("if flag:\n    next_carried = outer_sum")[0]
    elif change == "repeated_binding":
        inner.body.append(_body("masked_inner = cutlass.Float64(0)")[0])
    elif change == "lane_write":
        inner.body.append(_body("inner = 0")[0])
    elif change == "loop_else":
        inner.orelse = cast("list[ast.stmt]", _body("out.store(0)"))
    elif change == "dynamic_extent":
        inner.iter = ast.parse("range(runtime_extent)", mode="eval").body
    elif change == "raw_carry":
        inner.body.insert(7, _body("extra = extra + value")[0])
    elif change == "unknown_prefix_call":
        inner.body.insert(7, _body("extra = unknown(value)")[0])
    elif change == "constant_outer_input":
        inner.body[-3] = _body("masked_inner = cutlass.Float64(1)")[0]
    elif change.startswith("shadow_"):
        name = {
            "shadow_global": "cutlass",
            "shadow_range": "range",
            "shadow_math": "math",
            "shadow_operator": "operator",
            "shadow_marker": "_helion_lane_reduce",
            "shadow_ancestor": "cutlass",
        }[change]
        if change == "shadow_ancestor":
            program.insert(0, _body(f"{name} = replacement")[0])
        else:
            inner.body.insert(0, _body(f"{name} = replacement")[0])
    elif change == "own_wrapper_lane":
        inner.body[-4] = _body(f"inner_sum = {_marker('value', 'inner')} + inner")[0]
    elif change == "own_wrapper_helper":
        inner.body[-4] = _body(f"inner_sum = helper({_marker('value', 'inner')})")[0]
    with pytest.raises(exc.BackendUnsupported):
        normalized = _normalize(program)
        lanes.validate_lane_reduce_owners(normalized)


def _loop(
    block: int, body: list[ast.AST], *, lane: str | None
) -> lanes.DeviceLoopState:
    strategy = object.__new__(lanes.PerThreadNDTileStrategy)
    strategy.block_ids = [block]
    strategy._lane_var_by_block = {block: lane} if lane is not None else {}
    node = ast.parse("for offset in range(8):\n    pass").body[0]
    assert isinstance(node, ast.For)
    return lanes.DeviceLoopState(
        strategy=strategy,
        block_id_to_info={},
        for_node=node,
        inner_statements=body,
        lane_loop_blocks={block} if lane is not None else set(),
    )


@pytest.mark.parametrize(
    "mode",
    ["outside", "inside", "missing_owner", "missing_serial", "same_scope", "ambiguous"],
)
def test_serial_scope_must_be_proved_outside_reduction_owner(mode: str) -> None:
    owner_body: list[ast.AST] = []
    serial_body: list[ast.AST] = []
    owner = _loop(7, owner_body, lane="physical")
    serial = _loop(9, serial_body, lane=None)
    codegen = object.__new__(GenerateAST)
    codegen.active_device_loops = {7: [owner], 9: [serial]}
    codegen.statements_stack = [serial_body, owner_body]
    if mode == "inside":
        codegen.statements_stack = [owner_body, serial_body]
    elif mode == "missing_owner":
        codegen.statements_stack = [serial_body]
    elif mode == "missing_serial":
        codegen.statements_stack = [owner_body]
    elif mode == "same_scope":
        serial.inner_statements = owner_body
        codegen.statements_stack = [owner_body]
    elif mode == "ambiguous":
        second_body: list[ast.AST] = []
        codegen.active_device_loops[7].append(_loop(7, second_body, lane="second"))
        codegen.statements_stack.append(second_body)
    strategy = object.__new__(reductions.ReductionStrategy)
    strategy.block_ids = [7]
    result = strategy._lane_reduce_marker_unsupported(SimpleNamespace(codegen=codegen))
    assert result is (mode != "outside")


@pytest.mark.parametrize(
    "mode",
    [
        "concrete",
        "missing",
        "ambiguous",
        "two_lanes",
        "powered",
        "coefficient",
        "multi_threads",
        "cluster",
        "conflict",
    ],
)
def test_reshape_records_only_complete_physical_group_proofs(mode: str) -> None:
    first, second, third = sympy.symbols("b1 b2 b3", integer=True, positive=True)
    numel = first * second
    block = 3
    if mode == "powered":
        numel = first**2 * second
    elif mode == "coefficient":
        numel = 2 * first * second
    elif mode == "multi_threads":
        numel = first * second * third
        block = 4
    sizes = [SimpleNamespace(block_id=i, numel=1) for i in range(block + 1)]
    sizes[block].numel = numel
    axes = {0: (1, 4), 1: (2, 4), 2: (None, None), block: (block, 32)}
    if mode == "two_lanes":
        axes[1] = (None, None)
    elif mode == "multi_threads":
        axes[3] = (3, 2)
    env = SimpleNamespace(
        block_sizes=sizes,
        get_block_id={first: 1, second: 2, third: 3}.get,
        canonical_block_id=lambda value: value,
        backend=SimpleNamespace(
            name="cute", thread_linear_index_expr=lambda sizes: "lane_index"
        ),
    )
    first_loop = _loop(1, [], lane="first" if mode == "two_lanes" else None)
    if mode != "two_lanes":
        first_loop.block_thread_axes = {1: 2}
    second_loop = _loop(2, [], lane="physical")
    active = {1: [first_loop], 2: [second_loop]}
    if mode == "missing":
        active.pop(1)
    elif mode == "ambiguous":
        active[2].append(_loop(2, [], lane="other"))
    elif mode == "multi_threads":
        third_loop = _loop(3, [], lane=None)
        third_loop.block_thread_axes = {3: 3}
        active[3] = [third_loop]
    fallbacks = (
        {"synthetic": ("other", 4, 16, "lane_index")} if mode == "conflict" else {}
    )
    strategy = object.__new__(reductions.PersistentReductionStrategy)
    strategy.block_ids = [block]
    strategy._synthetic_cute_lane_var = "synthetic"
    fn = SimpleNamespace(
        cute_state=SimpleNamespace(
            reshape_lane_fallbacks=fallbacks,
            simt_cluster_n=2 if mode == "cluster" else 1,
        ),
        tile_strategy=SimpleNamespace(
            thread_axis_for_block_id=lambda value: axes[value][0],
            thread_extent_for_block_id=lambda value: axes[value][1],
        ),
    )
    strategy._fn = lambda: fn
    state = SimpleNamespace(
        codegen=SimpleNamespace(active_device_loops=active, current_grid_state=None)
    )
    with patch.object(CompileEnvironment, "current", return_value=env):
        group = strategy._reshape_merged_reduction_group_params()
        if mode == "conflict":
            with pytest.raises(exc.BackendUnsupported, match="conflicting lane"):
                strategy._lane_reduce_owner(state, reshape_group=group)
        else:
            assert (
                strategy._lane_reduce_owner(state, reshape_group=group) == "synthetic"
            )
            assert fallbacks == (
                {"synthetic": ("physical", 4, 16, "lane_index")}
                if mode == "concrete"
                else {}
            )


@pytest.mark.parametrize(
    "mode",
    [
        "pruned",
        "live_loop",
        "live_name",
        "wrong_owner",
        "wrong_group",
        "wrong_lane_expr",
        "cluster",
        "group_count",
    ],
)
def test_only_proven_pruned_owner_is_rebound(mode: str) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "value",
        "sum",
        "cutlass.Float32(0)",
        32,
        group_pre=4,
        group_span=16 if mode != "wrong_group" else 32,
        group_lane_expr="lane_index" if mode != "wrong_lane_expr" else "another_index",
        group_count=2 if mode == "group_count" else 1,
        group_cluster_n=2 if mode == "cluster" else 1,
        owner_lane="synthetic",
    )
    statement = _body(f"reduced = {marker}")[0]
    physical = lanes._create_lane_loop("physical", 8, [statement])
    if mode == "live_loop":
        physical.body = [lanes._create_lane_loop("synthetic", 4, [statement])]
    elif mode == "wrong_owner":
        physical.body = [lanes._create_lane_loop("different", 8, [statement])]
    body: list[ast.AST] = [physical]
    if mode == "live_name":
        body.insert(0, _body("coordinate = synthetic + 1")[0])
    resolve_pruned_lane_owners(body, {"synthetic": ("physical", 4, 16, "lane_index")})
    parsed = lanes._is_lane_reduce_marker_assign(statement)
    assert parsed is not None
    assert parsed.owner_lane == ("physical" if mode == "pruned" else "synthetic")
    if mode in ("pruned", "live_loop"):
        lanes.validate_lane_reduce_owners(body)
    else:
        with pytest.raises(exc.BackendUnsupported):
            lanes.validate_lane_reduce_owners(body)
