from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import sympy
import torch

import helion
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
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl


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


def test_marker_free_normalization_preserves_ast_and_skips_dependency_analysis() -> (
    None
):
    body = _body(
        "initial = 7\n"
        "for tile in range(8):\n"
        "    if tile % 2:\n"
        "        value = x[tile] + initial\n"
        "    else:\n"
        "        value = x[tile] - initial\n"
        "    out[tile] = value\n"
    )
    tree = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    before = ast.dump(tree, include_attributes=True)
    source = ast.unparse(tree)
    node_ids = [id(node) for node in ast.walk(tree)]
    with patch.object(
        lanes,
        "_update_scalar_definitions",
        side_effect=AssertionError("marker-free dependency analysis"),
    ):
        assert _normalize(body) is body
    assert ast.dump(tree, include_attributes=True) == before
    assert ast.unparse(tree) == source
    assert [id(node) for node in ast.walk(tree)] == node_ids


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
    fn.codegen = state.codegen
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


@pytest.mark.parametrize(
    "mode",
    [
        "complete",
        "uniform_loop",
        "uniform_branch",
        "singleton",
        "live_name",
        "live_loop",
        "serial_lane",
        "varying_branch",
        "varying_loop",
        "load_bound",
        "early_return",
        "wrong_group",
        "cluster",
        "multi_group",
        "matmul",
    ],
)
def test_physical_reshape_reduction_requires_complete_execution(mode: str) -> None:
    span = 1 if mode == "singleton" else 8
    marker = lanes._lane_reduce_marker_expr(
        "value",
        "sum",
        "cutlass.Float32(0)",
        32,
        group_pre=1,
        group_span=span,
        group_lane_expr="lane_index",
        group_cluster_n=2 if mode == "cluster" else 1,
        group_count=2 if mode == "multi_group" else 1,
        owner_lane="synthetic",
        matmul_contribution=mode == "matmul",
    )
    statement = _body(f"reduced = cutlass.Float32({marker})")[0]
    body: list[ast.AST] = [statement]
    if mode in ("live_loop", "serial_lane"):
        body = [
            lanes._create_lane_loop(
                "synthetic" if mode == "live_loop" else "other", 4, body
            )
        ]
    elif mode in ("uniform_loop", "varying_loop", "load_bound"):
        bound = {
            "uniform_loop": "end",
            "varying_loop": "cute.arch.thread_idx()[1]",
            "load_bound": "x.iterator.load()",
        }[mode]
        loop = _body(f"for offset in range({bound}):\n    pass")[0]
        assert isinstance(loop, ast.For)
        loop.body = body
        body = [loop]
    elif mode in ("uniform_branch", "varying_branch"):
        condition = (
            "flag" if mode == "uniform_branch" else "cute.arch.thread_idx()[1] < 4"
        )
        branch = _body(f"if {condition}:\n    pass")[0]
        assert isinstance(branch, ast.If)
        branch.body = body
        body = [branch]
    elif mode == "live_name":
        body.insert(0, _body("coordinate = synthetic + 1")[0])
    elif mode == "early_return":
        body.insert(0, _body("if flag:\n    return")[0])
    resolve_pruned_lane_owners(
        body,
        {},
        physical_fallbacks={
            "synthetic": (1, 16 if mode == "wrong_group" else span, "lane_index")
        },
        uniform_names={"end", "flag", "x"},
    )
    accepted = mode in ("complete", "uniform_loop", "uniform_branch", "singleton")
    assert (lanes._find_lane_reduce_call(statement) is None) == accepted
    if accepted:
        lanes.validate_lane_reduce_owners(body)
        assert ("_cute_grouped_reduce_warp" in ast.unparse(statement)) == (span > 1)
    elif mode != "live_loop":
        with pytest.raises(exc.BackendUnsupported):
            lanes.validate_lane_reduce_owners(body)


@pytest.mark.parametrize(
    "mode",
    [
        "two_axes",
        "one_axis",
        "singletons",
        "serial",
        "two_serial",
        "partial",
        "interleaved",
        "shared_axis",
        "missing",
        "ambiguous",
        "cluster",
        "cross_warp",
        "powered",
        "coefficient",
    ],
)
def test_physical_reshape_layout_proof_is_complete(mode: str) -> None:
    first, second = sympy.symbols("b1 b2", integer=True, positive=True)
    sizes = {1: 4, 2: 8, 3: 1}
    axes = {1: 1, 2: 2, 3: None}
    if mode == "one_axis":
        sizes[1], axes[1], axes[2] = 1, None, 1
    elif mode == "singletons":
        sizes[1] = sizes[2] = 1
        axes[1] = axes[2] = None
    elif mode == "interleaved":
        axes = {1: 0, 2: 2, 3: 1}
        sizes = {1: 2, 2: 4, 3: 2}
    elif mode == "shared_axis":
        axes[2] = 1
    elif mode == "cross_warp":
        sizes[2] = 16
    active = {}
    for bid in sizes:
        loop = _loop(
            bid,
            [],
            lane="lane"
            if mode in ("serial", "two_serial") and (bid == 2 or mode == "two_serial")
            else None,
        )
        if axes[bid] is not None:
            loop.block_thread_axes = {bid: axes[bid]}
        active[bid] = [loop]
    if mode == "missing":
        active.pop(1)
    elif mode == "ambiguous":
        active[1].append(_loop(1, [], lane=None))
    numel = first * second
    if mode == "powered":
        numel *= first
    elif mode == "coefficient":
        numel *= 2
    env = SimpleNamespace(
        backend=SimpleNamespace(
            name="cute", thread_linear_index_expr=lambda values: "lanes"
        ),
        block_sizes=[None, None, None, None, SimpleNamespace(numel=numel)],
        get_block_id={first: 1, second: 2}.get,
        canonical_block_id=lambda value: value,
    )
    fn = SimpleNamespace(
        cute_state=SimpleNamespace(simt_cluster_n=2 if mode == "cluster" else 1),
        resolved_block_size=lambda bid: (
            sizes[bid] * (2 if mode == "partial" and bid == 2 else 1)
        ),
        tile_strategy=SimpleNamespace(
            thread_axis_for_block_id=axes.get,
            thread_extent_for_block_id=sizes.get,
        ),
    )
    strategy = object.__new__(reductions.PersistentReductionStrategy)
    strategy.block_ids = [4]
    strategy._fn = lambda: fn
    state = SimpleNamespace(codegen=SimpleNamespace(active_device_loops=active))
    with patch.object(CompileEnvironment, "current", return_value=env):
        group = strategy._reshape_physical_reduction_group_params(state)
    assert group == {
        "two_axes": (1, 32, "lanes"),
        "one_axis": (1, 8, "lanes"),
        "singletons": (1, 1, "0"),
    }.get(mode)


@helion.kernel(backend="cute", static_shapes=True)
def _merged_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)])
    for row in hl.tile(x.size(0)):
        acc = hl.zeros([row], dtype=x.dtype)
        for first, second in hl.tile([x.size(1), x.size(2)]):
            acc += x[row, first, second].reshape(row, -1).sum(-1)
        out[row] = acc
    return out


def _simulate_merged_sum(source: str, x: torch.Tensor) -> np.ndarray:
    """Execute every generated thread, rendezvousing at actual grouped calls."""
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    kernel.decorator_list = []
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    dims = ast.literal_eval(
        next(kw.value for kw in launch.keywords if kw.arg == "block")
    )
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]

    class Rendezvous(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_cute_grouped_reduce_warp"
            ):
                keywords = {kw.arg: kw.value for kw in node.keywords}
                return ast.Yield(
                    value=ast.Tuple(
                        elts=[*node.args, keywords["pre"], keywords["group_span"]],
                        ctx=ast.Load(),
                    )
                )
            return node

    Rendezvous().visit(kernel)
    kernel.body.extend(_body("if False:\n    yield None"))
    actual = np.full(x.shape[0], np.nan, dtype=np.float32)
    writes = np.zeros(x.shape[0], dtype=np.int64)

    class Pointer:
        def __init__(self, array: np.ndarray, offset: int = 0, output: bool = False):
            self.array, self.offset, self.output = array, offset, output

        def __add__(self, offset):
            return Pointer(self.array, self.offset + int(offset), self.output)

        def load(self):
            assert 0 <= self.offset < self.array.size
            return self.array[self.offset]

        def store(self, value):
            assert self.output and 0 <= self.offset < self.array.size
            if writes[self.offset]:
                assert self.array[self.offset] == value
            writes[self.offset] += 1
            self.array[self.offset] = value

    source_array = x.numpy().copy()
    inputs = SimpleNamespace(
        iterator=Pointer(source_array.reshape(-1)),
        layout=SimpleNamespace(stride=x.stride()),
    )
    output = SimpleNamespace(
        iterator=Pointer(actual, output=True), layout=SimpleNamespace(stride=(1,))
    )
    current = [0, (0, 0, 0)]
    namespace = {
        "cutlass": SimpleNamespace(Int32=int, Float32=np.float32),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (current[0], 0, 0), thread_idx=lambda: current[1]
            )
        ),
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[*constants, kernel], type_ignores=[])
    )
    exec(compile(module, "<merged-reduction-model>", "exec"), namespace)
    coordinates = [
        (tx, ty, tz)
        for tz in range(dims[2])
        for ty in range(dims[1])
        for tx in range(dims[0])
    ]
    block_rows = namespace.get("_BLOCK_SIZE_0", 1)
    for block in range((x.shape[0] + block_rows - 1) // block_rows):
        current[0] = block
        workers = [namespace[kernel.name](inputs, output) for _ in coordinates]
        results = [None] * len(workers)
        while True:
            events = []
            for coordinate, worker, value in zip(
                coordinates, workers, results, strict=True
            ):
                current[1] = coordinate
                try:
                    events.append(worker.send(value))
                except StopIteration:
                    events.append(None)
            if all(event is None for event in events):
                break
            assert all(event is not None for event in events), "nonuniform collective"
            results = []
            for index, event in enumerate(events):
                value, kind, identity, lane, pre, span = event
                assert kind == "sum"
                members = [
                    other[0]
                    for j, other in enumerate(events)
                    if j // 32 == index // 32
                    and other[3] // span == lane // span
                    and other[3] % pre == lane % pre
                ]
                assert len(members) == span // pre
                results.append(np.sum(members, dtype=np.float32))
    assert np.all(writes > 0)
    np.testing.assert_array_equal(source_array, x.numpy())
    return actual


@pytest.mark.parametrize("blocks", [(4, 4, 8), (1, 4, 8), (1, 1, 8), (1, 1, 1)])
@pytest.mark.parametrize("shape", [(3, 4, 5), (5, 9, 13)])
def test_merged_reduction_physical_codegen_values(blocks, shape) -> None:
    from ._cute_binding import _cpu_bind
    from .cute_population_contracts import _target

    x = (torch.arange(np.prod(shape), dtype=torch.float32) % 17 - 8).reshape(shape)
    with _target():
        bound = _cpu_bind(_merged_sum_kernel, (x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = list(blocks)
        source = bound.to_code(config)
    assert "_helion_lane_reduce" not in source
    np.testing.assert_array_equal(
        _simulate_merged_sum(source, x), x.sum((1, 2)).numpy()
    )


def test_merged_reduction_multiple_serial_sources_stay_rejected() -> None:
    from ._cute_binding import _cpu_bind
    from .cute_population_contracts import _target

    with _target():
        bound = _cpu_bind(_merged_sum_kernel, (torch.ones(3, 4, 5),))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[1, 4, 8], num_threads=[1, 1, 1])
        with pytest.raises(exc.BackendUnsupported, match="lane owner"):
            bound.to_code(config)


def test_merged_reduction_generated_live_coordinate_stays_rejected() -> None:
    from ._cute_binding import _cpu_bind
    from .cute_population_contracts import _target
    from helion._compiler.cute import nested_lane_reductions

    original = nested_lane_reductions.resolve_pruned_lane_owners

    def keep_coordinate(body, fallbacks, **kwargs):
        proofs = kwargs["physical_fallbacks"]
        assert len(proofs) == 1
        coordinate = next(iter(proofs))
        body.insert(0, _body(f"coordinate_is_live = {coordinate} + 1")[0])
        return original(body, fallbacks, **kwargs)

    with _target():
        bound = _cpu_bind(_merged_sum_kernel, (torch.ones(3, 4, 5),))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [1, 4, 8]
        with (
            patch.object(
                nested_lane_reductions, "resolve_pruned_lane_owners", keep_coordinate
            ),
            pytest.raises(exc.BackendUnsupported, match="lane owner"),
        ):
            bound.to_code(config)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("blocks", [(4, 4, 8), (1, 4, 8), (1, 1, 8), (1, 1, 1)])
def test_merged_reduction_physical_native(blocks) -> None:
    for shape in ((3, 4, 5), (5, 9, 13)):
        x = (
            torch.arange(np.prod(shape), device=DEVICE, dtype=torch.float32) % 17 - 8
        ).reshape(shape)
        snapshot = x.clone()
        bound = _merged_sum_kernel.bind((x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = list(blocks)
        actual = bound.compile_config(config)(x)
        torch.testing.assert_close(actual, x.sum((1, 2)), rtol=0, atol=0)
        torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
