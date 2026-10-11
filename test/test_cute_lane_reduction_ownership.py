from __future__ import annotations

import ast
from itertools import accumulate
import math
import operator
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from examples.layer_norm import layer_norm_bwd
from examples.welford import welford
import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_interchanged_store_dce import _PAIRS
from test.test_cute_interchanged_store_dce import _execute as _execute_interchange
from test.test_cute_interchanged_store_dce import _program
from test.test_cute_sibling_layout_safety import _config

import helion
from helion._compiler import reduction_strategy as reductions
from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import HELION_VEC_LANE_OF_ATTR
from helion._compiler.ast_read_writes import ast_rename
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_id_reduced_store(x: torch.Tensor, out: torch.Tensor, masked: hl.constexpr):
    for row, col in hl.tile(x.shape):
        reduced = x[row, col].sum(-1)
        if masked:
            hl.store(out, [row, col.id], reduced, extra_mask=row.index % 2 == 0)
        else:
            out[row, col.id] = reduced
    return out


def _tile_id_program(rows: int, columns: int, masked: bool, threads=32):
    x = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    out = torch.full((rows, (columns + 127) // 128), -1.0)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_tile_id_reduced_store, (x, out, masked))
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[4, 128], num_threads=[1, 1 if columns == 1 else threads]
        )
        code = bound.to_code(config)
    return code, x, out


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rows,columns", [(3, 65), (7, 129), (5, 513)])
def test_tile_id_reduction_store_uses_scalar_tile_bounds(rows, columns, masked):
    code, _x, _out = _tile_id_program(rows, columns, masked)
    tree = ast.parse(code)
    device = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
    )
    stores = [
        n
        for n in ast.walk(device)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "store"
    ]
    assert len(stores) == 1
    parents = {
        child: parent
        for parent in ast.walk(device)
        for child in ast.iter_child_nodes(parent)
    }
    node = stores[0]
    guards = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.For):
            assert "lane_1" not in ast.unparse(node.target)
        if isinstance(node, ast.If):
            guards.append(ast.unparse(node.test))
    assert any(f"tile_offset_1 < {columns}" in test for test in guards)
    assert all("mask_1" not in test for test in guards)
    assert any("lane_idx" in test for test in guards)
    if masked:
        assert "_cute_python_mod" in code or "% 2" in code


def _execute_tile_id_program(code, x, output, *, require_unique_writes=True):
    """Replay the actual scalar body with an exact-integer warp-sum oracle."""
    from test.test_cute_affine_vector_io import _Pointer

    current = [0, 0]
    reads = dict.fromkeys(("scalar_loads", "scalar_stores"), 0)
    writes = []
    data = x.flatten().tolist()
    result = output.flatten().tolist()

    class OutputPointer(_Pointer):
        def __add__(self, offset):
            return OutputPointer(self.values, self.stats, self.offset + offset)

        def store(self, value):
            writes.append(self.offset)
            super().store(value)

    class Collectives(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node.func) == "cute.arch.warp_reduction_sum":
                assert len(node.args) == 1
                assert [
                    (kw.arg, ast.literal_eval(kw.value)) for kw in node.keywords
                ] == [("threads_in_group", 32)]
                return ast.Yield(node.args[0])
            return self.generic_visit(node)

    tree = ast.parse(code)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
    )
    function.decorator_list = []
    function = Collectives().visit(function)
    namespace = {
        "cutlass": SimpleNamespace(
            Int32=int,
            Int64=int,
            Float32=np.float32,
            Boolean=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (current[1], 0, 0),
                lane_idx=lambda: current[1],
                block_idx=lambda: (current[0], 0, 0),
            )
        ),
        "operator": operator,
        "_cute_python_mod": operator.mod,
        "_BLOCK_SIZE_0": 4,
        "_BLOCK_SIZE_1": 128,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([function], [])),
            "<tile-id-replay>",
            "exec",
        ),
        namespace,
    )
    args = {
        "x": SimpleNamespace(
            iterator=_Pointer(data, reads), layout=SimpleNamespace(stride=x.stride())
        ),
        "out": SimpleNamespace(
            iterator=OutputPointer(result, reads),
            layout=SimpleNamespace(stride=output.stride()),
        ),
    }
    assert {arg.arg for arg in function.args.args} == set(args)
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_launcher"
    )
    grid = eval(
        compile(ast.Expression(launch.args[1]), "<tile-id-grid>", "eval"), namespace
    )
    assert len(grid) == 1
    for block in range(grid[0]):
        current[0] = block
        if not any(isinstance(n, ast.Yield) for n in ast.walk(function)):
            current[1] = 0
            namespace[function.name](**args)
            continue
        generators = [namespace[function.name](**args) for _ in range(32)]
        sent = None
        while True:
            active = []
            for thread, generator in enumerate(generators):
                current[1] = thread
                try:
                    active.append(generator.send(sent))
                except StopIteration:
                    active.append(None)
            if all(value is None for value in active):
                break
            assert all(value is not None for value in active)
            # Integer-valued inputs keep all sums exactly representable FP32.
            sent = np.float32(sum(float(value) for value in active))
    if require_unique_writes:
        assert len(writes) == len(set(writes)), "duplicate reduction writer"
    return torch.tensor(result).reshape(output.shape), writes


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rows,columns", [(3, 1), (3, 65), (7, 129), (5, 513)])
def test_tile_id_reduction_store_generated_values(rows, columns, masked):
    code, x, out = _tile_id_program(rows, columns, masked)
    actual, writes = _execute_tile_id_program(code, x, out)
    expected = torch.stack(
        [x[:, col : col + 128].sum(-1) for col in range(0, columns, 128)], dim=-1
    )
    if masked:
        expected[1::2] = -1
    assert torch.equal(actual, expected)
    assert len(writes) == ((rows + 1) // 2 if masked else rows) * out.shape[1]


def _marker(owner: str | None, value: str = "partial") -> str:
    return lanes._lane_reduce_marker_expr(
        value, "sum", "cutlass.Float32(0)", 32, owner_lane=owner
    )


def _source(body: list[ast.AST]) -> str:
    return ast.unparse(ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[]))


def _stamped(body: list[ast.AST], lane_var: str = "lane") -> list[ast.AST]:
    """Mark ``for vec_lane in cutlass.range_constexpr(V)`` loops as ``lane_var``'s
    vector lane, as ``VecLaneWrapper`` does for the loops the strategies build."""
    for stmt in body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "vec_lane"
            ):
                setattr(node, HELION_VEC_LANE_OF_ATTR, lane_var)
    return body


def _body(source: str) -> list[ast.AST]:
    return list(ast.parse(source).body)


@pytest.mark.parametrize("outer,inner", [("feature", "row"), ("a", "b")])
@pytest.mark.parametrize("inner_extent", [2, 8])
def test_distinct_lane_owner_rejected_even_for_equal_extents(
    outer: str, inner: str, inner_extent: int
) -> None:
    inner_loop = lanes._create_lane_loop(
        inner,
        inner_extent,
        _body(
            f"partial = {outer} + {inner}\n"
            f"reduced = {_marker(outer)}\n"
            "sink.store(reduced)"
        ),
    )
    loop = lanes._create_lane_loop(outer, 8, [inner_loop])
    with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
        lanes.validate_lane_reduce_owners([loop])
    with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
        lanes.split_lane_loop_reductions([loop])


def test_marker_owner_survives_text_and_lane_cloning() -> None:
    statement = ast.parse(f"reduced = {_marker('axis')}").body[0]
    cloned = lanes._clone_stmt(statement)
    marker = lanes._is_lane_reduce_marker_assign(cloned)
    assert marker is not None and marker.owner_lane == "axis"
    loop = lanes._create_lane_loop("axis", 8, [statement])
    copied_loop = lanes._clone_lane_loop_with_body(loop, [cloned])
    lanes.validate_lane_reduce_owners([copied_loop])


def test_missing_lane_owner_cannot_reach_residual_restore() -> None:
    body = _body(f"reduced = {_marker('missing')}")
    with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
        lanes.validate_lane_reduce_owners(body)
    with pytest.raises(helion.exc.BackendUnsupported, match="no proved lane lowering"):
        lanes.restore_unprocessed_lane_reduce_markers(body)


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("copies", [1, 2])
def test_late_rename_carry_rejected_before_split_subpaths(
    guarded: bool, copies: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "snapshot = carried\n"
    if copies == 2:
        text += "snapshot_2 = snapshot\n"
    source = "snapshot_2" if copies == 2 else "snapshot"
    text += f"partial = lane + 1\nreduced = {_marker('lane')}\n"
    update = f"next_value = {source} + partial"
    text += (f"if external:\n    {update}" if guarded else update) + "\n"
    text += "sink.store(reduced)"
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    body: list[ast.AST] = [loop, *_body("later.store(carried)")]
    monkeypatch.setattr(
        lanes,
        "_split_lane_loop_with_register_stash",
        lambda *_: pytest.fail("subpath selected before carry legality"),
    )
    lanes.validate_lane_reduce_owners(body)
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            body, rename_groups={"next_value": "carried", "carried": "carried"}
        )


@pytest.mark.parametrize("owner", [None, "lane"])
def test_reduced_scalar_can_update_an_existing_online_carry(owner: str | None) -> None:
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"snapshot = carried\npartial = lane + 1\nreduced = {_marker(owner)}\n"
            "next_value = snapshot + reduced\nsink.store(next_value)"
        ),
    )
    lanes.validate_lane_reduce_owners([loop])
    result = lanes.split_lane_loop_reductions(
        [loop], rename_groups={"next_value": "carried", "carried": "carried"}
    )
    code = _source(lanes.restore_unprocessed_lane_reduce_markers(result))
    assert "warp_reduction_sum" in code
    assert code.count("next_value = snapshot + reduced") == 1
    assert "_helion_lane_reduce" not in code


def test_legacy_per_lane_carry_is_not_an_owned_reduction_proof() -> None:
    def generate(owner: str | None) -> str:
        loop = lanes._create_lane_loop(
            "lane",
            8,
            _body(
                "partial = lane + 1\ncarried = carried + partial\n"
                f"reduced = {_marker(owner)}\n"
                "sink.store(reduced + carried)"
            ),
        )
        lanes.validate_lane_reduce_owners([loop])
        result = lanes.split_lane_loop_reductions([loop])
        return _source(lanes.restore_unprocessed_lane_reduce_markers(result))

    assert "reduced = partial" in generate(None)
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        generate("lane")


class _Sink:
    def __init__(self, values: dict[int, float], offset: int = 0) -> None:
        self.values = values
        self.offset = offset

    @property
    def iterator(self) -> _Sink:
        return self

    def __add__(self, offset: int) -> _Sink:
        return _Sink(self.values, self.offset + offset)

    def store(self, value: float) -> None:
        self.values[self.offset] = value


def _execute_scalars(
    body: list[ast.AST], renames: dict[str, str] | None = None
) -> tuple[dict[str, dict[int, float]], int]:
    """Execute exact small-integer scalar schedules, not CUDA collectives."""
    calls = 0

    def grouped_reduce(
        value: float,
        operation: str,
        identity: float,
        lane: int,
        lane_in_group: int,
        lane_mod_pre: int,
        *,
        pre: int,
        group_span: int,
        group_count: int,
    ) -> float:
        nonlocal calls
        assert operation == "sum" and identity == 0
        assert lane == lane_in_group == lane_mod_pre == 0
        assert pre == group_count == 1 and group_span == 64
        calls += 1
        # All 64 inputs are identical finite integers. Any FP32 sum tree has
        # this exact result; this models the complete installed call signature.
        return value * group_span

    stores: dict[str, dict[int, float]] = {
        name: {} for name in ("out", "raw_out", "late_out")
    }
    namespace = {
        "cutlass": SimpleNamespace(
            Float32=float, Int32=int, range=range, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda extent, dtype: [0.0] * extent,
            arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0)),
        ),
        "_cute_grouped_reduce_shared_two_stage": grouped_reduce,
        **{name: _Sink(values) for name, values in stores.items()},
    }
    module = ast.parse(_source(body))
    ast_rename(module, renames or {})
    exec(
        compile(ast.fix_missing_locations(module), "<lane-carry-proof>", "exec"),
        namespace,
    )
    return stores, calls


@pytest.mark.parametrize("has_raw", [False, True])
@pytest.mark.parametrize("unduplicatable", [False, True])
def test_mixed_raw_and_renamed_carries_decline_before_stash(
    has_raw: bool, unduplicatable: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
        if unduplicatable
        else "lane + 1"
    )
    text = f"partial = {partial}\nsnapshot = late\nnext_late = snapshot + partial\n"
    if has_raw:
        text += "raw = raw + partial\n"
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    text += f"reduced = {marker}\n(out.iterator + lane).store(partial + reduced)"
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    prefix, suffix = (
        _body("raw = 0.0\nlate = 0.0"),
        _body("raw_out.store(raw)\nlate_out.store(late)"),
    )
    renames = {"next_late": "late", "late": "late"}
    expected = 36.0 * (64 if unduplicatable else 1)
    # Give the reference the complete eight-lane reduction, not a raw-input
    # stand-in; both external carry updates are separately observable.
    reference_loop = lanes._create_lane_loop(
        "lane", 8, _body(text.replace(marker, repr(expected)))
    )
    reference, calls = _execute_scalars([*prefix, reference_loop, *suffix], renames)
    assert reference["raw_out"][0] == (expected if has_raw else 0.0)
    assert reference["late_out"][0] == expected
    assert calls == (8 if unduplicatable else 0)
    monkeypatch.setattr(
        lanes,
        "_split_lane_loop_with_register_stash",
        lambda *_: pytest.fail("stash selected before observable carry proof"),
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            [*prefix, loop, *suffix], rename_groups=renames
        )


def test_owned_raw_restore_would_lose_complete_reduction_numerically() -> None:
    def make(owner: str | None) -> list[ast.AST]:
        marker = lanes._lane_reduce_marker_expr(
            "partial", "sum", "cutlass.Float32(0)", 1, owner_lane=owner
        )
        return [
            *_body("carried = 0.0"),
            lanes._create_lane_loop(
                "lane",
                8,
                _body(
                    "partial = lane + 1\ncarried = carried + partial\n"
                    f"reduced = {marker}\n"
                    "(out.iterator + lane).store(reduced + carried)"
                ),
            ),
            *_body("raw_out.store(carried)"),
        ]

    legacy = lanes.split_lane_loop_reductions(make(None))
    observed, calls = _execute_scalars(legacy)
    expected = [36.0 + value for value in accumulate(range(1, 9))]
    assert observed["raw_out"][0] == 36.0 and calls == 0
    assert observed["out"][7] == 44.0 and expected[-1] == 72.0
    assert list(observed["out"].values()) != expected
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(make("lane"))


@pytest.mark.parametrize("unduplicatable", [False, True])
def test_complete_owned_split_has_numeric_and_single_execution_proof(
    unduplicatable: bool,
) -> None:
    partial = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
        if unduplicatable
        else "lane + 1"
    )
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"partial = {partial}\nreduced = {marker}\n"
            "(out.iterator + lane).store(partial + reduced)"
        ),
    )
    lowered = lanes.split_lane_loop_reductions([loop])
    values, calls = _execute_scalars(lowered)
    scale = 64 if unduplicatable else 1
    assert values["out"] == {
        index: float((36 + index + 1) * scale) for index in range(8)
    }
    assert calls == (8 if unduplicatable else 0)


def test_marker_dependent_online_update_uses_full_lane_sum_once() -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"snapshot = carried\npartial = lane + 1\nreduced = {marker}\n"
            "next_value = snapshot + reduced\nout.store(next_value)"
        ),
    )
    renames = {"next_value": "carried", "carried": "carried"}
    lowered = lanes.split_lane_loop_reductions(
        [*_body("carried = 6.0"), loop], rename_groups=renames
    )
    values, calls = _execute_scalars(lowered, renames)
    assert values["out"] == {0: 42.0} and calls == 0


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("guarded", [False, True])
def test_mixed_marker_and_partial_carry_remains_lane_varying(
    renamed: bool, indirect: bool, guarded: bool
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    text = f"partial = lane + 1\nsnapshot = carried\nreduced = {marker}\n"
    if indirect:
        text += "mixed = partial + reduced\n"
    target = "next_value" if renamed else "carried"
    expression = "mixed" if indirect else "partial + reduced"
    update = f"{target} = snapshot + {expression}"
    text += (f"if flag:\n    {update}" if guarded else update) + "\n"
    text += "(out.iterator + lane).store(partial + reduced)"
    renames = {"next_value": "carried", "carried": "carried"} if renamed else {}
    prefix = _body("carried = 0.0\nflag = True")
    suffix = _body("late_out.store(carried)")
    reference = lanes._create_lane_loop("lane", 8, _body(text.replace(marker, "36.0")))
    values, calls = _execute_scalars([*prefix, reference, *suffix], renames)
    assert values["late_out"][0] == 324.0 and calls == 0
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    indices = {
        index
        for index, statement in enumerate(loop.body)
        if lanes._is_lane_reduce_marker_assign(statement) is not None
    }
    normalized = [lanes._clone_stmt(statement) for statement in loop.body]
    for statement in normalized:
        ast_rename(statement, renames)
    # The legacy marker-taint test misses the independent partial in both
    # direct and aliased expressions. Post-finalization variation must not.
    assert not lanes._has_extra_cross_lane_carry(normalized, "lane", indices)
    assert lanes._has_extra_cross_lane_carry(
        normalized, "lane", indices, finalized_markers=True
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            [*prefix, loop, *suffix], rename_groups=renames
        )


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_finalized_only_carry_is_uniform_after_reduction(
    renamed: bool, indirect: bool
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    text = f"partial = lane + 1\nsnapshot = carried\nreduced = {marker}\n"
    if indirect:
        text += "copy_reduced = reduced\n"
    target = "next_value" if renamed else "carried"
    text += f"{target} = snapshot + {'copy_reduced' if indirect else 'reduced'}\n"
    text += f"out.store({target})"
    renames = {"next_value": "carried", "carried": "carried"} if renamed else {}
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    lowered = lanes.split_lane_loop_reductions(
        [*_body("carried = 6.0"), loop], rename_groups=renames
    )
    values, calls = _execute_scalars(lowered, renames)
    assert values["out"] == {0: 42.0} and calls == 0


@pytest.mark.parametrize("unduplicatable", [False, True])
@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("inside_observed", [False, True])
def test_finalized_carry_requires_a_once_per_tile_schedule(
    unduplicatable: bool, renamed: bool, inside_observed: bool
) -> None:
    partial = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
        if unduplicatable
        else "64 * (lane + 1)"
    )
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    target = "next_value" if renamed else "carried"
    renames = {"carried": "carried", "next_value": "carried"} if renamed else {}
    consumer = target if inside_observed else "reduced"
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"partial = {partial}\nsnapshot = carried\nreduced = {marker}\n"
            f"{target} = snapshot + reduced\n"
            f"(out.iterator + lane).store(partial + {consumer})"
        ),
    )
    program = [*_body("carried = 6.0"), loop, *_body("late_out.store(carried)")]
    if unduplicatable:
        with pytest.raises(helion.exc.BackendUnsupported, match="once-per-tile"):
            lanes.split_lane_loop_reductions(program, rename_groups=renames)
    else:
        lowered = lanes.split_lane_loop_reductions(program, rename_groups=renames)
        values, calls = _execute_scalars(lowered, renames)
        assert values["late_out"] == {0: 2310.0} and calls == 0
        assert values["out"] == {
            lane: float((lane + 1) * 64 + (2310 if inside_observed else 2304))
            for lane in range(8)
        }


@pytest.mark.parametrize("compound", [False, True])
def test_finalized_carry_without_a_single_assignment_proof_declines(
    compound: bool,
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    update = (
        "if flag:\n    carried = carried + reduced"
        if compound
        else "carried = carried + reduced\ncarried = carried + 1"
    )
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(f"partial = lane + 1\nreduced = {marker}\n{update}"),
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="once-per-tile"):
        lanes.split_lane_loop_reductions(
            [*_body("carried = 6.0\nflag = True"), loop, *_body("out.store(carried)")]
        )


def test_owned_marker_cannot_use_an_unproved_direct_restore() -> None:
    loop = lanes._create_lane_loop(
        "lane", 8, _body(f"partial = lane + 1\nreduced = {_marker('lane')}")
    )
    marker = lanes._is_lane_reduce_marker_assign(loop.body[1])
    assert marker is not None
    with pytest.raises(
        helion.exc.BackendUnsupported, match="complete per-lane restore"
    ):
        lanes._restore_per_lane_markers(loop, [(1, marker)])


def test_strided_restore_marker_is_finalized_per_lane() -> None:
    # A device-lane-loop marker whose body kept the per-element strided
    # semantics is finalized across the thread group in place when the
    # two-pass split is refused, instead of rejecting the config.
    text = lanes._lane_reduce_marker_expr(
        "partial",
        "sum",
        "cutlass.Float32(0)",
        32,
        owner_lane="lane",
        strided_restore=True,
    )
    loop = lanes._create_lane_loop(
        "lane", 8, _body(f"partial = lane + 1\nreduced = {text}")
    )
    marker = lanes._is_lane_reduce_marker_assign(loop.body[1])
    assert marker is not None
    assert marker.owner_lane == "lane"
    assert marker.strided_restore
    restored = lanes._restore_per_lane_markers(loop, [(1, marker)])
    source = _source(list(restored.body))
    assert "_helion_lane_reduce" not in source
    assert "warp_reduction_sum" in source
    assert "reduced = " in source


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("dependent", [False, True])
def test_ordinary_owner_proof_does_not_change_supported_split(
    guarded: bool, dependent: bool
) -> None:
    def generate(owner: str | None) -> str:
        text = f"partial = lane + 1\nreduced = {_marker(owner)}\n"
        if dependent:
            text += "other = partial + reduced\n"
            text += f"reduced_again = {_marker(owner, 'other')}\n"
        text += f"sink.store({'reduced_again' if dependent else 'reduced'})"
        body: list[ast.AST] = _body(text)
        if guarded:
            body = [
                ast.If(
                    test=ast.Name(id="flag", ctx=ast.Load()),
                    body=cast("list[ast.stmt]", body),
                    orelse=[],
                )
            ]
        loop = lanes._create_lane_loop("lane", 8, body)
        lanes.validate_lane_reduce_owners([loop])
        result = lanes.split_lane_loop_reductions([loop], uniform_names={"flag"})
        return _source(lanes.restore_unprocessed_lane_reduce_markers(result))

    assert generate(None) == generate("lane")


def test_supported_serial_interchange_preserves_owned_marker() -> None:
    def generate(owned: bool) -> str:
        loop = _program()
        if owned:
            for node in ast.walk(loop):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "_helion_lane_reduce":
                        assert len(node.args) == 8
                        node.args.extend(
                            [ast.Constant(value=1), ast.Constant(value="lane")]
                        )
        lanes.validate_lane_reduce_owners([loop])
        body = lanes.interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )
        lanes.validate_lane_reduce_owners(body)
        body = lanes.split_lane_loop_reductions(
            body,
            uniform_names={"rows", "valid_rows", "valid_cols"},
            proven_disjoint_tensor_pairs=_PAIRS,
        )
        return _source(lanes.restore_unprocessed_lane_reduce_markers(body))

    assert generate(False) == generate(True)


@pytest.mark.parametrize(
    ("rows", "valid_rows", "valid_cols"), [(3, 3, 4), (3, 2, 3), (0, 0, 4)]
)
def test_owned_serial_interchange_has_complete_numeric_reduction(
    rows: int, valid_rows: int, valid_cols: int
) -> None:
    loop = _program()
    for node in ast.walk(loop):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_helion_lane_reduce"
        ):
            node.args.extend([ast.Constant(value=1), ast.Constant(value="lane")])
    lanes.validate_lane_reduce_owners([loop])
    lowered = lanes.interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    lanes.validate_lane_reduce_owners(lowered)
    actual = _execute_interchange(lowered, rows, valid_rows, valid_cols)
    values = actual["source"].values.reshape(rows, 4)
    mask = (np.arange(rows)[:, None] < valid_rows) & (
        np.arange(4)[None, :] < valid_cols
    )
    masked = np.where(mask, values, 0.0)
    product = masked * actual["weight"].values
    expected = np.where(mask, product - 2 * product.sum(-1, keepdims=True), -99.0)
    np.testing.assert_array_equal(actual["output"].values.reshape(rows, 4), expected)
    np.testing.assert_array_equal(actual["column"].values, masked.sum(0))
    assert actual["output"].stores == valid_rows * valid_cols
    assert actual["column"].stores == 4


@skipUnlessBackends(["cute"])
def test_exact_rejected_backward_config_declines_before_emission() -> None:
    x = torch.empty((4096, 4096), dtype=torch.bfloat16)
    args = (
        torch.empty_like(x),
        x,
        torch.empty(4096),
        torch.empty(4096),
        torch.empty(4096, dtype=torch.bfloat16),
        True,
    )
    kernel = helion.kernel(
        layer_norm_bwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_region_fission=True,
        cute_materialize_transformed_operands=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    config = helion.Config.from_dict(
        {
            "block_sizes": [256, 256],
            "cute_cluster_n": 1,
            "cute_host_paired_sum": "off",
            "cute_lane_layouts": ["blocked", "blocked", "blocked"],
            "cute_min_blocks_per_mp": 4,
            "cute_reduction_group_rows": 4,
            "cute_reduction_reloads": ["gmem"],
            "cute_reduction_schedule": "pipelined",
            "cute_reduction_sequence": "bounded_layout",
            "cute_vector_widths": [2, 8, 1],
            "load_eviction_policies": ["streaming", "last", "last", "l2_last", "first"],
            "num_threads": [0, 32, 256],
        }
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, args)
        with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
            bound.to_code(config)


# --- a reduction input a device loop of the lane body accumulates -----------

_TID = "cutlass.Int32(cute.arch.thread_idx()[0])"
_COLLECTIVE_END = (
    "amax = _cute_grouped_reduce_shared_two_stage(length, 'max', cutlass.Int64(0), "
    "0, 0, 0, pre=1, group_span=256, group_count=1)\n"
)
# The device function's aliases of the carried values: the jagged loop's
# output ``v_8`` is ``row_sums``, the lane body's ``v_9`` is ``mean_acc``.
_ROW_SUMS_RENAMES = {
    "v_8": "row_sums",
    "row_sums": "row_sums",
    "v_9": "mean_acc",
    "mean_acc": "mean_acc",
}


def _masked_marker(value: str, *, strided_restore: bool = True) -> str:
    return lanes._lane_reduce_marker_expr(
        value,
        "sum",
        "cutlass.Float32(0)",
        256,
        group_pre=1,
        group_span=256,
        group_lane_expr=_TID,
        group_count=1,
        owner_lane="lane",
        strided_restore=strided_restore,
    )


def _accumulated_rows_body(
    *, collective_end: bool, strided_restore: bool = True
) -> str:
    """The nested-row seed's column lane body under dynamic shapes.

    A jagged device loop accumulates ``row_sums`` under its loop-output name
    ``v_8`` (``row_sums_copy = row_sums`` at the top of its body is the
    carry's phi copy); the column mask then turns ``row_sums.sum()`` into a
    masked lane reduction of ``_mask_to``, and the result updates the
    cross-lane ``mean_acc`` carry.
    """
    end = _COLLECTIVE_END if collective_end else "amax = length\n"
    return (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "mask = indices < M\n"
        "mean_acc_copy = mean_acc\n"
        "row_sums = cutlass.Float32(0.0)\n"
        f"{end}"
        "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(amax), "
        "cutlass.Int32(32)):\n"
        "    row_sums_copy = row_sums\n"
        "    partial = (x.iterator + cutlass.Int32(tile_offset_2 * M + indices)"
        " * cutlass.Int32(x.layout.stride[0])).load() if mask else "
        "cutlass.Float32(0)\n"
        "    v_8 = row_sums_copy + partial\n"
        "_mask_to = cutlass.Float32(row_sums) if mask else cutlass.Float32(0)\n"
        f"sum_2 = cutlass.Float32({_masked_marker('_mask_to', strided_restore=strided_restore)})\n"
        "v_9 = mean_acc_copy + sum_2\n"
    )


def _split(
    text: str,
    renames: dict[str, str],
    disjoint: set[frozenset[str]] | None = None,
) -> str:
    loop = lanes._create_lane_loop("lane", 2, _body(text))
    lanes.validate_lane_reduce_owners([loop])
    result = lanes.split_lane_loop_reductions(
        [loop], rename_groups=renames, proven_disjoint_tensor_pairs=disjoint
    )
    return _source(lanes.restore_unprocessed_lane_reduce_markers(result))


def _lane_passes(code: str) -> list[str]:
    """The bodies of the lane loops of ``code``, in order."""
    return [
        _source(list(node.body))
        for node in ast.parse(code).body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "lane"
    ]


def test_lane_reduction_of_an_accumulated_row_runs_after_the_loop_that_accumulates_it() -> (
    None
):
    """The accumulate pass folds ``row_sums`` only after the jagged loop ran.

    The slice producing the reduction input reads the rename groups: the loop
    writes ``v_8``, which is ``row_sums``.  The two-pass split used to fold
    the fresh zero in a first pass and run the loop in the second (the
    dynamic-shape nested-row seeds of jagged_layer_norm returned mean =
    variance = 0).
    """
    code = _split(_accumulated_rows_body(collective_end=False), _ROW_SUMS_RENAMES)
    accumulate = _lane_passes(code)[0]
    assert "sum_2_lane_acc = sum_2_lane_acc +" in accumulate
    assert accumulate.index("for tile_offset_2 in") < accumulate.index(
        "sum_2_lane_acc = sum_2_lane_acc +"
    )
    assert code.count("v_9 = mean_acc_copy + sum_2") == 1
    assert code.index("sum_2 = cutlass.Float32(sum_2_lane_acc_reduced)") < code.index(
        "v_9 = mean_acc_copy + sum_2"
    )


def test_an_accumulated_row_under_a_collective_end_is_reduced_per_lane() -> None:
    """The jagged loop's trip count is a cross-thread max, which no pass may re-run: the owned marker is restored per lane, after the loop, and an owner without a per-lane restore is declined rather than split."""
    code = _split(_accumulated_rows_body(collective_end=True), _ROW_SUMS_RENAMES)
    (body,) = _lane_passes(code)
    assert "_lane_acc" not in code
    assert body.index("for tile_offset_2 in") < body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    assert body.index("sum_2 = cutlass.Float32(_mask_to_reduced)") < body.index(
        "v_9 = mean_acc_copy + sum_2"
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            _accumulated_rows_body(collective_end=True, strided_restore=False),
            _ROW_SUMS_RENAMES,
        )


def test_a_carry_the_rename_groups_do_not_map_is_read_off_its_phi_copy() -> None:
    """Without the alias of ``v_8`` the slice sees the initializer alone; the loop's ``row_sums_copy = row_sums`` marks it as the carry, and the marker is restored per lane instead of being folded ahead of the loop."""
    code = _split(_accumulated_rows_body(collective_end=False), {})
    (body,) = _lane_passes(code)
    assert "_lane_acc" not in code
    assert body.index("for tile_offset_2 in") < body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(_accumulated_rows_body(collective_end=False, strided_restore=False), {})


def test_a_loop_reading_the_reduction_input_without_carrying_it_stays_a_consumer() -> (
    None
):
    """A loop that only reads an input of the reduction (no phi copy) is no producer: the split keeps it in the consume pass."""
    text = (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "mask = indices < M\n"
        "mean_acc_copy = mean_acc\n"
        "row_sums = (x.iterator + cutlass.Int32(indices) * "
        "cutlass.Int32(x.layout.stride[0])).load() if mask else cutlass.Float32(0)\n"
        "for k in range(4):\n"
        "    scaled = row_sums * k\n"
        "_mask_to = cutlass.Float32(row_sums) if mask else cutlass.Float32(0)\n"
        f"sum_2 = cutlass.Float32({_masked_marker('_mask_to')})\n"
        "v_9 = mean_acc_copy + sum_2\n"
    )
    code = _split(text, _ROW_SUMS_RENAMES)
    accumulate, consume = _lane_passes(code)
    assert "sum_2_lane_acc = sum_2_lane_acc +" in accumulate
    assert "for k in range(4)" not in accumulate
    assert "for k in range(4)" in consume


_XYZ_DISJOINT = {frozenset({"x", "y"}), frozenset({"x", "z"}), frozenset({"y", "z"})}
# The K loop's output ``v_8`` and the later rewrite ``v_11`` are both ``acc``.
_ACCUMULATED_RENAMES = {
    "v_8": "acc",
    "v_11": "acc",
    "acc": "acc",
    "v_9": "mean_acc",
    "mean_acc": "mean_acc",
}
_CONTINUING_LOOP = (
    "for tile_offset_3 in range(cutlass.Int32(0), cutlass.Int32(4), "
    "cutlass.Int32(1)):\n"
    "    acc_copy_1 = acc\n"
    "    v_11 = acc_copy_1 + (z.iterator + cutlass.Int32(tile_offset_3 * M + "
    "indices)).load()\n"
)
_JOIN_REWRITE = (
    "if flag:\n    v_11 = acc + cutlass.Float32(1.0)\nelse:\n    v_11 = acc\n"
)


def _stored_accumulator_body(*, after_marker: str = "") -> str:
    """A per-column accumulator reduced across the columns and stored.

    A K loop accumulates ``acc`` under its loop-output name ``v_8``, the
    unmasked (static-shape) reduction of ``acc`` across the column lanes
    updates the carried ``mean_acc``, and ``acc`` is stored per column, after
    ``after_marker`` rewrote it under another spelling of its group.
    """
    return (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "mean_acc_copy = mean_acc\n"
        "acc = cutlass.Float32(0.0)\n"
        "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(4), "
        "cutlass.Int32(1)):\n"
        "    acc_copy = acc\n"
        "    partial = (x.iterator + cutlass.Int32(tile_offset_2 * M + indices))"
        ".load()\n"
        "    v_8 = acc_copy + partial\n"
        f"sum_2 = cutlass.Float32({_masked_marker('acc')})\n"
        "v_9 = mean_acc_copy + sum_2\n"
        f"{after_marker}"
        "(y.iterator + cutlass.Int32(indices)).store(cutlass.Float32(acc))\n"
    )


def test_the_initializer_of_an_accumulator_a_lane_varying_loop_rewrites_runs_in_every_pass() -> (
    None
):
    """``acc = 0`` is lane-varying through the K loop's rewrite of ``acc``.

    The loop writes ``acc`` under ``v_8``; the classification of the consume
    pass read the names as spelled, took the initializer for lane-invariant
    and emitted it once between the passes, so the consume pass's loop
    continued every lane after the first from the previous lane's final
    ``acc`` and stored that.
    """
    code = _split(_stored_accumulator_body(), _ACCUMULATED_RENAMES, _XYZ_DISJOINT)
    accumulate, consume = _lane_passes(code)
    init = "acc = cutlass.Float32(0.0)"
    assert code.count(init) == 2
    assert (
        accumulate.index(init)
        < accumulate.index("for tile_offset_2 in")
        < accumulate.index("sum_2_lane_acc = sum_2_lane_acc +")
    )
    assert (
        consume.index(init)
        < consume.index("for tile_offset_2 in")
        < consume.index(".store(")
    )
    assert code.count("v_9 = mean_acc_copy + sum_2") == 1


@pytest.mark.parametrize(
    "rewrite", [_CONTINUING_LOOP, _JOIN_REWRITE], ids=["loop", "join"]
)
def test_a_rewrite_of_the_reduction_input_after_its_marker_is_a_consumer(
    rewrite: str,
) -> None:
    """A later loop or if-join rewriting ``acc`` is no producer of the reduction.

    The producer slice is read over the statements before the marker: sliced
    over the whole body, the later rewrite of ``acc``'s group was taken into
    the accumulate pass and the fold read the value behind it.
    """
    code = _split(
        _stored_accumulator_body(after_marker=rewrite),
        _ACCUMULATED_RENAMES,
        _XYZ_DISJOINT,
    )
    accumulate, consume = _lane_passes(code)
    head = rewrite.partition("\n")[0]
    assert head not in accumulate
    assert accumulate.index("for tile_offset_2 in") < accumulate.index(
        "sum_2_lane_acc = sum_2_lane_acc +"
    )
    assert (
        consume.index("for tile_offset_2 in")
        < consume.index(head)
        < consume.index(".store(")
    )


def test_a_dependent_chains_consume_pass_restarts_an_accumulator_a_lane_varying_loop_rewrites() -> (
    None
):
    """The chained split's final pass re-runs ``acc = 0`` per lane too.

    Two chained reductions (the second's input reads the first's result) are
    followed by a loop that accumulates ``acc`` from the second result under
    ``v_8`` and a store of ``acc``: the initializer belongs to the final lane
    pass, not to the invariant tail between the passes.
    """
    text = (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "p = (x.iterator + cutlass.Int32(indices)).load()\n"
        f"m1 = cutlass.Float32({_masked_marker('p')})\n"
        "q = p * m1\n"
        f"m2 = cutlass.Float32({_masked_marker('q')})\n"
        "acc = cutlass.Float32(0.0)\n"
        "for k in range(cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)):\n"
        "    acc_copy = acc\n"
        "    v_8 = acc_copy + q * m2\n"
        "(y.iterator + cutlass.Int32(indices)).store(cutlass.Float32(acc))\n"
    )
    code = _split(text, {"v_8": "acc", "acc": "acc"}, _XYZ_DISJOINT)
    passes = _lane_passes(code)
    assert len(passes) == 3
    init = "acc = cutlass.Float32(0.0)"
    assert code.count(init) == 1
    consume = passes[-1]
    assert (
        consume.index(init) < consume.index("for k in range") < consume.index(".store(")
    )


_STORED_TOTAL = (
    "(tot.iterator + cutlass.Int32(tile_offset) * cutlass.Int32(tot.layout.stride[0]))"
    ".store(cutlass.Float32(sum_2))\n"
)
_SCALED_ROW = (
    "(out.iterator + cutlass.Int32(indices) * cutlass.Int32(out.layout.stride[0]))"
    ".store(cutlass.Float32(_mask_to * sum_2))\n"
)
_STORED_CARRY_COPY = (
    "(tot.iterator + cutlass.Int32(tile_offset) * cutlass.Int32(tot.layout.stride[0]))"
    ".store(cutlass.Float32(mean_acc_copy))\n"
)
# The tensors the restored bodies store to are not the one they load.
_RESTORED_DISJOINT = {frozenset({"x", "tot"}), frozenset({"x", "out"})}
_CARRY_COPIES = {
    0: "",
    1: "mean_acc_copy = mean_acc\n",
    2: "mean_acc_copy = mean_acc\nmean_acc_copy_0 = mean_acc_copy\n",
}


def _restored_rows_body(tail: str, *, carry_copies: int = 0) -> str:
    """``_accumulated_rows_body`` under a collective end, ``tail`` for its carry update.

    The jagged loop's collective trip count keeps the two-pass split off this
    body, so its marker is restored per lane.  ``carry_copies`` chained phi
    copies of ``mean_acc`` (``mean_acc_copy = mean_acc``, ``mean_acc_copy_0 =
    mean_acc_copy``) precede the loop for a ``tail`` that updates the carry.
    """
    body = _accumulated_rows_body(collective_end=True)
    update = "v_9 = mean_acc_copy + sum_2\n"
    copy = "mean_acc_copy = mean_acc\n"
    assert body.endswith(update) and body.count(copy) == 1
    return body[: -len(update)].replace(copy, _CARRY_COPIES[carry_copies]) + tail


@pytest.mark.parametrize("carry_copies", [1, 2])
def test_a_strided_restore_is_kept_for_a_lane_carry_update(carry_copies: int) -> None:
    """A lane's share is complete through a carry: the update off the last of the chained phi copies runs per lane and the marker stays restored (jagged_layer_norm's schedule)."""
    copy = "mean_acc_copy" if carry_copies == 1 else "mean_acc_copy_0"
    code = _split(
        _restored_rows_body(f"v_9 = {copy} + sum_2\n", carry_copies=carry_copies),
        _ROW_SUMS_RENAMES,
        _RESTORED_DISJOINT,
    )
    (body,) = _lane_passes(code)
    assert "_lane_total" not in code and "_lane_acc" not in code
    assert body.index("_cute_grouped_reduce_shared_two_stage(_mask_to,") < body.index(
        f"v_9 = {copy} + sum_2"
    )


def test_a_stored_strided_reduction_is_totalled_across_the_lanes() -> None:
    """A store of the result would read one lane's share: the lane loop stays whole, each lane's share is folded into a total the loop carries, and the store runs once after the loop on the total, owner-guarded."""
    code = _split(
        _restored_rows_body(_STORED_TOTAL), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT
    )
    (body,) = _lane_passes(code)
    assert "_lane_acc" not in code
    assert code.index("sum_2_lane_total = cutlass.Float32(0)") < code.index(
        "for lane in range(2):"
    )
    assert body.index("for tile_offset_2 in") < body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    assert "sum_2_lane_share = _mask_to_reduced" in body
    assert (
        "sum_2_lane_total = sum_2_lane_total + cutlass.Float32(sum_2_lane_share)"
        in body
    )
    assert "tot.iterator" not in body
    after = code[code.index("sum_2 = cutlass.Float32(sum_2_lane_total)") :]
    assert after.index(
        "if cutlass.Int32(cute.arch.thread_idx()[0]) % 256 < 1:"
    ) < after.index("tot.iterator")


@pytest.mark.parametrize(
    ("tail", "carry_copies"),
    [
        (_SCALED_ROW, 0),
        ("v_9 = mean_acc_copy + sum_2\n" + _STORED_TOTAL, 1),
        ("v_9 = mean_acc_copy + sum_2\n" + _STORED_CARRY_COPY, 1),
    ],
    ids=["scaled_per_lane", "carry_and_store", "second_reader_of_the_carry"],
)
def test_a_strided_reduction_without_a_whole_loop_lowering_is_declined(
    tail: str, carry_copies: int
) -> None:
    """A result consumed inside the lanes (scaled back into the per-lane values), beside a carry update, or whose carry is read elsewhere has neither a complete restore nor a lane-invariant tail: the config is declined."""
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            _restored_rows_body(tail, carry_copies=carry_copies),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        )


def test_the_restore_declines_a_stored_strided_reduction() -> None:
    loop = lanes._create_lane_loop("lane", 2, _body(_restored_rows_body(_STORED_TOTAL)))
    (index,) = [
        i
        for i, stmt in enumerate(loop.body)
        if lanes._is_lane_reduce_marker_assign(stmt) is not None
    ]
    marker = lanes._is_lane_reduce_marker_assign(loop.body[index])
    assert marker is not None and marker.strided_restore
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        lanes._restore_per_lane_markers(loop, [(index, marker)], _ROW_SUMS_RENAMES)


_MAX_CARRY_UPDATES = {
    "cute_math_max_under_casts": (
        "v_9 = cute.math.max(cutlass.Float32(mean_acc_copy), "
        "cutlass.Float32(amax_2), propagate_nan=True)\n"
    ),
    "ternary": "v_9 = mean_acc_copy if mean_acc_copy > amax_2 else amax_2\n",
    "ternary_swapped": "v_9 = amax_2 if mean_acc_copy < amax_2 else mean_acc_copy\n",
    "builtin_max": "v_9 = max(mean_acc_copy, amax_2)\n",
}
_MIN_OF_A_MAX_MARKER = {
    "ternary_min": "v_9 = mean_acc_copy if mean_acc_copy < amax_2 else amax_2\n",
    "cute_math_min": (
        "v_9 = cute.math.min(cutlass.Float32(mean_acc_copy), "
        "cutlass.Float32(amax_2), propagate_nan=True)\n"
    ),
}
_GUARDED_ATOMIC_ADD = (
    "if cute.arch.thread_idx()[0] == 0:\n"
    "    cute.arch.atomic_add((tot.iterator + cute.crd2idx((tile_offset,), "
    "tot.layout)).llvm_ptr, val=cutlass.Float32(sum_2), sem='relaxed')\n"
    "else:\n"
    "    pass\n"
)
_SUM_MARKER_LINE = f"sum_2 = cutlass.Float32({_masked_marker('_mask_to')})\n"
_STORED_AMAX = (
    "(tot2.iterator + cutlass.Int32(tile_offset) * cutlass.Int32(tot2.layout.stride[0]))"
    ".store(cutlass.Float32(amax_3))\n"
)
_TWO_TENSORS_DISJOINT = _RESTORED_DISJOINT | {
    frozenset({"x", "tot2"}),
    frozenset({"tot", "tot2"}),
}


def _max_marker(value: str) -> str:
    return lanes._lane_reduce_marker_expr(
        value,
        "max",
        "cutlass.Float32(float('-inf'))",
        256,
        group_pre=1,
        group_span=256,
        group_lane_expr=_TID,
        group_count=1,
        owner_lane="lane",
        strided_restore=True,
    )


def _restored_rows_max_body(tail: str) -> str:
    """``_restored_rows_body`` with a max marker ``amax_2`` in place of the sum, one phi copy of the carry."""
    body = _restored_rows_body("", carry_copies=1)
    assert body.count(_SUM_MARKER_LINE) == 1
    return (
        body.replace(
            _SUM_MARKER_LINE,
            f"amax_2 = cutlass.Float32({_max_marker('_mask_to')})\n",
        )
        + tail
    )


def _assert_restored_before(code: str, update: str) -> None:
    (body,) = _lane_passes(code)
    assert "_lane_total" not in code
    assert body.index("_cute_grouped_reduce_shared_two_stage(_mask_to,") < body.index(
        update.strip()
    )


@pytest.mark.parametrize(
    "update", list(_MAX_CARRY_UPDATES.values()), ids=list(_MAX_CARRY_UPDATES)
)
def test_a_strided_restore_is_kept_for_a_running_max_carry(update: str) -> None:
    """The max over the lanes of the lanes' maxes is the max over the tile: a running-max carry (``cute.math.max`` under casts, the ternary, ``max()``) keeps the per-lane restore."""
    _assert_restored_before(
        _split(_restored_rows_max_body(update), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT),
        update,
    )


@pytest.mark.parametrize(
    "update", list(_MIN_OF_A_MAX_MARKER.values()), ids=list(_MIN_OF_A_MAX_MARKER)
)
def test_a_min_carry_of_a_max_reduction_is_declined(update: str) -> None:
    """A carry folding the other extremum would keep one lane's max per lane: it is no carry of this marker, and the tail touching the carry has no re-reduce either."""
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(_restored_rows_max_body(update), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT)


@pytest.mark.parametrize(
    "update",
    [
        "v_9 = mean_acc_copy + cutlass.Float32(sum_2)\n",
        "v_9 = cutlass.Float32(mean_acc_copy + sum_2)\n",
    ],
    ids=["cast_operand", "cast_sum"],
)
def test_a_strided_restore_is_kept_for_a_carry_update_under_a_cast(
    update: str,
) -> None:
    """A cast to the accumulator dtype around an operand or the sum is the render's spelling of the same carry update."""
    _assert_restored_before(
        _split(
            _restored_rows_body(update, carry_copies=1),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        ),
        update,
    )


def test_a_carry_update_under_a_narrowing_cast_is_declined() -> None:
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            _restored_rows_body(
                "v_9 = mean_acc_copy + cutlass.BFloat16(sum_2)\n", carry_copies=1
            ),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        )


def test_a_strided_restore_is_kept_for_a_carry_update_in_an_if_join() -> None:
    """An if-join that adds the share under a lane-invariant condition and keeps the carry otherwise adds every lane's share or none: still the carry's reduction over the tile."""
    update = (
        "if cond_u:\n    v_9 = mean_acc_copy + sum_2\nelse:\n    v_9 = mean_acc_copy\n"
    )
    code = _split(
        _restored_rows_body(update, carry_copies=1),
        _ROW_SUMS_RENAMES,
        _RESTORED_DISJOINT,
    )
    (body,) = _lane_passes(code)
    assert "_lane_total" not in code
    assert body.index("_cute_grouped_reduce_shared_two_stage(_mask_to,") < body.index(
        "if cond_u:"
    )


@pytest.mark.parametrize(
    "update",
    [
        "if mask:\n    v_9 = mean_acc_copy + sum_2\nelse:\n    v_9 = mean_acc_copy\n",
        (
            "if mean_acc_copy > 0:\n    v_9 = mean_acc_copy + sum_2\n"
            "else:\n    v_9 = mean_acc_copy\n"
        ),
        (
            "if cond_u:\n    v_9 = mean_acc_copy + sum_2\n"
            "else:\n    v_9 = cutlass.Float32(0.0)\n"
        ),
    ],
    ids=["lane_varying_condition", "condition_reads_the_carry", "else_resets"],
)
def test_an_if_join_that_is_not_the_same_carry_in_every_lane_is_declined(
    update: str,
) -> None:
    """A condition varying with the lane or reading the carry, or a branch that resets the carry, would leave some lanes' shares out."""
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="no proved complete per-lane restore|loop-carried value",
    ):
        _split(
            _restored_rows_body(update, carry_copies=1),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        )


def test_an_atomic_add_of_a_strided_reduction_runs_once_on_the_lanes_total() -> None:
    """The emitted atomic spells its operand by keyword (``val=``); it is a lane-invariant consumer of the total, so the shares are totalled and the one guarded atomic runs after the loop."""
    code = _split(
        _restored_rows_body(_GUARDED_ATOMIC_ADD), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT
    )
    (body,) = _lane_passes(code)
    assert (
        "sum_2_lane_total = sum_2_lane_total + cutlass.Float32(sum_2_lane_share)"
        in body
    )
    assert "atomic_add" not in body
    after = code[code.index("sum_2 = cutlass.Float32(sum_2_lane_total)") :]
    assert after.count("cute.arch.atomic_add(") == 1
    assert after.index("if cute.arch.thread_idx()[0] == 0:") < after.index(
        "cute.arch.atomic_add("
    )


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("cute.arch.atomic_add(p, val=v, sem='relaxed')", ("p", "v")),
        ("cute.arch.atomic_add(p, v)", ("p", "v")),
        ("cute.arch.atomic_cas(p, cmp=a, val=b, sem='relaxed')", None),
    ],
    ids=["keyword", "positional", "compare_and_swap"],
)
def test_the_atomic_operand_parser_reads_the_keyword_form(
    call: str, expected: tuple[str, str] | None
) -> None:
    node = ast.parse(call, mode="eval").body
    assert isinstance(node, ast.Call)
    parts = lanes._atomic_pointer_and_value(node)
    if expected is None:
        assert parts is None
    else:
        assert parts is not None
        assert tuple(ast.unparse(part) for part in parts) == expected


def test_a_second_marker_input_produced_between_the_markers_joins_the_prefix() -> None:
    """Two stored reductions of one accumulator under dynamic shapes render the second's masked input between the markers; it reads prefix values only, so it moves before the first marker and both shares are totalled."""
    body = _restored_rows_body(_STORED_TOTAL).replace(
        _SUM_MARKER_LINE,
        _SUM_MARKER_LINE
        + _STORED_TOTAL
        + "_mask_to2 = cutlass.Float32(row_sums) if mask else "
        "cutlass.Float32(float('-inf'))\n"
        + f"amax_3 = cutlass.Float32({_max_marker('_mask_to2')})\n",
        1,
    )
    body = body[: -len(_STORED_TOTAL)] + _STORED_AMAX
    code = _split(body, _ROW_SUMS_RENAMES, _TWO_TENSORS_DISJOINT)
    (lane_body,) = _lane_passes(code)
    assert (
        code.index("sum_2_lane_total = cutlass.Float32(0)")
        < code.index("amax_3_lane_total = cutlass.Float32(float('-inf'))")
        < code.index("for lane in range(2):")
    )
    assert lane_body.index("_mask_to2 = ") < lane_body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    assert "tot.iterator" not in lane_body and "tot2.iterator" not in lane_body
    after = code[code.index("amax_3 = cutlass.Float32(amax_3_lane_total)") :]
    assert after.count("% 256 < 1") == 2
    assert after.index("tot.iterator") < after.index("tot2.iterator")


def test_a_second_marker_input_reading_the_first_result_stays_declined() -> None:
    body = _restored_rows_body(_STORED_TOTAL).replace(
        _SUM_MARKER_LINE,
        _SUM_MARKER_LINE
        + "_mask_to2 = _mask_to * sum_2\n"
        + f"sum_3 = cutlass.Float32({_masked_marker('_mask_to2')})\n",
        1,
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            body + _STORED_AMAX.replace("amax_3", "sum_3"),
            _ROW_SUMS_RENAMES,
            _TWO_TENSORS_DISJOINT,
        )


def _online_softmax_mm(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Attention-style online softmax: two carried row vectors (``m_i``,
    ``l_i``) plus a matmul accumulator, with the key tile reduced by
    ``amax`` / ``sum`` and contracted by the ``p @ v`` product."""
    b, m_dim, d = q.size()
    n = k.size(1)
    out = torch.empty_like(q)
    for tile_b, tile_m in hl.tile([b, m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, d], dtype=torch.float32)
        for tile_n in hl.tile(n):
            qk = torch.bmm(q[tile_b, tile_m, :], k[tile_b, tile_n, :].transpose(1, 2))
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            p = torch.exp2(qk - m_ij[:, :, None])
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            acc = torch.baddbmm(acc, p.to(v.dtype), v[tile_b, tile_n, :])
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out


def _online_softmax_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    scores = torch.bmm(q.float(), k.float().transpose(1, 2))
    # ``exp2(x) == exp(x * ln 2)``
    probabilities = torch.softmax(scores * math.log(2.0), dim=-1)
    return torch.bmm(probabilities, v.float()).to(q.dtype)


def _online_softmax_kernel() -> helion.Kernel:
    return helion.kernel(
        _online_softmax_mm,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


# ``block_sizes=[1, 16, 16]`` is the default config of the attention examples:
# the head_dim matmul reduction owns 64 threads on axis 0 (a persistent
# reduction state), ``tile_m`` 2 threads on axis 1 and ``tile_n`` 4 threads on
# axis 2 times a 4-iteration per-thread lane loop. The key-tile reductions are
# therefore lane-strided AND have a live thread axis whose lanes sit 128
# threads apart.
_ONLINE_SOFTMAX_BLOCKS = [1, 16, 16]


def _emitted_tile_reduction_markers(
    monkeypatch: pytest.MonkeyPatch, config: helion.Config
) -> tuple[str, list[lanes._LaneReduceMarker]]:
    """CPU-only codegen of the online-softmax kernel; capture every lane
    reduction marker ``BlockReductionStrategy`` emits and return the code."""
    captured: list[str] = []
    original = reductions.BlockReductionStrategy._lane_loop_marker_expr

    def capture(
        self: reductions.BlockReductionStrategy, *args: object, **kwargs: object
    ) -> str:
        expression = original(self, *args, **kwargs)  # pyrefly: ignore [bad-argument-type]
        captured.append(expression)
        return expression

    monkeypatch.setattr(
        reductions.BlockReductionStrategy, "_lane_loop_marker_expr", capture
    )
    q = torch.empty((4, 256, 64), dtype=torch.bfloat16)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(_online_softmax_kernel(), (q, q, q))
        code = bound.to_code(config)
    markers = []
    for expression in captured:
        marker = lanes._is_lane_reduce_marker_assign(
            ast.parse(f"reduced = {expression}").body[0]
        )
        assert marker is not None
        markers.append(marker)
    return code, markers


@skipUnlessBackends(["cute"])
def test_lane_looped_tile_reduction_marker_uses_full_launch_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane-looped tile block with a live thread axis must emit the two-pass
    marker (not a thread-only strided reduce), and the marker's physical group
    must come from the full launch layout: ``pre`` counts the 64 persistent
    reduction threads on axis 0 times the 2 ``tile_m`` threads on axis 1, so
    the finalize folds the 4 ``tile_n`` threads that are 128 lanes apart."""
    code, markers = _emitted_tile_reduction_markers(
        monkeypatch, helion.Config(block_sizes=_ONLINE_SOFTMAX_BLOCKS)
    )
    assert [marker.reduction_type for marker in markers] == ["max", "sum"]
    owners = {marker.owner_lane for marker in markers}
    assert len(owners) == 1 and next(iter(owners))
    for marker in markers:
        assert (
            marker.threads_in_group,
            marker.group_pre,
            marker.group_span,
            marker.group_count,
        ) == (4, 128, 512, 1)
        # Linear thread index over all three launch axes: axis 1 strides by
        # the 64 persistent threads, axis 2 by 64 * 2.
        assert "cute.arch.thread_idx()[0]" in marker.group_lane_expr
        assert "thread_idx()[1])) * cutlass.Int32(64)" in marker.group_lane_expr
        assert "thread_idx()[2])) * cutlass.Int32(128)" in marker.group_lane_expr
    # The post-pass finalized both markers with the cross-warp grouped reduce
    # of that physical group; no thread-only partial reduce survives.
    assert "_helion_lane_reduce" not in code
    assert code.count("pre=128, group_span=512, group_count=1") >= 2
    assert "_cute_grouped_reduce_warp(" not in code


@skipUnlessBackends(["cute"])
def test_vectorized_lane_tile_reduction_declines_before_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cute_vector_widths > 1`` on the reduced key tile nests the per-element
    body in a constexpr vector lane. The ``amax`` / ``sum`` keep the legacy
    single-pass flow (V-folded by ``hoist_warp_reduce``), but the owned
    ``p @ v`` contribution marker sits inside the vector lane where it used to
    be silently restored to its raw per-lane input (25% wrong attention
    output); it must decline instead."""
    with pytest.raises(helion.exc.BackendUnsupported, match="constexpr vector lane"):
        _emitted_tile_reduction_markers(
            monkeypatch,
            helion.Config(
                block_sizes=_ONLINE_SOFTMAX_BLOCKS, cute_vector_widths=[1, 1, 1, 4]
            ),
        )


def test_constexpr_vector_lane_is_not_a_serial_interchange_loop() -> None:
    """The interchange pass must not treat a ``cutlass.range_constexpr``
    unroll as a serial device loop: an owned marker nested in it is left to
    the lane split, which folds the vector lane into its accumulate pass
    (all ``lanes x V`` elements, one warp combine, the carried update once)
    instead of restoring the raw input."""
    vector_loop = ast.parse(
        f"for vec_lane in cutlass.range_constexpr(4):\n"
        f"    partial = lane + vec_lane\n"
        f"    reduced = {_marker('lane')}\n"
        f"    carried = carried + reduced"
    ).body[0]
    _stamped([vector_loop])
    assert not lanes._is_serial_for(vector_loop)
    loop = lanes._create_lane_loop("lane", 2, [vector_loop])
    unchanged = lanes.interchange_lane_outside_serial_reductions([loop])
    assert "_helion_lane_reduce" in _source(unchanged)
    lowered = lanes.split_lane_loop_reductions(unchanged)
    assert [ast.unparse(stmt) for stmt in lowered] == [
        "reduced_lane_acc = cutlass.Float32(0)",
        (
            "for lane in range(2):\n"
            "    for vec_lane in cutlass.range_constexpr(4):\n"
            "        partial = lane + vec_lane\n"
            "        reduced_lane_acc = reduced_lane_acc + cutlass.Float32(partial)"
        ),
        "reduced = cute.arch.warp_reduction_sum(reduced_lane_acc, threads_in_group=32)",
        "carried = carried + reduced",
    ]


def _vector_lane_marker(value: str) -> str:
    return lanes._lane_reduce_marker_expr(
        value, "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )


def test_vector_lane_markers_fold_into_the_lane_split() -> None:
    """A lane loop whose markers sit in its constexpr vector lane (the
    per-element body of a ``cute_vector_widths > 1`` tile axis below the
    vector load of each lane step, named ``vec_<lane>`` by the strategies) is
    split as the flat lane loop over the lane prefix plus the vector elements,
    and every emitted pass re-nests the vector lane: welford's chunk sum and
    its dependent centered sum reduce over all ``lanes x V`` elements, the
    prefix stays per lane step and the lane-invariant carried updates run
    once.  The fold admits no memory write, so the passes are the accumulate
    passes alone and the results leave the loop through the carries."""
    text = (
        "base = lane * 4\n"
        "packet = base + 1\n"
        "for vec_lane in cutlass.range_constexpr(4):\n"
        "    count_copy = count\n"
        "    m2_copy = m2_total\n"
        "    element = packet + vec_lane\n"
        f"    total = {_vector_lane_marker('element')}\n"
        "    mean = total / 8\n"
        "    centered = element - mean\n"
        "    squared = centered * centered\n"
        f"    m2 = {_vector_lane_marker('squared')}\n"
        "    next_count = count_copy + total\n"
        "    next_m2_total = m2_copy + m2\n"
    )
    renames = {
        "next_count": "count",
        "count": "count",
        "next_m2_total": "m2_total",
        "m2_total": "m2_total",
    }
    lowered = lanes.split_lane_loop_reductions(
        [
            *_body("count = 0.0\nm2_total = 0.0"),
            lanes._create_lane_loop("lane", 2, _stamped(_body(text))),
            *_body("late_out.store(count)\n(late_out.iterator + 1).store(m2_total)"),
        ],
        rename_groups=renames,
    )
    code = _source(lowered)
    assert "_helion_lane_reduce" not in code
    assert "vec_lane = lane" not in code
    lane_loops = [stmt for stmt in lowered if isinstance(stmt, ast.For)]
    # One accumulate pass per dependency level; nothing is left for a
    # consume pass.
    assert len(lane_loops) == 2
    for lane_loop in lane_loops:
        assert ast.unparse(lane_loop.body[0]) == "base = lane * 4"
        vector_lane = lane_loop.body[-1]
        assert isinstance(vector_lane, ast.For)
        assert ast.unparse(vector_lane.target) == "vec_lane"
        assert ast.unparse(vector_lane.iter) == "cutlass.range_constexpr(4)"
    assert "total_lane_acc = total_lane_acc + cutlass.Float32(element)" in code
    assert "m2_lane_acc = m2_lane_acc + cutlass.Float32(squared)" in code
    assert code.count("next_count = count_copy + total") == 1
    assert code.count("next_m2_total = m2_copy + m2") == 1
    values, calls = _execute_scalars(lowered, renames)
    elements = [lane * 4 + 1 + vec for lane in range(2) for vec in range(4)]
    total = float(sum(elements))
    mean = total / 8
    m2 = sum((element - mean) ** 2 for element in elements)
    assert values["late_out"] == {0: total, 1: m2}
    assert calls == 0


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "base = lane * 4\n"
            "for vec_lane in cutlass.range_constexpr(4):\n"
            f"    total = {_vector_lane_marker('base')}\n"
            "    scaled = base * total\n",
            id="input-independent-of-the-element",
        ),
        pytest.param(
            "base = lane * 4\n"
            "for vec_lane in cutlass.range_constexpr(4):\n"
            "    element = base + vec_lane\n"
            f"    total = {_vector_lane_marker('element')}\n"
            "    flushed = element - total\n"
            "(out.iterator + lane).store(flushed)\n",
            id="store-flush-after-the-vector-lane",
        ),
        pytest.param(
            "base = lane * 4\n"
            "for vec_lane in cutlass.range_constexpr(4):\n"
            "    element = _cute_grouped_reduce_shared_two_stage("
            "cutlass.Float32(base + vec_lane), 'sum', cutlass.Float32(0), "
            "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
            "pre=1, group_span=64, group_count=1)\n"
            f"    total = {_vector_lane_marker('element')}\n"
            "    scaled = element * total\n",
            id="collective-in-the-vector-lane",
        ),
        pytest.param(
            # Lane 1 loads the packet lane 0 stores; the rolled loop orders
            # lane 0's store first, a fold would run every lane's loads first.
            "base = lane * 4\n"
            "for vec_lane in cutlass.range_constexpr(4):\n"
            "    element = (x.iterator + base + vec_lane).load()\n"
            "    previous = (x.iterator + base + vec_lane - 4).load()\n"
            "    summed = element + previous\n"
            f"    total = {_vector_lane_marker('summed')}\n"
            "    (x.iterator + base + vec_lane).store(element * total)\n",
            id="aliasing-store-in-the-vector-lane",
        ),
        pytest.param(
            "base = lane * 4\n"
            "for vec_lane in cutlass.range_constexpr(4):\n"
            "    element = base + vec_lane\n"
            f"    total = {_vector_lane_marker('element')}\n"
            "    packet[vec_lane] = element - total\n",
            id="fragment-write-in-the-vector-lane",
        ),
        pytest.param(
            "base = lane * 4\n"
            "for step in cutlass.range_constexpr(4):\n"
            "    element = base + step\n"
            f"    total = {_vector_lane_marker('element')}\n"
            "    scaled = element * total\n",
            id="foreign-constexpr-loop",
        ),
    ],
)
def test_vector_lane_markers_outside_the_fold_shape_decline(text: str) -> None:
    """The fold admits only the per-element shape it proves: every marker
    input depends on the vector element (the original body reduces a
    per-lane-step value V times), nothing follows the vector lane (a vector
    store flush reads values of every element), no collective sits in it (the
    passes re-run the element producers), nothing in the loop writes memory
    (the alias proof runs on the flat body, where the ``vec_lane = lane``
    sentinel makes it read a packet-shifted address in the wrong iteration
    space: the scalar analog of the aliasing shape declines as a reordered
    aliasing write) and the constexpr loop is this lane's own ``vec_lane``
    wrapper (any other constexpr loop is a serial unroll)."""
    loop = lanes._create_lane_loop("lane", 4, _stamped(_body(text)))
    with pytest.raises(helion.exc.BackendUnsupported, match="constexpr vector lane"):
        lanes.split_lane_loop_reductions([loop])


def test_owned_marker_in_serial_loop_without_store_declines() -> None:
    """An owned marker inside a genuine serial loop that no broadcast store
    consumes has no interchange; its raw per-lane input is not a complete
    reduction, so the pass declines rather than restoring it."""
    serial_loop = ast.parse(
        f"for mb in range(0, 64, 8):\n"
        f"    partial = lane + mb\n"
        f"    reduced = {_marker('lane')}\n"
        f"    carried = carried + reduced"
    ).body[0]
    assert lanes._is_serial_for(serial_loop)
    loop = lanes._create_lane_loop("lane", 8, [serial_loop])
    with pytest.raises(helion.exc.BackendUnsupported, match="no proved complete"):
        lanes.interchange_lane_outside_serial_reductions([loop])
    legacy_loop = lanes._create_lane_loop(
        "lane",
        8,
        [
            ast.parse(
                f"for mb in range(0, 64, 8):\n"
                f"    partial = lane + mb\n"
                f"    reduced = {_marker(None)}\n"
                f"    carried = carried + reduced"
            ).body[0]
        ],
    )
    restored = _source(lanes.interchange_lane_outside_serial_reductions([legacy_loop]))
    assert "reduced = partial" in restored and "_helion_lane_reduce" not in restored


@onlyBackends(["cute"])
class TestLaneLoopedTileReductionNumerics(TestCase):
    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        return tuple(  # pyrefly: ignore [bad-return]
            torch.randn(4, 256, 64, device=DEVICE, dtype=torch.bfloat16)
            for _ in range(3)
        )

    def test_online_softmax_default_attention_config(self) -> None:
        args = self._inputs()
        code, out = code_and_output(
            _online_softmax_kernel(), args, block_sizes=_ONLINE_SOFTMAX_BLOCKS
        )
        torch.testing.assert_close(
            out.float(), _online_softmax_reference(*args).float(), rtol=1e-1, atol=1e-1
        )
        self.assertNotIn("_helion_lane_reduce", code)
        self.assertIn("pre=128, group_span=512, group_count=1", code)

    def test_welford_vector_lane_fold_with_padded_columns(self) -> None:
        """welford's scalar-stats sibling layout nests the chunk count, sum
        and dependent centered sum in the vector lane of its 128-wide column
        tile (one thread, 32 packets of 4 elements). The owned reduction
        flattens both coordinates and folds all 128 elements of a chunk; the
        raw per-element restore
        this shape used to get (one element per Welford merge) only matched
        for fully valid rows and divided by a zero count on the 48 padded
        columns of an 80-column row."""
        torch.manual_seed(0)
        columns = 80
        weight = torch.rand(columns, device=DEVICE, dtype=torch.float32)
        bias = torch.rand(columns, device=DEVICE, dtype=torch.float32)
        x = torch.randn(17, columns, device=DEVICE, dtype=torch.float32)
        kernel = helion.kernel(
            welford.fn, backend="cute", static_shapes=True, autotune_effort="none"
        )
        bound = kernel.bind((weight, bias, x))
        config = _config("scalar_stats")
        code = bound.to_code(config)
        out = bound.compile_config(config)(weight, bias, x)
        expected = torch.nn.functional.layer_norm(x, (columns,), weight, bias, eps=1e-5)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        self.assertNotIn("_helion_lane_reduce", code)
        self.assertIn("for lane_1 in range(128):", code)
        self.assertIn("lane_1 // 4", code)
        self.assertIn("lane_1 % 4", code)
        self.assertIn("sum_2_lane_acc = sum_2_lane_acc + cutlass.Float32(chunk)", code)
        self.assertIn(
            "m2_c_lane_acc = m2_c_lane_acc + cutlass.Float32(_mask_to_2)", code
        )

    def test_online_softmax_vectorized_key_tile_declines(self) -> None:
        """The GPU bind takes the same decline the CPU-only
        ``test_vectorized_lane_tile_reduction_declines_before_emission`` pins:
        the owned ``p @ v`` contribution marker sits inside the constexpr
        vector lane, which has no proved two-pass lowering."""
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "constexpr vector lane"
        ):
            code_and_output(
                _online_softmax_kernel(),
                self._inputs(),
                block_sizes=_ONLINE_SOFTMAX_BLOCKS,
                cute_vector_widths=[1, 1, 1, 4],
            )


@pytest.mark.parametrize("observed", [False, True])
def test_owned_marker_input_updated_by_aliased_serial_loop_splits(
    observed: bool,
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "row_sums", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    # The inner serial loop updates ``row_sums`` through the SSA alias
    # ``next_row_sums`` that only the final rename pass restores
    # (jagged_layer_norm's per-feature row sums).
    text = (
        "total_copy = total\n"
        "row_sums = 0.0\n"
        "for step in range(3):\n"
        "    row_sums_copy = row_sums\n"
        "    next_row_sums = row_sums_copy + (lane + 1)\n"
        f"reduced = {marker}\n"
        "next_total = total_copy + reduced\n"
    )
    if observed:
        text += "(out.iterator + lane).store(row_sums + reduced)\n"
    renames = {
        "next_row_sums": "row_sums",
        "row_sums": "row_sums",
        "next_total": "total",
        "total": "total",
    }

    def make() -> list[ast.AST]:
        return [
            *_body("total = 6.0"),
            lanes._create_lane_loop("lane", 8, _body(text)),
            *_body("late_out.store(total)"),
        ]

    # Without the recorded alias the update loop is an unproved producer.
    with pytest.raises(
        helion.exc.BackendUnsupported, match="complete per-lane restore"
    ):
        lanes.split_lane_loop_reductions(make())
    lowered = lanes.split_lane_loop_reductions(make(), rename_groups=renames)
    code = _source(lowered)
    assert "_helion_lane_reduce" not in code
    assert code.count("next_total = total_copy + reduced") == 1
    values, calls = _execute_scalars(lowered, renames)
    # Each lane accumulates 3 * (lane + 1); the complete lane sum is 108.
    assert values["late_out"] == {0: 114.0} and calls == 0
    if observed:
        # The per-lane row sum stays lane-varying after the alias is restored.
        assert values["out"] == {lane: float(3 * (lane + 1) + 108) for lane in range(8)}


def test_dependent_markers_with_unduplicatable_producer_decline() -> None:
    first = lanes._lane_reduce_marker_expr(
        "acc", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    second = lanes._lane_reduce_marker_expr(
        "centered", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    # ``acc`` is re-derived by a collective under an SSA alias and feeds a
    # chain of dependent reductions whose result updates a carried scalar, so
    # neither the stash nor the dependent split may re-run the producer.
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            "total_copy = total\n"
            "acc = 0.0\n"
            "for step in range(2):\n"
            "    acc_copy = acc\n"
            "    partial = _cute_grouped_reduce_shared_two_stage("
            "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
            "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
            "pre=1, group_span=64, group_count=1)\n"
            "    next_acc = acc_copy + partial\n"
            f"reduced = {first}\n"
            "centered = acc - reduced\n"
            f"reduced_again = {second}\n"
            "next_total = total_copy + reduced_again\n"
            "(out.iterator + lane).store(centered + reduced_again)"
        ),
    )
    renames = {
        "next_acc": "acc",
        "acc": "acc",
        "next_total": "total",
        "total": "total",
    }
    with pytest.raises(
        helion.exc.BackendUnsupported, match="complete per-lane restore"
    ):
        lanes.split_lane_loop_reductions(
            [*_body("total = 6.0"), loop, *_body("late_out.store(total)")],
            rename_groups=renames,
        )


@pytest.mark.parametrize("extent", [8, 512])
def test_register_stash_declines_beyond_its_lane_extent_limit(extent: int) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "acc", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    # matmul_layernorm's shape: a collective K loop re-derives the per-lane
    # ``acc`` that the reduction over the lane axis consumes.
    loop = lanes._create_lane_loop(
        "lane",
        extent,
        _body(
            "acc = 0.0\n"
            "for step in range(2):\n"
            "    acc_copy = acc\n"
            "    partial = _cute_grouped_reduce_shared_two_stage("
            "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
            "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
            "pre=1, group_span=64, group_count=1)\n"
            "    acc_1 = acc_copy + partial\n"
            f"reduced = {marker}\n"
            "centered = acc - reduced\n"
            "(out.iterator + lane).store(centered)"
        ),
    )
    renames = {"acc_1": "acc", "acc": "acc"}
    if extent > 256:
        # Each lane's live-out value needs one register slot per lane, so the
        # stash does not apply and no other path lowers this shape.
        with pytest.raises(
            helion.exc.BackendUnsupported, match="complete per-lane restore"
        ):
            lanes.split_lane_loop_reductions([loop], rename_groups=renames)
        return
    lowered = lanes.split_lane_loop_reductions([loop], rename_groups=renames)
    values, calls = _execute_scalars(lowered, renames)
    # The collective runs once per lane and K step, in the stash pass only.
    assert calls == 2 * extent
    reduced = 128.0 * sum(range(1, extent + 1))
    assert values["out"] == {
        lane: 128.0 * (lane + 1) - reduced for lane in range(extent)
    }


@pytest.mark.parametrize("feeds_reduction", [False, True])
def test_collective_outside_the_reduction_slice_runs_once_per_lane(
    feeds_reduction: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    collective = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
    )
    # rms_norm_bwd's resident-row seed: the grad_weight sum over the row lanes
    # is independent of the per-row thread-group mean consumed by the grad_x
    # store, so the two-pass split runs that collective once per lane in the
    # consume pass exactly as the original loop did.  Only a collective that
    # feeds the reduction AND a live lane-varying consumer would run in both
    # passes; once the register stash has declined, that shape must decline.
    if feeds_reduction:
        text = f"partial = {collective}\nreduced = {marker}\nrow_mean = partial\n"
    else:
        text = f"partial = lane + 1\nreduced = {marker}\nrow_mean = {collective}\n"
    text += "(out.iterator + lane).store(row_mean + reduced)"
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    monkeypatch.setattr(lanes, "_split_lane_loop_with_register_stash", lambda *_: None)
    if feeds_reduction:
        with pytest.raises(
            helion.exc.BackendUnsupported, match="complete per-lane restore"
        ):
            lanes.split_lane_loop_reductions([loop])
        return
    lowered = lanes.split_lane_loop_reductions([loop])
    code = _source(lowered)
    assert "_helion_lane_reduce" not in code
    assert code.count("_cute_grouped_reduce_shared_two_stage(") == 1
    # The collective follows the finalized reduction (in the consume loop).
    assert code.index("_cute_grouped_reduce_shared_two_stage(") > code.index(
        "reduced = "
    )
    values, calls = _execute_scalars(lowered)
    assert calls == 8
    assert values["out"] == {lane: float(64 * (lane + 1) + 36) for lane in range(8)}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_id_multi_axis_store(x: torch.Tensor, out: torch.Tensor):
    for row, left, right in hl.tile(x.shape):
        out[row, left.id, right.id] = x[row, left, right].sum(-1).sum(-1)
    return out


@pytest.mark.parametrize("shape", [(3, 7, 65), (5, 17, 129)])
def test_tile_id_multi_axis_keeps_nested_reduction_rejection(shape):
    x = torch.ones(shape)
    out = torch.empty((shape[0], (shape[1] + 7) // 8, (shape[2] + 127) // 128))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_tile_id_multi_axis_store, (x, out))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[4, 8, 128], num_threads=[1, 1, 32])
        with pytest.raises(helion.exc.BackendUnsupported, match="nested reduction"):
            bound.to_code(config)


@pytest.mark.parametrize("shape", [(0, 65), (3, 0)])
def test_tile_id_empty_domain_has_no_output_writes(shape):
    code, x, out = _tile_id_program(*shape, False)
    assert out.numel() == 0
    # The ordinary launcher receives an empty grid. No scalar tile mask
    # can manufacture an output when the logical domain contains no tiles.
    assert x.numel() == 0
    host = next(
        n
        for n in ast.parse(code).body
        if isinstance(n, ast.FunctionDef) and n.name == "_tile_id_reduced_store"
    )
    launch = next(
        n
        for n in ast.walk(host)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_launcher"
    )
    constants = {"_BLOCK_SIZE_0": 4, "_BLOCK_SIZE_1": 128}
    assert 0 in eval(
        compile(ast.Expression(launch.args[1]), "<empty-grid>", "eval"), constants
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("masked", [False, True])
def test_tile_id_reduction_store_native(masked):
    x = torch.arange(7 * 129, device=DEVICE, dtype=torch.float32).reshape(7, 129)
    out = torch.full((7, 2), -1.0, device=DEVICE)
    bound = _tile_id_reduced_store.bind((x, out, masked))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[4, 128], num_threads=[1, 32])
    actual = bound.compile_config(config)(x, out, masked)
    expected = torch.stack([x[:, :128].sum(-1), x[:, 128:].sum(-1)], -1)
    if masked:
        expected[1::2] = -1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tile_id_wider_sibling_keeps_original_mask_and_arbitration():
    from helion.language.memory_ops import _cute_tile_id_thread_extent_is_complete

    observed = []

    def wider(state, owner, block_id):
        axis = owner.block_thread_axes.get(block_id)
        if axis is None:
            return _cute_tile_id_thread_extent_is_complete(state, owner, block_id)
        planned = list(state.device_function.tile_strategy.thread_block_dims())
        planned[axis] = owner.thread_axis_sizes[axis] * 2
        with patch.object(
            state.device_function.tile_strategy,
            "thread_block_dims",
            return_value=tuple(planned),
        ):
            answer = _cute_tile_id_thread_extent_is_complete(state, owner, block_id)
        observed.append(answer)
        return answer

    with (
        patch(
            "helion.language.memory_ops._cute_tile_id_thread_extent_is_complete",
            side_effect=wider,
        ),
        pytest.raises(helion.exc.BackendUnsupported),
    ):
        _tile_id_program(3, 129, False)
    assert observed and not any(observed)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_id_serial_store(
    x: torch.Tensor,
    out: torch.Tensor,
    begin: hl.constexpr,
    end: hl.constexpr,
    masked: hl.constexpr,
):
    for row in hl.tile(x.size(0)):
        for col in hl.tile(begin, end):
            value = x[row, col].sum(-1)
            if masked:
                hl.store(out, [row, col.id], value, extra_mask=row.index % 2 == 0)
            else:
                out[row, col.id] = value
    return out


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("begin,end", [(1, 130), (65, 194), (129, 258)])
def test_tile_id_serial_tail_preserves_element_mask(begin, end, masked):
    # Only the first element of the final tile is live. Later serial lanes must
    # retain their original mask. Dense exact-valued inputs also expose a store
    # of one lane's partial sum in place of the complete first tile's reduction.
    x = torch.arange(3 * 513, dtype=torch.float32).reshape(3, 513) % 17 + 1
    out = torch.full((3, 5), -1.0)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_tile_id_serial_store, (x, out, begin, end, masked))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[4, 128], num_threads=[1, 32])
        code = bound.to_code(config)
        # This serial ownership path must remain byte-identical to the original
        # element-mask route; the root-grid optimization is tested separately.
        with patch(
            "helion.language.memory_ops._cute_tile_id_thread_extent_is_complete",
            return_value=False,
        ):
            original_mask_code = bound.to_code(config)
    assert code == original_mask_code
    actual, writes = _execute_tile_id_program(code, x, out, require_unique_writes=False)
    expected = out.clone()
    for start in range(begin, end, 128):
        expected[:, start // 128] = x[:, start : min(start + 128, end)].sum(-1)
    if masked:
        expected[1::2] = -1
    assert torch.equal(actual, expected)
    # Four sequential writes of the same complete first-tile value and one
    # nonempty tail write per live row, with no concurrent thread duplicates.
    assert len(writes) == (2 if masked else 3) * 5
    assert "if cutlass.Int32(cute.arch.lane_idx()) % 32 == 0:" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("masked", [False, True])
def test_tile_id_serial_tail_native(masked):
    x = torch.arange(3 * 513, device=DEVICE, dtype=torch.float32).reshape(3, 513)
    x = x % 17 + 1
    out = torch.full((3, 5), -1.0, device=DEVICE)
    bound = _tile_id_serial_store.bind((x, out, 1, 130, masked))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[4, 128], num_threads=[1, 32])
    actual = bound.compile_config(config)(x, out, 1, 130, masked)
    expected = torch.full_like(out, -1.0)
    expected[:, 0] = x[:, 1:129].sum(-1)
    expected[:, 1] = x[:, 129]
    if masked:
        expected[1::2] = -1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _bounded_store_coordinate_definitions():
    definitions = {}
    for text in (
        "for base in range(1, 130, _BLOCK_SIZE_1): pass",
        "for serial in range(4): pass",
    ):
        definitions = lanes._loop_body_scalar_definitions(
            _body(text)[0], definitions, {}
        )
    return definitions


@pytest.mark.parametrize("scale", [1, 2, 4, 32])
@pytest.mark.parametrize("comparison", ["<", "<="])
def test_bounded_store_predicate_selects_zero_without_losing_live_writes(
    scale, comparison
):
    expression = f"base + cutlass.Int32(cute.arch.thread_idx()[0]) * {scale} + serial {comparison} 130"
    predicate = ast.parse(expression, mode="eval").body
    definitions = _bounded_store_coordinate_definitions()
    assert lanes._predicate_accepts_zero_thread_axis(
        predicate, 0, {}, definitions, require_bounded=True
    )
    assert not lanes._predicate_accepts_zero_thread_axis(predicate, 0, {}, definitions)
    # Check the implication, including empty serial tails: an enabled thread
    # implies that thread zero is enabled; an empty lane stays empty.
    for base in (1, 65, 129):
        for serial in range(4):
            values = [base + thread * scale + serial for thread in range(1024)]
            live = [
                value < 130 if comparison == "<" else value <= 130 for value in values
            ]
            assert not any(live) or live[0]


@pytest.mark.parametrize(
    "expression",
    [
        "tid > 0",
        "tid >= 1 and tid < 130",
        "base - tid * 4 + serial < 130",
        "tid % 2 == 1",
        "(tid << 1) < 130",
        "unknown + tid < 130",
        "cutlass.Int32(2147483647 + tid) < 130",
        "cutlass.Int32(-2147483648 - tid) < 130",
        "cutlass.Int32(4294967296 + tid) < 130",
        "base + tid < dynamic_end",
    ],
)
def test_bounded_store_predicate_rejects_unsafe_coordinates(expression):
    definitions = _bounded_store_coordinate_definitions()
    definitions["tid"] = ast.parse(
        "cutlass.Int32(cute.arch.thread_idx()[0])", mode="eval"
    ).body
    assert not lanes._predicate_accepts_zero_thread_axis(
        ast.parse(expression, mode="eval").body,
        0,
        {"tid": frozenset({0})},
        definitions,
        require_bounded=True,
    )


def test_bounded_store_induction_facts_are_scoped_and_invalidated():
    definitions = _bounded_store_coordinate_definitions()
    expression = ast.parse("base + serial", mode="eval").body
    assert lanes._scalar_integer_bounds(expression, definitions) == (1, 132)
    inherited = dict(definitions)
    unknown = lanes._loop_body_scalar_definitions(
        _body("for base in range(dynamic_end): pass")[0], definitions, {}
    )
    assert lanes._scalar_integer_bounds(expression, unknown) is None
    assert definitions == inherited
    for mutation in ("base = load()", "base = base + 1", "if flag: base = load()"):
        changed = dict(definitions)
        lanes._update_scalar_definitions(_body(mutation)[0], changed)
        assert lanes._scalar_integer_bounds(expression, changed) is None
    marker = lanes._is_lane_reduce_marker_assign(_body(f"total = {_marker('lane')}")[0])
    assert marker is not None
    safe = lanes._StoreThreadAxes(set(), set(), {0}, set(), {0}, frozenset({0}))
    assert lanes._lane_reduce_owner_expr(marker, safe) is not None
    for axes in (
        lanes._StoreThreadAxes({0}, set(), {0}, set(), {0}, frozenset({0})),
        lanes._StoreThreadAxes(set(), set(), {0}, set(), set()),
    ):
        with pytest.raises(helion.exc.BackendUnsupported, match="thread-varying store"):
            lanes._lane_reduce_owner_expr(marker, axes)


@pytest.mark.parametrize(
    "predicate",
    [
        "cutlass.Int32(2147483647 + tid) < _BLOCK_SIZE_1",
        "base + tid < 130 and unknown + tid < _BLOCK_SIZE_1",
    ],
)
def test_reduced_axis_requires_bounded_proof_for_every_predicate(predicate):
    definitions = _bounded_store_coordinate_definitions()
    definitions["tid"] = ast.parse(
        "cutlass.Int32(cute.arch.thread_idx()[0])", mode="eval"
    ).body
    axes = lanes._store_thread_axes(
        _body(f"if {predicate}: out.store(total)")[0],
        {"tid": frozenset({0})},
        definitions,
    )
    assert axes is not None
    # Preserve the legacy redundant-axis classification while refusing its
    # weaker proof for newly enabled reduced-axis stores.
    assert axes.zero_safe_predicate == (
        {0} if predicate.startswith("cutlass.Int32") else set()
    )
    assert not axes.bounded_zero_safe_predicate
    marker = lanes._is_lane_reduce_marker_assign(_body(f"total = {_marker('lane')}")[0])
    assert marker is not None
    with pytest.raises(helion.exc.BackendUnsupported, match="thread-varying store"):
        lanes._lane_reduce_owner_expr(marker, axes)


@pytest.mark.parametrize("carry_name", ["base", "phi_base"])
@pytest.mark.parametrize(
    "header", ["for outer in range(2)", "while flag", "for outer, other in pairs"]
)
def test_bounded_store_loop_context_drops_backedge_and_else_facts(carry_name, header):
    definitions = _bounded_store_coordinate_definitions()
    definitions["alias"] = ast.parse("base + 1", mode="eval").body
    rename_groups = {"phi_base": "base"}
    loop = _body(
        f"{header}:\n    consume()\n    {carry_name} = base + 2147483647\nelse:\n    consume()"
    )[0]
    observed = []

    def observe(body, **kwargs):
        observed.append(kwargs["scalar_definitions"])
        return body

    with patch.object(lanes, "split_lane_loop_reductions", side_effect=observe):
        lanes._split_stmt_lane_reductions(
            loop, set(), set(), {}, {}, definitions, rename_groups
        )
    assert len(observed) == 2
    for context in observed:
        assert "base" not in context and "alias" not in context
    expected = (0, 1) if header == "for outer in range(2)" else None
    assert lanes._scalar_integer_bounds(ast.Name(id="outer"), observed[0]) == expected
    assert lanes._scalar_integer_bounds(ast.Name(id="outer"), observed[1]) is None
    assert "base" in definitions and "alias" in definitions
    shadow = _body("for base in range(200, 201): pass\nelse: consume()")[0]
    assert (
        lanes._scalar_integer_bounds(
            ast.Name(id="base"),
            lanes._loop_invariant_scalar_definitions(shadow, definitions, {}),
        )
        is None
    )


@pytest.mark.parametrize(
    "loop",
    [
        "for base in range(129 - 2 * tid, 130): pass",
        "for base in range(1, 130 + tid): pass",
        "for base in range(1, 130, 1 + tid): pass",
        "for base in range(start, 130): pass",
    ],
)
def test_bounded_store_induction_rejects_thread_dependent_ranges(loop):
    definitions = {
        "tid": ast.parse("cutlass.Int32(cute.arch.thread_idx()[0])", mode="eval").body,
        "start": ast.parse("129 - 2 * tid", mode="eval").body,
    }
    axes = {"tid": frozenset({0}), "start": frozenset({0})}
    definitions = lanes._loop_body_scalar_definitions(_body(loop)[0], definitions, axes)
    predicate = ast.parse("base + tid < 129", mode="eval").body
    assert not lanes._predicate_accepts_zero_thread_axis(
        predicate, 0, axes, definitions, require_bounded=True
    )
    # At each thread's first iteration this enables thread one, not zero.
    assert not 129 + 0 < 129
    assert (129 - 2) + 1 < 129


@pytest.mark.parametrize("inside", [False, True])
def test_bounded_store_canonical_alias_write_invalidates_prior_constant(inside):
    assignments = _body("x = 0\nphi_x = cutlass.Int32(2147483647)")
    body = _body(
        "partial = cutlass.Float32(lane + 1)\n"
        f"reduced = {_marker('lane')}\n"
        "if cutlass.Int32(x + cutlass.Int32(cute.arch.thread_idx()[0])) < 130:\n"
        "    out.store(reduced)"
    )
    loop = lanes._create_lane_loop("lane", 4, [*assignments, *body] if inside else body)
    with pytest.raises(helion.exc.BackendUnsupported):
        lanes.split_lane_loop_reductions(
            [loop] if inside else [*assignments, loop], rename_groups={"phi_x": "x"}
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_id_scalar_copy(x: torch.Tensor, weights: torch.Tensor):
    out = torch.empty_like(weights)
    for row, col in hl.tile(x.shape):
        out[row, col.id] = weights[row, col.id]
    return out


@pytest.mark.parametrize("complete", [False, True])
def test_tile_id_source_load_keeps_uniform_value_with_store_owner_masks(complete):
    from test.test_indexing import _execute_pointwise_thread_program

    from helion._compiler.backend_registry import repair_backend_codegen
    from helion._compiler.cute import memory_ops as cute_memory_ops
    import helion.language.memory_ops as memory_ops

    repair_backend_codegen("cute")
    x = torch.empty(3, 129)
    weights = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 11
    observed = []
    original = memory_ops._cute_combined_mask

    def observe(state, subscript, extra_mask, tensor=None, **kwargs):
        result = original(state, subscript, extra_mask, tensor, **kwargs)
        observed.append((kwargs.get("for_store", False), result))
        return result

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch.object(memory_ops, "_cute_combined_mask", observe),
        patch.object(cute_memory_ops, "_cute_combined_mask", observe),
        patch.object(
            memory_ops, "_cute_tile_id_thread_extent_is_complete", return_value=complete
        ),
    ):
        bound = _cpu_bind(_tile_id_scalar_copy, (x, weights))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[4, 128], num_threads=[1, 32])
        source = bound.to_code(config)
    reads = [mask for store, mask in observed if not store]
    writes = [mask for store, mask in observed if store]
    assert reads and writes
    assert all(mask is None or "mask_1" not in mask for mask in reads)
    assert any(("mask_1" in (mask or "")) != complete for mask in writes)
    actual, _launches = _execute_pointwise_thread_program(source, (x, weights))
    torch.testing.assert_close(actual, weights, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True)
def _group_absmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    groups = n // 128
    out = torch.empty([m, groups], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_g, tile_n in hl.tile([groups, 128], block_size=[None, 128]):
            columns = tile_g.index[:, None] * 128 + tile_n.index[None, :]
            values = x[tile_m.index[:, None, None], columns[None, :, :]].abs()
            out[tile_m, tile_g] = torch.amax(values, dim=-1)
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop_order", [[0, 1], [1, 0]])
def test_reduced_device_loop_lane_nests_innermost(loop_order: list[int]) -> None:
    """Both blocks of the inner loop run as serial lanes on one thread.  The
    lane of the reduced ``tile_n`` nests inside the ``tile_g`` lane for either
    ``loop_order``: the lane reduction finalizes in its innermost owner."""
    x = torch.randn(4, 2048, device=DEVICE)
    code, out = code_and_output(
        _group_absmax, (x,), block_sizes=[16], loop_orders=[loop_order]
    )
    torch.testing.assert_close(out, x.view(4, 16, 128).abs().amax(-1))
    assert code.index("for lane_1 in range(") < code.index("for lane_2 in range(")
