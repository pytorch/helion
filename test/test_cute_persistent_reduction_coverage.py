from __future__ import annotations

import ast
import copy
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize("width", [1024, 2048, 4096])
def test_graph_reduction_covers_extent_beyond_hardware_thread_cap(
    name: str, width: int
) -> None:
    from examples.layer_norm import layer_norm_bwd
    from examples.rms_norm import rms_norm_bwd

    x = torch.empty(64, width, dtype=torch.float16)
    weight = torch.empty(width, dtype=torch.float16)
    if name == "rms_norm":
        example = rms_norm_bwd
        inputs = (torch.empty_like(x), x, weight, torch.empty(64, 1))
    else:
        example = layer_norm_bwd
        inputs = (
            torch.empty_like(x),
            x,
            torch.empty(64),
            torch.empty(64),
            weight,
            True,
        )
    bound = helion.kernel(
        example.fn,
        backend="cute",
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ).bind(inputs)
    # Both row axes are scalar. The persistent feature reduction can initially
    # claim the entire 1024-thread block, but wider rows still require lanes.
    config = helion.Config(block_sizes=[32, 1])
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        source = ast.parse(bound.to_code(config))
    if "fragment_buffer" in ast.unparse(source):
        assert name == "layer_norm"
        _assert_fragment_norm_coverage(ast.unparse(source), 64, width)
        return
    lane_extents = {
        ast.literal_eval(node.iter.args[0])
        for node in ast.walk(source)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        and isinstance(node.iter, ast.Call)
    }
    block = next(
        ast.literal_eval(keyword.value)
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
        for keyword in node.keywords
        if keyword.arg == "block"
    )
    assert len(lane_extents) <= 1
    per_thread = next(iter(lane_extents), 1)
    assert block[0] * per_thread == width
    assert block[1:] == (1, 1)


def _assert_fragment_norm_coverage(source: str, rows: int, width: int):
    """Execute actual launch/address controls; discard only data-value math.

    This is an address/ownership proof, not a floating-point simulator. Each
    selected global load/store retains its exact enclosing loops and masks.
    Backward slicing retains their scalar reaching definitions and rejects a
    shared/data-dependent address. All emitted physical threads are executed.
    """
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    constants = [n for n in tree.body if isinstance(n, ast.Assign)]
    assert "fragment_buffer" in source

    def names(node):
        return {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    def events(node):
        found = []
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("load", "store")
                and any(
                    isinstance(c, ast.Attribute) and c.attr == "iterator"
                    for c in ast.walk(n.func.value)
                )
            ):
                found.append(
                    ast.Expr(
                        ast.Call(
                            ast.Name("_record_" + n.func.attr, ast.Load()),
                            [copy.deepcopy(n.func.value)],
                            [],
                        )
                    )
                )
        return found

    def slice_block(body, live, select_events):
        kept = []
        live = set(live)
        for statement in reversed(body):
            if isinstance(statement, ast.For):
                child, inner = slice_block(statement.body, live, select_events)
                if child:
                    assert not statement.orelse
                    assert isinstance(statement.target, ast.Name)
                    live.update(inner - {statement.target.id})
                    live.update(names(statement.iter))
                    kept.append(
                        ast.For(
                            copy.deepcopy(statement.target),
                            copy.deepcopy(statement.iter),
                            child,
                            [],
                        )
                    )
            elif isinstance(statement, ast.If):
                yes, live_yes = slice_block(statement.body, live, select_events)
                no, live_no = slice_block(statement.orelse, live, select_events)
                if yes or no:
                    live.update(live_yes | live_no | names(statement.test))
                    kept.append(
                        ast.If(copy.deepcopy(statement.test), yes or [ast.Pass()], no)
                    )
            else:
                selected = select_events(statement)
                if selected:
                    # The fragment emitter uses statement-level masks for its
                    # host memory operations. Do not erase an expression mask.
                    assert not any(
                        isinstance(n, ast.IfExp) for n in ast.walk(statement)
                    )
                    for event in selected:
                        live.update(names(event))
                    kept.extend(reversed(selected))
                elif isinstance(statement, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in live for t in statement.targets
                ):
                    assert all(isinstance(t, ast.Name) for t in statement.targets)
                    # No loaded/shared payload may influence this address proof.
                    assert not any(
                        isinstance(n, ast.Subscript)
                        and isinstance(n.value, ast.Name)
                        and n.value.id.startswith("fragment_buffer")
                        for n in ast.walk(statement.value)
                    )
                    live.difference_update(t.id for t in statement.targets)
                    live.update(names(statement.value))
                    kept.append(copy.deepcopy(statement))
                elif isinstance(statement, (ast.While, ast.Try, ast.With)):
                    raise AssertionError("unsupported control in address proof")
        return list(reversed(kept)), live

    body, live = slice_block(fn.body, set(), events)
    declared = (
        {arg.arg for arg in fn.args.args}
        | {t.id for n in constants for t in n.targets if isinstance(t, ast.Name)}
        | {"cutlass", "cute", "range", "min", "max", "_record_load", "_record_store"}
    )
    assert live <= declared, live - declared
    launches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_launcher"
    ]
    assert len(launches) == 1
    block = next(
        ast.literal_eval(k.value) for k in launches[0].keywords if k.arg == "block"
    )
    scalar = {}
    exec(
        compile(ast.Module(body=constants, type_ignores=[]), "<constants>", "exec"),
        scalar,
    )
    grid = eval(
        compile(ast.Expression(launches[0].args[1]), "<actual-grid>", "eval"), scalar
    )
    assert len(grid) == 1
    blocks = grid[0]
    assert blocks == (rows + scalar["_BLOCK_SIZE_0"] - 1) // scalar["_BLOCK_SIZE_0"]
    sizes = {
        "x": rows * width,
        "grad_out": rows * width,
        "weight": width,
        "mean": rows,
        "rstd": rows,
        "grad_x": rows * width,
        "grad_weight_blocks": blocks * width,
        "grad_bias_blocks": blocks * width,
    }
    reads = {name: np.zeros(size, dtype=np.int32) for name, size in sizes.items()}
    writes = {name: np.zeros(size, dtype=np.int32) for name, size in sizes.items()}

    class Pointer:
        def __init__(self, name, offset=0):
            self.name = name
            self.offset = int(offset)

        def __add__(self, offset):
            return Pointer(self.name, self.offset + int(offset))

    def record(counters, pointer):
        assert 0 <= pointer.offset < sizes[pointer.name], (
            pointer.name,
            pointer.offset,
            sizes[pointer.name],
        )
        counters[pointer.name][pointer.offset] += 1

    current = {"block": (0, 0, 0), "thread": (0, 0, 0)}
    env = {
        **scalar,
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: current["thread"], block_idx=lambda: current["block"]
            )
        ),
        "_record_load": lambda p: record(reads, p),
        "_record_store": lambda p: record(writes, p),
    }
    env.update({name: SimpleNamespace(iterator=Pointer(name)) for name in sizes})
    code = compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        "<actual-address-controls>",
        "exec",
    )
    for cta in range(blocks):
        current["block"] = (cta, 0, 0)
        for thread in itertools.product(*(range(n) for n in block)):
            current["thread"] = thread
            exec(code, env)
    for name in ("x", "grad_out", "weight", "mean", "rstd"):
        assert np.all(reads[name] > 0), (name, "missing input coordinate")
    for name in ("grad_x", "grad_weight_blocks", "grad_bias_blocks"):
        assert np.all(writes[name] == 1), (
            name,
            "missing/duplicate output coordinate",
            np.unique(writes[name], return_counts=True),
        )
    assert all(
        not np.any(writes[name]) for name in ("x", "grad_out", "weight", "mean", "rstd")
    )
    # Track the two feature contractions through the same shared buffers
    # filled by the actual weight/x/grad_out loads. Enumerate shared addresses
    # at every reduction iteration, retaining coordinate definitions.
    host_buffers = {}
    phases = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("fragment_index")
    ]
    for phase in phases:
        host_names = {
            a.value.id
            for n in ast.walk(phase)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "load"
            for a in ast.walk(n.func.value)
            if isinstance(a, ast.Attribute)
            and a.attr == "iterator"
            and isinstance(a.value, ast.Name)
        }
        buffers = {
            t.value.id
            for n in ast.walk(phase)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and t.value.id.startswith("fragment_buffer")
        }
        if host_names:
            assert len(host_names) == len(buffers) == 1
            host_buffers[next(iter(buffers))] = next(iter(host_names))
    feature_phases = [
        phase
        for phase in phases
        if any(
            isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id.startswith("fragment_reduce_index")
            for n in ast.walk(phase)
        )
        and any(
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and host_buffers.get(n.value.id) == "weight"
            for n in ast.walk(phase)
        )
    ]
    assert len(feature_phases) == 2, "both feature contractions must remain present"
    feature_counts = []
    for phase in feature_phases:
        count = ast.literal_eval(phase.iter.args[1])
        observed = {
            n: np.zeros((count, width), dtype=np.int32)
            for n in ("x", "grad_out", "weight")
        }

        def feature_events(node, phase=phase, observed=observed):
            return [
                ast.Expr(
                    ast.Call(
                        ast.Name("_record_feature", ast.Load()),
                        [
                            ast.Constant(host_buffers[n.value.id]),
                            ast.Name(phase.target.id, ast.Load()),
                            copy.deepcopy(n.slice),
                        ],
                        [],
                    )
                )
                for n in ast.walk(node)
                if isinstance(n, ast.Subscript)
                and isinstance(n.ctx, ast.Load)
                and isinstance(n.value, ast.Name)
                and host_buffers.get(n.value.id) in observed
            ]

        feature_body, feature_live = slice_block([phase], set(), feature_events)
        assert feature_live <= {
            "fragment_thread",
            "range",
            "_record_feature",
            "cutlass",
        }, feature_live

        def record_feature(name, row, index, observed=observed):
            feature = int(index) % width
            if name != "weight":
                assert int(index) // width == row, (name, row, index)
            else:
                assert int(index) == feature
            observed[name][row, feature] += 1

        env["_record_feature"] = record_feature
        feature_code = compile(
            ast.fix_missing_locations(ast.Module(body=feature_body, type_ignores=[])),
            "<actual-feature-coordinates>",
            "exec",
        )
        for thread in range(block[0]):
            env["fragment_thread"] = thread
            exec(feature_code, env)
        assert np.all(observed["grad_out"] == 1)
        assert np.all(observed["weight"] == 1)
        assert not np.any(observed["x"]) or np.all(observed["x"] == 1)
        feature_counts.append(
            {name: int(values.sum()) for name, values in observed.items()}
        )
    assert sum(counts["x"] > 0 for counts in feature_counts) == 1
    return {
        "feature_contractions": feature_counts,
        "rows": rows,
        "width": width,
        "blocks": blocks,
        "threads": block,
        "read_counts": {n: int(v.sum()) for n, v in reads.items() if v.any()},
        "write_counts": {n: int(v.sum()) for n, v in writes.items() if v.any()},
        "sliced_ast": ast.unparse(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        ),
    }


@pytest.fixture(scope="module")
def _partial_fragment_norm_source():
    from examples.layer_norm import layer_norm_bwd

    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    x = torch.empty(35, 64, dtype=torch.float16)
    inputs = (
        torch.empty_like(x),
        x,
        torch.empty(35),
        torch.empty(35),
        torch.empty(64, dtype=x.dtype),
        True,
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        kernel = helion.kernel(
            layer_norm_bwd.fn,
            backend="cute",
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        return kernel.bind(inputs).to_code(helion.Config(block_sizes=[32, 4]))


def test_fragment_norm_coverage_partial_row_tile(_partial_fragment_norm_source):
    _assert_fragment_norm_coverage(_partial_fragment_norm_source, 35, 64)


@pytest.mark.parametrize(
    "damage", ["grid", "dx_coordinate", "load_extent", "reduction_extent", "tail_mask"]
)
def test_fragment_norm_coverage_rejects_missing_coordinates(
    _partial_fragment_norm_source, damage
):
    tree = ast.parse(_partial_fragment_norm_source)
    if damage == "grid":
        launch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_launcher"
        )
        launch.args[1] = ast.Tuple([ast.Constant(0)], ast.Load())
    elif damage == "dx_coordinate":
        store = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "store"
            and "grad_x.iterator" in ast.unparse(n.func.value)
        )
        coordinate = next(
            n
            for n in ast.walk(store.func.value)
            if isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Mod)
            and isinstance(n.right, ast.Constant)
            and n.right.value == 64
        )
        coordinate.right = ast.Constant(32)
    elif damage == "load_extent":
        phase = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id.startswith("fragment_index")
            and "x.iterator" in ast.unparse(n)
        )
        phase.iter.args[1] = ast.Constant(ast.literal_eval(phase.iter.args[1]) - 1)
    elif damage == "reduction_extent":
        reduction = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id.startswith("fragment_reduce_index")
            and ast.literal_eval(n.iter.args[0]) == 64
        )
        reduction.iter.args[0] = ast.Constant(63)
    else:
        mask = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "store"
                and "grad_x.iterator" in ast.unparse(call.func.value)
                for call in ast.walk(n)
            )
        )
        mask.test = ast.Constant(True)
    with pytest.raises(AssertionError):
        _assert_fragment_norm_coverage(ast.unparse(tree), 35, 64)
