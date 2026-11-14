"""Address coverage of a CuTe norm-backward kernel's fragment emitter, on CPU.

Shared by the CuTe codegen tests that render ``layer_norm_bwd`` /
``rms_norm_bwd`` and find the fragment-buffer schedule in the source.
"""

from __future__ import annotations

import ast
import copy
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def assert_fragment_norm_coverage(
    source: str, rows: int, width: int
) -> dict[str, object]:
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

    def names(node: ast.AST) -> set[str]:
        return {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    def events(node: ast.AST) -> list[ast.stmt]:
        found: list[ast.stmt] = []
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

    def slice_block(
        body: Sequence[ast.stmt],
        live: set[str],
        select_events: Callable[[ast.AST], list[ast.stmt]],
    ) -> tuple[list[ast.stmt], set[str]]:
        kept: list[ast.stmt] = []
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
                    live.difference_update(
                        t.id for t in statement.targets if isinstance(t, ast.Name)
                    )
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
    scalar: dict[str, Any] = {}
    constant_body: list[ast.stmt] = [*constants]
    exec(
        compile(ast.Module(body=constant_body, type_ignores=[]), "<constants>", "exec"),
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
        def __init__(self, name: str, offset: int = 0) -> None:
            self.name = name
            self.offset = int(offset)

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.name, self.offset + int(offset))

    def record(counters: dict[str, np.ndarray], pointer: Pointer) -> None:
        assert 0 <= pointer.offset < sizes[pointer.name], (
            pointer.name,
            pointer.offset,
            sizes[pointer.name],
        )
        counters[pointer.name][pointer.offset] += 1

    current: dict[str, tuple[int, ...]] = {"block": (0, 0, 0), "thread": (0, 0, 0)}
    env: dict[str, object] = {
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
    host_buffers: dict[str, str] = {}
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
    feature_counts: list[dict[str, int]] = []
    for phase in feature_phases:
        count = ast.literal_eval(cast("ast.Call", phase.iter).args[1])
        observed = {
            n: np.zeros((count, width), dtype=np.int32)
            for n in ("x", "grad_out", "weight")
        }

        def feature_events(
            node: ast.AST,
            phase: ast.For = phase,
            observed: dict[str, np.ndarray] = observed,
        ) -> list[ast.stmt]:
            return [
                ast.Expr(
                    ast.Call(
                        ast.Name("_record_feature", ast.Load()),
                        [
                            ast.Constant(host_buffers[n.value.id]),
                            ast.Name(cast("ast.Name", phase.target).id, ast.Load()),
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

        def record_feature(
            name: str,
            row: int,
            index: int,
            observed: dict[str, np.ndarray] = observed,
        ) -> None:
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
