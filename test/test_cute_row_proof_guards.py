from __future__ import annotations

import ast
import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from test.test_cute_rolled_online_epilogue import _config
from test.test_cute_rolled_online_epilogue import _program

from helion._compiler.cute import register_row_reduction
from helion._compiler.cute import scalar_epilogue

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction


@pytest.mark.parametrize(
    "write",
    (
        "p = cutlass.Float32(2)",
        "if True:\n    p = cutlass.Float32(2)",
        "for i in range(1):\n    p = cutlass.Float32(2)",
    ),
)
def test_scalar_sinking_preserves_a_crossed_dependency_write(write: str) -> None:
    original = scalar_epilogue.lower_scalar_epilogue
    witnesses: dict[str, ast.Module] = {}

    def intercept(
        body: list[ast.stmt],
        fn: DeviceFunction,
        constants: dict[str, int] | None,
        threads: tuple[int, int, int],
        disjoint: set[frozenset[str]],
    ) -> list[ast.stmt]:
        synthetic = ast.parse(
            "p = cutlass.Float32(1)\nq = p\n"
            + write
            + "\nlane = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            "answer = _cute_grouped_reduce_cluster(p, 'sum', cutlass.Float32(0), "
            "lane, buffer, barrier, group_span=256, pre=1)\n"
            "(out.iterator + 0).store(q + answer)\n"
        ).body
        witnesses["before"] = ast.Module(body=synthetic, type_ignores=[])
        witnesses["after"] = ast.Module(
            body=original(copy.deepcopy(synthetic), fn, constants, threads, disjoint),
            type_ignores=[],
        )
        return original(body, fn, constants, threads, disjoint)

    with patch.object(scalar_epilogue, "lower_scalar_epilogue", intercept):
        _program(config=_config(online=False, epilogue=True))

    class Pointer:
        def __init__(self) -> None:
            self.values: list[float] = []

        def __add__(self, offset: int) -> Pointer:
            return self

        def store(self, value: float) -> None:
            self.values.append(value)

    outputs = []
    for tree in witnesses.values():
        ast.fix_missing_locations(tree)
        destination = Pointer()
        namespace = {
            "cutlass": SimpleNamespace(Float32=float, Int32=int),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0), block_idx_in_cluster=lambda: 0
                )
            ),
            "out": SimpleNamespace(iterator=destination),
            "buffer": None,
            "barrier": None,
            "_cute_grouped_reduce_cluster": lambda value, *args, **kwargs: value,
        }
        exec(compile(tree, "<scalar-sinking-write>", "exec"), namespace)
        outputs.append(destination.values)
    assert outputs == [[3.0], [3.0]]


@pytest.mark.parametrize("keep_inputs", [False, True])
def test_uniform_inputs_can_remain_before_the_collective(keep_inputs: bool) -> None:
    original = scalar_epilogue.lower_scalar_epilogue
    witnesses: dict[str, list[ast.stmt]] = {}

    def intercept(
        body: list[ast.stmt],
        fn: DeviceFunction,
        constants: dict[str, int] | None,
        threads: tuple[int, int, int],
        disjoint: set[frozenset[str]],
    ) -> list[ast.stmt]:
        synthetic = ast.parse(
            "p = cutlass.Float32(1)\nq = p\n"
            "lane = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            "answer = _cute_grouped_reduce_cluster(p, 'sum', cutlass.Float32(0), "
            "lane, buffer, barrier, group_span=256, pre=1)\n"
            "(out.iterator + 0).store(q + answer)\n"
        ).body
        witnesses["body"] = original(synthetic, fn, constants, threads, disjoint)
        return original(body, fn, constants, threads, disjoint)

    config = _config(online=False, epilogue=True)
    config.config["cute_scalar_epilogue_keep_inputs"] = keep_inputs
    with patch.object(scalar_epilogue, "lower_scalar_epilogue", intercept):
        _program(config=config)
    changed = witnesses["body"]
    guard = next(node for node in changed if isinstance(node, ast.If))
    in_guard = any(
        isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "q"
        for node in guard.body
    )
    assert in_guard is not keep_inputs
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_cute_grouped_reduce_cluster"
        for node in ast.walk(guard)
    )


def test_register_row_loop_indices_do_not_capture_an_existing_name() -> None:
    original = register_row_reduction.lower_register_row_reduction
    witnesses: dict[str, list[ast.stmt]] = {}

    def intercept(
        body: list[ast.stmt], constants: dict[str, int] | None
    ) -> list[ast.stmt]:
        synthetic = [
            *ast.parse("_register_index_0 = cutlass.Int32(123)").body,
            *ast.parse(ast.unparse(ast.Module(body=body, type_ignores=[]))).body,
            *ast.parse("saved_index = _register_index_0").body,
        ]
        changed = original(synthetic, constants)
        witnesses["body"] = changed
        return original(body, constants)

    config = _config(epilogue=False)
    config.config["cute_register_row"] = True
    with patch.object(
        register_row_reduction, "lower_register_row_reduction", intercept
    ):
        _program(config=config)
    tree = ast.Module(body=witnesses["body"], type_ignores=[])
    assert "_register_row =" in ast.unparse(tree)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert loops
    assert all(
        not isinstance(node.target, ast.Name) or node.target.id != "_register_index_0"
        for node in loops
    )
