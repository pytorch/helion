from __future__ import annotations

import ast
from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest


class _CollectiveYields(ast.NodeTransformer):
    """Run the actual helper, stopping threads at CUDA collective operations."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = cast("ast.Call", self.generic_visit(node))
        name = ast.unparse(node.func)
        if name == "cute.arch.warp_reduction_sum":
            group = next(
                (kw.value for kw in node.keywords if kw.arg == "threads_in_group"),
                ast.Constant(32),
            )
            return ast.copy_location(
                ast.Yield(
                    ast.Tuple([ast.Constant("warp"), node.args[0], group], ast.Load())
                ),
                node,
            )
        if name == "cute.arch.sync_threads":
            return ast.copy_location(
                ast.Yield(
                    ast.Tuple(
                        [ast.Constant("barrier"), ast.Constant(None), ast.Constant(0)],
                        ast.Load(),
                    )
                ),
                node,
            )
        return node


class _Shared:
    def __init__(self, size: int) -> None:
        self.values = np.empty(size, dtype=np.float32)
        self.written = np.zeros(size, dtype=np.bool_)
        self.reads = 0

    def __getitem__(self, index: int) -> np.float32:
        assert self.written[index], "shared partial read before it was published"
        self.reads += 1
        return self.values[index]

    def __setitem__(self, index: int, value: np.float32) -> None:
        self.values[index] = value
        self.written[index] = True


_Request = tuple[str, np.float32 | None, int]
_Lane = Generator[_Request, np.float32 | None, np.ndarray]


def _run_sums(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Interpret warp shuffles and barriers; all other helper code runs unchanged."""
    threads, statistics = values.shape
    path = (
        Path(__file__).parent.parent
        / "helion/_compiler/cute/resident_reduction_runtime.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_cute_resident_sums"
    )
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    function = cast("ast.FunctionDef", _CollectiveYields().visit(function))
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    current_thread = 0
    shared = _Shared(statistics * (threads // 32))

    def allocate(dtype: object, size: int) -> _Shared:
        assert dtype is np.float32
        assert size == shared.values.size
        return shared

    namespace: dict[str, object] = {
        "cutlass": SimpleNamespace(
            Int32=np.int32,
            Float32=np.float32,
            range_constexpr=range,
            const_expr=lambda value: value,
            utils=SimpleNamespace(
                SmemAllocator=lambda: SimpleNamespace(allocate_tensor=allocate)
            ),
        ),
        "cute": SimpleNamespace(
            size=lambda value: value.size,
            make_layout=lambda size: size,
            make_rmem_tensor=lambda shape, dtype: np.empty(shape, dtype=dtype),
            arch=SimpleNamespace(thread_idx=lambda: (current_thread, 0, 0)),
        ),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    helper = cast("Callable[[np.ndarray, int], _Lane]", namespace[function.name])
    lanes = [helper(values[thread], threads) for thread in range(threads)]
    responses: list[np.float32 | None] = [None] * threads
    barriers = 0
    while True:
        requests: list[_Request] = []
        results: list[np.ndarray] = []
        for current_thread, lane in enumerate(lanes):
            try:
                requests.append(lane.send(responses[current_thread]))
            except StopIteration as result:
                results.append(result.value)
        if results:
            assert len(results) == threads, (
                "threads reached different collective counts"
            )
            return np.stack(results), barriers, shared.reads
        assert len(requests) == threads
        kind, _value, group = requests[0]
        assert all((request[0], request[2]) == (kind, group) for request in requests)
        if kind == "barrier":
            barriers += 1
            responses = [None] * threads
        else:
            assert kind == "warp"
            assert 1 < group <= min(threads, 32) and group & (group - 1) == 0
            reduced = np.array([request[1] for request in requests], dtype=np.float32)
            # NVIDIA's butterfly shuffle exchanges lanes differing in one bit.
            # All groups have power-of-two width and stay inside a physical warp.
            offset = group // 2
            while offset:
                reduced = reduced + reduced[np.arange(threads) ^ offset]
                offset //= 2
            responses = list(reduced)


@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("statistics", [1, 2, 4, 8])
def test_resident_sum_reaches_every_feature_lane(threads: int, statistics: int) -> None:
    values = (
        np.arange(threads * statistics).reshape(threads, statistics) % 37 - 18
    ).astype(np.float32)
    actual, barriers, reads = _run_sums(values)
    expected = np.broadcast_to(values.sum(axis=0, dtype=np.float64), actual.shape)
    np.testing.assert_array_equal(actual, expected)
    assert barriers == (2 if threads > 32 else 0)
    # One published partial per lane replaces serially loading every warp.
    assert reads == (threads * statistics if threads > 32 else 0)


@pytest.mark.parametrize("threads", [64, 128, 256, 512, 1024])
def test_resident_sum_fp32_rounding(threads: int) -> None:
    values = np.random.default_rng(1701).normal(size=(threads, 8)).astype(np.float32)
    actual, _barriers, _reads = _run_sums(values)
    expected = np.broadcast_to(values.sum(axis=0, dtype=np.float64), actual.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("threads", [1, 16, 32, 64, 256, 512, 1024])
def test_resident_sum_keeps_zero_identity_and_nonfinite_values(threads: int) -> None:
    zero = np.full((threads, 1), np.float32(-0.0))
    actual, _barriers, _reads = _run_sums(zero)
    assert not np.any(actual)
    np.testing.assert_array_equal(np.signbit(actual), threads <= 32)

    values = np.zeros((threads, 4), dtype=np.float32)
    values[0] = [np.inf, -np.inf, np.nan, np.inf]
    if threads > 1:
        values[-1, 3] = -np.inf
    with np.errstate(invalid="ignore"):
        actual, _barriers, _reads = _run_sums(values)
    assert np.all(np.isposinf(actual[:, 0]))
    assert np.all(np.isneginf(actual[:, 1]))
    assert np.all(np.isnan(actual[:, 2]))
    assert np.all(np.isnan(actual[:, 3]) if threads > 1 else np.isposinf(actual[:, 3]))
