from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_fragment_coverage import assert_fragment_norm_coverage

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize("width", [1024, 2048, 4096])
def test_graph_reduction_covers_extent_beyond_hardware_thread_cap(
    name: str, width: int
) -> None:
    layer_norm_bwd = import_example("layer_norm").layer_norm_bwd
    rms_norm_bwd = import_example("rms_norm").rms_norm_bwd

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
        "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        source = ast.parse(bound.to_code(config))
    if "fragment_buffer" in ast.unparse(source):
        assert name == "layer_norm"
        assert_fragment_norm_coverage(ast.unparse(source), 64, width)
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


@pytest.fixture(scope="module")
def _partial_fragment_norm_source():
    layer_norm_bwd = import_example("layer_norm").layer_norm_bwd

    from test.cute_population_contracts import _target

    from helion._testing.cute_codegen import forbid_native_compile
    from helion._testing.cute_codegen import mock_cuda_unavailable

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
        mock_cuda_unavailable(),
        _target(),
        forbid_native_compile(),
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
    assert_fragment_norm_coverage(_partial_fragment_norm_source, 35, 64)


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
        assert_fragment_norm_coverage(ast.unparse(tree), 35, 64)
