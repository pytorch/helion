from __future__ import annotations

import ast
from collections import Counter
from itertools import starmap
from types import SimpleNamespace
from unittest.mock import patch

from examples.jagged_layer_norm import jagged_layer_norm_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_sibling_loop_threads import _context
from test.test_cute_sibling_loop_threads import _row_ir

import helion
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._compiler.tile_strategy import PerThreadNDTileStrategy
from helion._testing import skipUnlessBackends


def _code(blocks: list[int], threads: list[int], vec: int = 4) -> str:
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_flatten_nested_reductions=True,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(
            kernel,
            (torch.empty((32641, 128)), torch.empty(257, dtype=torch.int64), 1e-6),
        )
        with patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ):
            return bound.to_code(
                helion.Config(
                    block_sizes=blocks,
                    num_threads=threads,
                    cute_vector_widths=[1, vec, vec, vec],
                    cute_lane_layouts=["blocked", "strided", "strided", "strided"],
                    cute_cluster_n=1,
                )
            )


def _launch_dims(tree: ast.Module) -> tuple[int, int, int]:
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    return ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "block"))


def _coordinates(
    tree: ast.Module, stage: int, axis: int, threads: int, offset: int
) -> Counter[int]:
    """Evaluate emitted coordinate expressions, independently of strategy math."""
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(device)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    ranges = {
        node.target.id: int(ast.literal_eval(node.iter.args[0]))
        for node in ast.walk(device)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith(("lane_", "vec_lane_"))
        and isinstance(node.iter, ast.Call)
        and len(node.iter.args) == 1
    }
    counts: Counter[int] = Counter()
    for thread in range(threads):
        thread_idx = [0, 0, 0]
        thread_idx[axis] = thread
        scope = {
            "cutlass": SimpleNamespace(Int32=int, Int64=int),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(thread_idx=lambda value=thread_idx: value)
            ),
            f"tile_offset_{stage}": offset,
        }
        for lane in range(ranges.get(f"lane_{stage}", 1)):
            for vector_lane in range(ranges.get(f"vec_lane_{stage}", 1)):
                scope[f"lane_{stage}"] = lane
                scope[f"vec_lane_{stage}"] = vector_lane
                base_name = f"lane_base_{stage}"
                if base_name in assignments:
                    scope[base_name] = eval(ast.unparse(assignments[base_name]), scope)
                counts[eval(ast.unparse(assignments[f"indices_{stage}"]), scope)] += 1
    return counts


@pytest.mark.parametrize(
    ("blocks", "threads", "launch", "axis"),
    (
        ([1, 4096, 4096, 4096], [1, 8, 8, 16], (16, 1, 1), 0),
        ([4, 128, 128, 128], [4, 4, 8, 8], (4, 8, 1), 1),
    ),
)
@skipUnlessBackends(["cute"])
def test_rejected_jagged_configs_cover_each_tile_once(
    blocks: list[int], threads: list[int], launch: tuple[int, int, int], axis: int
) -> None:
    code = _code(blocks, threads)
    tree = ast.parse(code)
    assert _launch_dims(tree) == launch
    for stage in (1, 2, 3):
        for offset in (0, blocks[stage]):
            counts = _coordinates(tree, stage, axis, launch[axis], offset)
            assert counts == Counter(range(offset, offset + blocks[stage]))
    if axis == 0:
        assert code.count("threads_in_group=16") == 2
        assert "threads_in_group=32" not in code
    else:
        assert code.count("pre=4, group_span=32") == 2
        assert "group_span=16" not in code


@pytest.mark.parametrize("order", ((4, 8), (8, 4)))
@pytest.mark.parametrize("block_sizes", ((128, 128), (4, 128), (128, 4)))
@pytest.mark.parametrize("vec", (1, 4, 8))
def test_shared_axes_repartition_or_mask_every_tile(
    order: tuple[int, int], block_sizes: tuple[int, int], vec: int
) -> None:
    # A four-element tile must request at most four original threads.
    requested = list(starmap(min, zip(block_sizes, order, strict=True)))
    ir = _row_ir()
    config = helion.Config(
        block_sizes=[2, *block_sizes],
        num_threads=[2, *requested],
        cute_vector_widths=[1, vec, vec],
    )
    with _context(ir, config) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        widest = max(requested)
        assert dispatch.thread_block_dims() == (2, widest, 1)
        for block_id, block_size in enumerate(block_sizes, start=1):
            strategy = dispatch.block_id_to_strategy[(block_id,)]
            assert isinstance(strategy, PerThreadNDTileStrategy)
            assert dispatch.thread_extent_for_block_id(block_id) == widest
            elements = strategy._elements_per_thread_for_block(block_id)
            counts = Counter(
                thread * elements + lane
                for thread in range(widest)
                for lane in range(elements)
                if thread * elements + lane < block_size
            )
            assert counts == Counter(range(block_size))
            assert dispatch.has_surplus_threads_for_block_id(block_id) == (
                widest > block_size
            )
            actual_vec = strategy._cute_lane_vec_width_by_block.get(block_id, 1)
            assert elements % actual_vec == 0


@skipUnlessBackends(["cute"])
def test_jagged_tile_smaller_than_shared_axis_masks_surplus_threads() -> None:
    code = _code([4, 4, 128, 128], [4, 4, 8, 8])
    assert _launch_dims(ast.parse(code)) == (4, 8, 1)
    assert "cute.arch.thread_idx()[1]) < 4" in code
    assert code.count("pre=4, group_span=32") == 2


def test_scalar_axis_is_not_expanded_by_a_sibling() -> None:
    ir = _row_ir()
    config = helion.Config(block_sizes=[2, 128, 128], num_threads=[2, 1, 8])
    with _context(ir, config) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        assert dispatch.thread_extent_for_block_id(1) is None
        assert dispatch.thread_extent_for_block_id(2) == 8


def test_matrix_layouts_retain_their_own_thread_contract() -> None:
    ir = _row_ir()
    config = helion.Config(block_sizes=[2, 128, 128], num_threads=[2, 4, 8])
    with _context(ir, config) as (env, fn):
        with patch.object(env.config_spec, "matmul_facts", [SimpleNamespace()]):
            dispatch = TileStrategyDispatch(fn, config)
            fn.tile_strategy = dispatch
        assert dispatch.thread_extent_for_block_id(1) == 4
        assert dispatch.thread_extent_for_block_id(2) == 8
