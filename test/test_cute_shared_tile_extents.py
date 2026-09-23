from __future__ import annotations

import ast
from collections import Counter
from itertools import starmap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test.test_cute_sibling_loop_threads import _context
from test.test_cute_sibling_loop_threads import _row_ir

import helion
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._compiler.tile_strategy import PerThreadNDTileStrategy


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


def _launch_dims(tree: ast.Module) -> tuple[int, int, int]:
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    return ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "block"))
