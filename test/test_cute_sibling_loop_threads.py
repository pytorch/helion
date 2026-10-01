from __future__ import annotations

from contextlib import contextmanager
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteSiblingRowHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.compile_environment import BlockSizeInfo
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import LoopSpecBlockSizeSource
from helion._compiler.cute.layout_propagation import _plan_warp_per_row_execution
from helion._compiler.cute.loop_nesting import sibling_row_loop_blocks
from helion._compiler.cute.loop_nesting import tile_loop_paths
from helion._compiler.cute.thread_budget import tile_loop_thread_count
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import GraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.host_function import HostFunction
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteLaneLayoutSpec
from helion.autotuner.config_spec import CuteVectorWidthSpec
from helion.autotuner.config_spec import MemoryOpFact
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.language import _tracing_ops
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction


def _row_ir(
    *,
    nested: bool = False,
    last_end: int | None = None,
    columns: int = 1024,
    dtype: torch.dtype = torch.float32,
) -> DeviceIR:
    ir = DeviceIR()
    root, first, second = (torch.fx.Graph() for _ in range(3))
    source = root.placeholder("source")
    source.meta["val"] = torch.empty((16, columns), dtype=dtype)
    root.call_function(_tracing_ops._for_loop, (1, [0], [columns], []))
    (first if nested else root).call_function(
        _tracing_ops._for_loop,
        (2, [0], [columns if last_end is None else last_end], []),
    )
    ir.graphs = [
        RootGraphInfo(0, root),
        ForLoopGraphInfo(1, first, [], [1]),
        ForLoopGraphInfo(2, second, [], [2]),
    ]
    ir.root_ids = [0]
    ir.grid_block_ids = [[0]]
    return ir


class _Function:
    def __init__(self, ir: DeviceIR, config: Config) -> None:
        self.config = config
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}
        self.codegen = SimpleNamespace(
            codegen_graphs=ir.graphs,
            current_root_graph_info=ir.graphs[ir.root_ids[0]],
        )

    def new_var(self, name: str, *, dce: bool = False) -> str:
        return name


@contextmanager
def _context(
    ir: DeviceIR,
    config: Config,
    extents: tuple[int, ...] = (16, 1024, 1024),
) -> Iterator[tuple[CompileEnvironment, DeviceFunction]]:
    backend = CuteBackend()
    spec = ConfigSpec(
        backend=backend,
        target_device_capability=(10, 0),
        device=torch.device("cpu"),
        num_sm=148,
    )
    for block_id, extent in enumerate(extents):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=extent))
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=extent))
        spec.cute_vector_widths.append(
            CuteVectorWidthSpec(block_id=block_id, size_hint=extent)
        )
        spec.cute_lane_layouts.append(CuteLaneLayoutSpec(block_id=block_id))
    env = cast(
        "CompileEnvironment",
        SimpleNamespace(
            backend=backend,
            backend_name="cute",
            config_spec=spec,
            known_equal=operator.eq,
            is_jagged_tile=lambda block_id: False,
            block_sizes=[
                BlockSizeInfo(
                    block_id=block_id,
                    size=extent,
                    var=cast("torch.SymInt", block_id),
                    reduction=False,
                    block_size_source=LoopSpecBlockSizeSource(),
                )
                for block_id, extent in enumerate(extents)
            ],
        ),
    )
    fn = _Function(ir, config)
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch.object(
            HostFunction, "current", return_value=SimpleNamespace(device_ir=ir)
        ),
        patch.object(
            CuteBackend, "_cute_matmul_contraction_thread_reserve", return_value=1
        ),
        patch(
            "helion._compiler.cute.backend._is_mma_candidate_loop", return_value=False
        ),
        patch(
            "helion._compiler.cute.backend._detect_attention_mma_loop",
            return_value=False,
        ),
        patch(
            "helion._compiler.cute.backend._detect_specialized_mma_loop",
            return_value=False,
        ),
        patch(
            "helion._compiler.cute.backend._kernel_specialized_mma_plan",
            return_value=None,
        ),
    ):
        yield env, cast("DeviceFunction", fn)


@pytest.mark.parametrize(("nested", "expected"), ((False, 256), (True, 4096)))
def test_thread_budget_counts_nesting_not_siblings(nested: bool, expected: int) -> None:
    ir = _row_ir(nested=nested)
    config = Config(block_sizes=[16, 16, 16])
    with _context(ir, config) as (env, fn):
        assert tile_loop_thread_count(env, ir, ir.graphs, config) == expected
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        assert dispatch.thread_extent_for_block_id(1) == (None if nested else 16)
        assert dispatch.thread_extent_for_block_id(2) == (None if nested else 16)
        if not nested:
            assert dispatch.thread_block_dims() == (16, 16, 1)
            assert dispatch.thread_axis_for_block_id(1) == 1
            assert dispatch.thread_axis_for_block_id(2) == 1


def test_thread_budget_uses_maximum_of_each_axis() -> None:
    ir = _row_ir()
    ir.graphs[1].block_ids = [1, 2]
    ir.graphs[2].block_ids = [3, 4]
    config = Config(block_sizes=[2, 32, 4, 4, 32])
    with _context(ir, config, (16, 1024, 1024, 1024, 1024)) as (env, fn):
        # Each individual path needs 256 threads, but their shared launch
        # shape is (2, 32, 32), which exceeds the hardware limit.
        assert tile_loop_thread_count(env, ir, ir.graphs, config) == 2048


@pytest.mark.parametrize("second_grid_threads", (1, 16))
def test_multiple_roots_reuse_sibling_loop_axes(second_grid_threads: int) -> None:
    ir = _row_ir()
    ir.graphs.append(RootGraphInfo(3, torch.fx.Graph()))
    ir.root_ids.append(3)
    ir.grid_block_ids.append([3])
    config = Config(block_sizes=[16, 16, 16, second_grid_threads])
    with _context(ir, config, (16, 1024, 1024, 16)) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        assert tile_loop_thread_count(env, ir, ir.graphs, config) == 256
        assert dispatch.thread_block_dims() == (16, 16, 1)
        assert dispatch.thread_axis_for_block_id(1) == 1
        assert dispatch.thread_axis_for_block_id(2) == 1
        assert all(dispatch._strategy_branches())


def test_paths_follow_branches_while_bodies_and_separate_roots() -> None:
    ir = _row_ir()
    root = torch.fx.Graph()
    root.call_function(_tracing_ops._if, (True, 3, 4, [], []))
    root.call_function(_tracing_ops._while_loop, (5, 6, [], 7))
    branch = torch.fx.Graph()
    branch.call_function(_tracing_ops._for_loop, (1, [0], [1024], []))
    while_body = torch.fx.Graph()
    while_body.call_function(_tracing_ops._for_loop, (2, [0], [1024], []))
    ir.graphs[0].graph = root
    ir.graphs.extend(
        (
            GraphInfo(3, branch),
            GraphInfo(4, torch.fx.Graph()),
            GraphInfo(5, torch.fx.Graph()),
            GraphInfo(6, while_body),
            GraphInfo(7, torch.fx.Graph()),
            RootGraphInfo(8, torch.fx.Graph()),
            # An unreachable graph copy must not claim launch threads.
            ForLoopGraphInfo(9, torch.fx.Graph(), [], [9]),
        )
    )
    ir.root_ids.append(8)
    ir.grid_block_ids.append([3])
    assert tile_loop_paths(ir, ir.graphs) == (((0,), (1,)), ((0,), (2,)), ((3,),))


def test_nested_loops_keep_distinct_axes_in_dispatch() -> None:
    ir = _row_ir(nested=True)
    config = Config(block_sizes=[2, 16, 16], num_threads=[2, 16, 16])
    with _context(ir, config) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        assert dispatch.thread_block_dims() == (2, 16, 16)
        assert dispatch.thread_axis_for_block_id(1) == 1
        assert dispatch.thread_axis_for_block_id(2) == 2


def test_unreachable_loop_copies_do_not_claim_axes() -> None:
    ir = _row_ir()
    ir.graphs.append(ForLoopGraphInfo(3, torch.fx.Graph(), [], [3]))
    config = Config(block_sizes=[16, 16, 16, 16])
    with _context(ir, config, (16, 1024, 1024, 1024)) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        assert tile_loop_thread_count(env, ir, ir.graphs, config) == 256
        assert dispatch.thread_block_dims() == (16, 16, 1)
        assert dispatch.thread_axis_for_block_id(3) is None


def test_sibling_passes_can_share_a_warp_per_row() -> None:
    ir = _row_ir()
    config = Config(
        block_sizes=[4, 1024, 1024],
        num_threads=[4, 32, 32],
        cute_vector_widths=[1, 4, 4],
        cute_lane_layouts=["blocked", "strided", "strided"],
    )
    with _context(ir, config) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        _plan_warp_per_row_execution(ir.graphs[0], dispatch)
        root = cast("RootGraphInfo", ir.graphs[0])
        assert len(root.cute_grid_execution_plans) == 1
        assert dispatch.thread_block_dims() == (32, 4, 1)
        assert dispatch.thread_axis_for_block_id(0) == 1
        assert dispatch.thread_axis_for_block_id(1) == 0
        assert dispatch.thread_axis_for_block_id(2) == 0


@pytest.mark.parametrize("case", ("nested", "different_bounds"))
def test_row_layout_rejects_incompatible_passes(case: str) -> None:
    ir = _row_ir(
        nested=case == "nested", last_end=512 if case == "different_bounds" else 1024
    )
    config = Config(
        block_sizes=[2, 128, 128],
        num_threads=[2, 32, 32],
    )
    with _context(ir, config) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        _plan_warp_per_row_execution(ir.graphs[0], dispatch)
        assert not cast("RootGraphInfo", ir.graphs[0]).cute_grid_execution_plans


def test_row_layout_uses_normalized_sibling_thread_extents() -> None:
    ir = _row_ir()
    config = Config(block_sizes=[2, 128, 128], num_threads=[2, 32, 64])
    with _context(ir, config) as (env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        assert dispatch.thread_extent_for_block_id(1) == 64
        assert dispatch.thread_extent_for_block_id(2) == 64
        _plan_warp_per_row_execution(ir.graphs[0], dispatch)
        assert cast("RootGraphInfo", ir.graphs[0]).cute_grid_execution_plans
        assert dispatch.thread_block_dims() == (64, 2, 1)


@pytest.mark.parametrize(
    ("columns", "dtype", "vec"),
    ((1024, torch.float32, 4), (4096, torch.float16, 8), (32768, torch.bfloat16, 8)),
)
def test_sibling_seeds_configure_all_passes_by_block_id(
    columns: int, dtype: torch.dtype, vec: int
) -> None:
    ir = _row_ir(columns=columns, dtype=dtype)
    with _context(ir, Config(block_sizes=[16, 16, 16]), (16, columns, columns)) as (
        env,
        fn,
    ):
        spec = env.config_spec
        # Sequence registration order can differ between config fields.
        spec.num_threads[:] = list(reversed(spec.num_threads))
        spec.cute_vector_widths[:] = list(reversed(spec.cute_vector_widths))
        seeds = CuteSiblingRowHeuristic.get_seed_configs(env, ir)
        assert len(seeds) >= 4
        assert CuteSiblingRowHeuristic in get_heuristics("cute")
        assert CuteSiblingRowHeuristic.get_seed_config(env, ir) == seeds[0]
        assert not CuteSiblingRowHeuristic.should_promote(env)
        assert any(seed.block_sizes == [1, columns, columns] for seed in seeds)
        if columns <= 2048:
            assert any(seed.block_sizes == [4, columns, columns] for seed in seeds)
        for seed in seeds:
            values = seed.config.copy()
            spec.normalize(values)
            assert seed.block_sizes[1] == seed.block_sizes[2]
            assert spec.num_threads.config_get(
                seed.num_threads, 1
            ) == spec.num_threads.config_get(seed.num_threads, 2)
            widths = cast("list[int]", seed.config["cute_vector_widths"])
            width = spec.cute_vector_widths.config_get(widths, 1)
            if seed.config.get("cute_independent_reduction"):
                assert width in (vec, vec // 2)
                assert seed.config["cute_replicated_reduction"] is True
                assert seed.config["cute_vector_packet_unroll"] is True
            else:
                assert width == vec
            assert spec.cute_vector_widths.config_get(widths, 2) == width
            assert seed.config["cute_lane_layouts"][1:] == ["strided", "strided"]
            assert tile_loop_thread_count(env, ir, ir.graphs, seed) <= 1024


@pytest.mark.parametrize("case", ("nested", "different_bounds", "rolled_reduction"))
def test_sibling_seeds_require_matching_row_passes(case: str) -> None:
    ir = _row_ir(
        nested=case == "nested", last_end=512 if case == "different_bounds" else 1024
    )
    with _context(ir, Config(block_sizes=[16, 16, 16])) as (env, fn):
        if case == "rolled_reduction":
            env.config_spec.reduction_loops.append(
                ReductionLoopSpec(block_id=3, size_hint=1024)
            )
        else:
            assert sibling_row_loop_blocks(env, ir, ir.graphs) is None
        assert not CuteSiblingRowHeuristic.get_seed_configs(env, ir)


@pytest.mark.parametrize("reverse_facts", (False, True))
def test_sibling_cache_seeds_follow_tensor_pass_and_eviction_slot(
    reverse_facts: bool,
) -> None:
    ir = _row_ir(columns=4096, dtype=torch.float16)
    with _context(ir, Config(block_sizes=[1, 4096, 4096]), (16, 4096, 4096)) as (
        env,
        fn,
    ):
        spec = env.config_spec
        spec.load_eviction_policies.length = 6
        # Eviction slots need not follow memory-fact or pass order. Explicit
        # loads have no slot, and a single-use input keeps its default policy.
        spec.memory_op_facts = [
            MemoryOpFact(
                indexing_index=index,
                kind="load",
                eviction_index=slot,
                tensor_name=name,
                dtype=torch.float16,
                ndim=2,
                num_reuses=1,
                matmul_operand=None,
                subscript_block_ids=(0, pass_block),
            )
            for index, (name, slot, pass_block) in enumerate(
                (
                    ("x", 3, 1),
                    ("y", 5, 1),
                    ("x", 0, 2),
                    ("y", 2, 2),
                    ("explicit", None, 1),
                    ("explicit", 4, 2),
                    ("once", 1, 2),
                )
            )
        ]
        if reverse_facts:
            spec.memory_op_facts.reverse()
        seeds = CuteSiblingRowHeuristic.get_seed_configs(env, ir)
        expected = ["l1_l2_first", "", "l1_l2_first", "l1_l2_last", "", "l1_l2_last"]
        assert any("load_eviction_policies" not in seed.config for seed in seeds)
        cache_seeds = [
            seed for seed in seeds if "load_eviction_policies" in seed.config
        ]
        assert cache_seeds
        for seed in cache_seeds:
            assert seed.config["load_eviction_policies"] == expected
            spec.normalize(seed.config.copy())
        assert any(
            seed.block_sizes == [1, 4096, 4096] and seed.num_threads[1:] == [256, 256]
            for seed in cache_seeds
        )
