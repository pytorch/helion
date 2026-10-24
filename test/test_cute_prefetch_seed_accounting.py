from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_chunk_row_seeds import bind
from test.test_cute_full_chunk_row_seeds import normalize_many
from test.test_cute_low_thread_independent_seeds import low_seeds

import helion
from helion._compiler.autotuner_heuristics.cute_full_chunk_row import (
    CuteFullChunkRowHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_rolled_cluster_packet import (
    CuteRolledClusterPacketHeuristic,
)
from helion._compiler.cute.prefetch_reduction_packets import _containers
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.inductor_lowering import ReductionLowering
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.language.memory_ops import load
from helion.language.memory_ops import store


@pytest.fixture(autouse=True)
def cpu_only():
    initialized = torch.cuda.is_initialized()
    with (
        _cpu_target(),
        _forbid_native_compile() as native,
        patch.object(
            CuteRolledClusterPacketHeuristic, "get_seed_configs", return_value=[]
        ),
    ):
        yield
        assert all(mock.call_count == 0 for mock in native)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("size", [7168, 8192])
def test_counted_prefetch_appends_t128_with_actual_cache_bound(dtype, size):
    with patch.object(
        CuteFullChunkRowHeuristic, "_consumer_prefetch_is_counted", return_value=False
    ):
        previous = bind(3, size, dtype)
    bound = bind(3, size, dtype)
    old = previous.config_spec.compiler_seed_configs
    current = bound.config_spec.compiler_seed_configs
    assert current[: len(old)] == old
    added = current[len(old) :]
    assert len(added) == 4
    assert all(seed.num_threads == [1, 128] for seed in added)
    with previous.env:
        default = previous.config_spec.default_config()
        reference = previous.config_spec.autotune_reference_config()
        fingerprint = previous.config_spec.structural_fingerprint()
    with bound.env:
        assert bound.config_spec.default_config() == default
        assert bound.config_spec.autotune_reference_config() == reference
        assert bound.config_spec.structural_fingerprint() == fingerprint
    assert bound.to_code(default) == previous.to_code(default)
    for seed in old:
        assert bound.to_code(seed) == previous.to_code(seed)
    for seed in added:
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            flat, effective = generation.strict_config_pair(seed)
            assert generation.strict_unflatten(flat) == effective
            assert all(effective.config[k] == value for k, value in seed.config.items())
        code = bound.to_code(effective)
        kernel = next(
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef)
            and any(ast.unparse(item) == "cute.kernel" for item in node.decorator_list)
        )
        classified = _containers(kernel.body, {})
        assert classified is not None
        assert classified[2] == 128
        allocations = [
            node
            for node in kernel.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
        ]
        assert len(allocations) == 2
        assert all(node.value.args[0].value == 64 for node in allocations)
        loops = [node for node in kernel.body if isinstance(node, ast.For)]
        assert len(loops) == 2
        assert "weight.iterator" in ast.unparse(loops[0])
        assert "weight.iterator" not in ast.unparse(loops[1])


@pytest.mark.parametrize(
    ("matrix_count", "expected"),
    [(1, [256, 128]), (2, [256]), (3, [256]), (4, [])],
)
def test_resource_charge_keeps_every_original_load(matrix_count, expected):
    matrices = tuple(
        torch.empty((3, 8192), dtype=torch.bfloat16) for _ in range(matrix_count)
    )
    bound = (
        bind(3, 8192, torch.bfloat16)
        if matrix_count == 1
        else helion.kernel(
            normalize_many, backend="cute", static_shapes=True
        )._bind_isolated(
            (matrices[0], matrices[1:], torch.empty(8192, dtype=torch.bfloat16))
        )
    )
    assert CuteFullChunkRowHeuristic._consumer_prefetch_is_counted(
        bound.host_function.device_ir
    )
    additions = low_seeds(bound)
    assert list(dict.fromkeys(seed.num_threads[1] for seed in additions)) == expected
    for seed in additions:
        tree = ast.parse(bound.to_code(seed))
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(ast.unparse(item) == "cute.kernel" for item in node.decorator_list)
        )
        classified = _containers(kernel.body, {})
        assert classified is not None and classified[2] <= 128


def test_proof_declines_extra_graphs_and_multiple_statistics():
    bound = bind(3, 7168, torch.bfloat16)
    ir = bound.host_function.device_ir
    proof = CuteFullChunkRowHeuristic._consumer_prefetch_is_counted
    assert proof(ir)
    with patch.object(ir, "graphs", [*ir.graphs, ir.graphs[0]]):
        assert not proof(ir)
    original = ir.graphs[ir.rolled_reductions[0].original_graph_id].graph
    reduction = next(
        node
        for node in original.nodes
        if isinstance(node.meta.get("lowering"), ReductionLowering)
    )
    duplicate = original.node_copy(reduction, lambda node: node)
    try:
        assert not proof(ir)
    finally:
        original.erase_node(duplicate)
    assert proof(ir)


def test_proof_retains_reserve_without_a_consumer_only_load():
    bound = bind(3, 7168, torch.bfloat16)
    ir = bound.host_function.device_ir
    proof = CuteFullChunkRowHeuristic._consumer_prefetch_is_counted
    assert proof(ir)
    producer, consumer = [
        info.graph for info in ir.graphs if isinstance(info, ReductionLoopGraphInfo)
    ]
    weight_load = next(
        node for node in consumer.nodes if node.target is load and node.name == "load_1"
    )
    extra = producer.node_copy(weight_load, lambda node: node)
    try:
        assert not proof(ir)
    finally:
        producer.erase_node(extra)
    assert proof(ir)


def test_proof_retains_reserve_with_an_additional_store():
    bound = bind(3, 7168, torch.bfloat16)
    ir = bound.host_function.device_ir
    proof = CuteFullChunkRowHeuristic._consumer_prefetch_is_counted
    assert proof(ir)
    original = ir.graphs[ir.rolled_reductions[0].original_graph_id].graph
    output_store = next(node for node in original.nodes if node.target is store)
    extra = original.node_copy(output_store, lambda node: node)
    try:
        assert not proof(ir)
    finally:
        original.erase_node(extra)
    assert proof(ir)


@pytest.mark.parametrize("mutation", ["name", "address", "source", "opaque"])
def test_proof_declines_unmatched_or_opaque_load_paths(mutation):
    bound = bind(3, 7168, torch.bfloat16)
    ir = bound.host_function.device_ir
    proof = CuteFullChunkRowHeuristic._consumer_prefetch_is_counted
    assert proof(ir)
    consumer = [info for info in ir.graphs if isinstance(info, ReductionLoopGraphInfo)][
        1
    ].graph
    node = next(
        node for node in consumer.nodes if node.target is load and node.name == "load_1"
    )
    if mutation == "name":
        with patch.object(node, "name", "not_in_original"):
            assert not proof(ir)
    elif mutation == "address":
        original_args = node.args
        node.args = (node.args[0], [slice(1, None)])
        try:
            assert not proof(ir)
        finally:
            node.args = original_args
    elif mutation == "source":
        original = ir.graphs[ir.rolled_reductions[0].original_graph_id].graph
        old = next(item for item in original.nodes if item.name == node.name)
        assert node.args[0].meta is not old.args[0].meta
        with patch.dict(node.args[0].meta, {"val": torch.empty(7168)}):
            assert not proof(ir)
    else:
        original = ir.graphs[ir.rolled_reductions[0].original_graph_id].graph
        opaque = original.call_function(abs, args=(-1,))
        try:
            assert not proof(ir)
        finally:
            original.erase_node(opaque)
    assert proof(ir)


@pytest.mark.parametrize("threads", [(128,), (256,), (512, 1024)])
def test_new_profiles_respect_narrowed_thread_domain(threads):
    bound = bind(3, 7168, torch.bfloat16)
    spec = bound.config_spec
    item = spec.num_threads.block_id_lookup(spec.reduction_loops[0].block_id)
    with patch.object(item, "_fragment", return_value=EnumFragment(threads)):
        seeds = low_seeds(bound)
    assert all(seed.num_threads[1] in threads for seed in seeds)
    assert len(seeds) == (4 if threads in [(128,), (256,)] else 0)
