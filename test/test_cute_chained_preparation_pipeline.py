from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_pointwise_residency import _finite_consumers
from helion._compiler.cute.chained_preparation_pipeline import plan_preparation_pipeline
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.language import memory_ops
from helion.language import view_ops


def _config(warps=8, *, pipeline=False, consumer_warps=4):
    return helion.Config(
        num_warps=warps,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=pipeline,
        cute_chained_pipeline_consumer_warps=consumer_warps,
    )


def _args(device="cpu", steps=3, late=False):
    torch.manual_seed(9341)
    return (
        *(
            torch.randn((steps, 16, 16), device=device, dtype=torch.bfloat16) * 0.125
            for _ in range(3)
        ),
        torch.randn((16, 16), device=device) * 0.125,
        late,
    )


def test_image_domain_guard_only_ignores_unchanged_preparation_uses_cpu():
    graph = Graph()
    image = graph.placeholder("image")
    prep_load = graph.call_function(memory_ops.load, (image, [image]))
    consumer = graph.call_function(torch.ops.aten.exp.default, (image,))
    gather = graph.call_function(
        view_ops.subscript, (consumer, (slice(None, None, 2),))
    )
    assert not _finite_consumers(image, set())
    assert _finite_consumers(image, set(), within=frozenset((consumer,)))
    assert not _finite_consumers(image, set(), within=frozenset((consumer, gather)))
    assert not _finite_consumers(image, set(), within=frozenset((consumer, prep_load)))


@contextmanager
def _pipeline_plan():
    """Observe the plan selected through the public configuration path."""
    original = chain.plan_chained_matmul
    captured = []

    def plan_pipeline(graphs):
        plan = original(graphs)
        assert plan is not None
        assert plan.preparation_pipeline is not None
        captured.append(plan.preparation_pipeline)
        return plan

    with patch.object(chain, "plan_chained_matmul", plan_pipeline):
        yield captured


@pytest.mark.parametrize("late", [False, True])
def test_pipeline_partition_keeps_late_independent_dag_and_orders_state_cpu(late):
    args = _args(late=late)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)),
            _pipeline_plan() as plans,
        ):
            source = bound.to_code(_config(pipeline=True))
    assert len(plans) == 1
    pipeline = plans[0]
    assert pipeline.preparation_threads == pipeline.recurrence_threads == 128
    assert pipeline.residency is None
    tree = ast.parse(source)
    roles = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "chain_thread < 128"
    )
    prep, recurrence = (
        ast.unparse(ast.Module(body=roles.body, type_ignores=[])),
        ast.unparse(ast.Module(body=roles.orelse, type_ignores=[])),
    )
    assert "chain_prep_thread = chain_thread" in prep
    assert "chain_recurrence_thread = chain_thread - 128" in recurrence
    for body in (prep, recurrence):
        assert "cute.arch.sync_threads()" not in body
        assert "alloc_smem(" not in body and "chain_allocator" not in body
        assert (
            "chain_frame_address.toint(), cute.AddressSpace.smem, assumed_align=128"
            in body
        )
    assert pipeline.frame.layout.allocated_bytes % 128 == 0
    assert all(
        region.byte_offset % 128 == 0 for region in pipeline.frame.layout.regions
    )
    assert "chain_iteration >= 2" in prep
    assert "chain_generation - 1 & 1" in prep
    assert "chain_generation & 1" in recurrence
    assert "chain_prep_barrier.arrive_and_wait()" in prep
    assert "chain_recurrence_barrier.arrive_and_wait()" in recurrence
    assert "chain_store_0" not in prep and "chain_store_0" in recurrence
    assert "chain_final_store_0" not in prep and "chain_final_store_0" not in recurrence
    assert "chain_final_store_0" in source
    assert not any(
        isinstance(child, ast.IfExp)
        for assignment in ast.walk(tree)
        if isinstance(assignment, ast.Assign)
        for target in assignment.targets
        for child in ast.walk(target)
    )


def test_actual_kda_pipeline_fits_only_with_proven_explicit_accumulator_residency_cpu():
    kernel, args = _kda_fixture()
    config = _config(16)
    config.config.update(
        block_sizes=[128],
        num_stages=2,
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
    )
    plan, cut, shapes = _capture(kernel, args, config)
    pipeline = plan_preparation_pipeline(plan, cut, shapes, 232448)
    assert pipeline is not None and pipeline.residency is not None
    assert pipeline.recurrence.layout.allocated_bytes == 81920
    assert pipeline.shared_bytes == 225152
    assert (
        plan_preparation_pipeline(plan, cut, shapes, pipeline.shared_bytes - 1) is None
    )
    assert (
        plan_preparation_pipeline(plan, cut, shapes, pipeline.shared_bytes) == pipeline
    )
    assert (
        plan_preparation_pipeline(replace(plan, threads=128), cut, shapes, 232448)
        is None
    )
    assert plan_preparation_pipeline(plan, cut, shapes, 232448, slots=1) is None
    for total_threads, consumer_warps in ((384, 8), (512, 8), (640, 16), (1024, 16)):
        widened = plan_preparation_pipeline(
            replace(plan, threads=total_threads),
            cut,
            shapes,
            232448,
            consumer_warps=consumer_warps,
        )
        assert widened is not None
        assert widened.recurrence_threads == 32 * consumer_warps
        assert widened.preparation_threads == total_threads - 32 * consumer_warps
        assert widened.shared_bytes == pipeline.shared_bytes
        assert widened.frame == pipeline.frame
        assert widened.recurrence == pipeline.recurrence
    for total_threads, consumer_warps in (
        (512, True),
        (512, 4.0),
        (512, "4"),
        (512, 0),
        (512, 12),
        (256, 8),
        (512, 16),
        (416, 8),
    ):
        assert (
            plan_preparation_pipeline(
                replace(plan, threads=total_threads),
                cut,
                shapes,
                232448,
                consumer_warps=consumer_warps,
            )
            is None
        )
    config.config["cute_chained_preparation_pipeline"] = True
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
            _pipeline_plan(),
        ):
            source = bound.to_code(config)
    assert "chain_12_c =" not in source
    assert "chain_12_seed" not in source
    assert "chain_13_seed_14_segment" in source
    assert "chain_13_seed_13_segment" not in source
    assert "chain_frames = cute.arch.alloc_smem(cutlass.Uint8, 99840" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("steps", [0, 1, 3, 5])
@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("teams", [(8, 4), (16, 8), (32, 16)])
def test_preparation_pipeline_gpu_keeps_typed_state_and_slot_generations(
    steps, late, teams
):
    args = _args(DEVICE, steps, late)
    saved = tuple(value.clone() for value in args[:4])
    warps, consumer_warps = teams
    config = _config(warps)
    original = _typed_sequence._bind_isolated(args).compile_config(config)
    config.config["cute_chained_preparation_pipeline"] = True
    config.config["cute_chained_pipeline_consumer_warps"] = consumer_warps
    bound = _typed_sequence._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)),
        _pipeline_plan(),
    ):
        pipeline = bound.compile_config(config)
    expected = original(*args)
    actual = pipeline(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pipeline(*args), actual, rtol=0, atol=0)
    torch.testing.assert_close(args[:4], saved, rtol=0, atol=0)
