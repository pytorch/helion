from __future__ import annotations

import ast
from dataclasses import replace
import re
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_workspace_integration import _args
from .test_cute_chained_loop_workspace_integration import _config
from .test_cute_chained_loop_workspace_integration import _paired_carries
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._compiler.cute import chained_loop
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_collectives import collective_bindings
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_preparation_cut import plan_preparation_cut
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_prepared_values import bind_frame_buffers
from helion._compiler.cute.chained_prepared_values import buffer_layout
from helion._compiler.cute.chained_prepared_values import emit_prepared_value
from helion._compiler.cute.chained_prepared_values import frontier_bindings
from helion._compiler.cute.chained_prepared_values import storage_dtype
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts


def test_materialized_scalar_uses_one_element_without_empty_guard():
    assert chain._materialized_value("scalar", (), (), "cutlass.Float32") == "scalar[0]"
    assert (
        chain._materialized_value("predicate", (), (), "cutlass.Boolean")
        == "cutlass.Boolean(predicate[0])"
    )


def test_existing_float_boundary_text_is_unchanged():
    assert (
        chain._materialized_value("tile", (16, 32), ("row", "col"), "cutlass.Float32")
        == "(tile[row, col] if 0 <= (row) < 16 and 0 <= (col) < 32 else cutlass.Float32(0))"
    )


def test_predicate_image_is_byte_addressed_and_restored_to_boolean():
    assert storage_dtype(torch.bool) == "cutlass.Uint8"
    assert (
        chain._materialized_value("tile", (17,), ("row",), "cutlass.Boolean")
        == "(cutlass.Boolean(tile[row]) if 0 <= (row) < 17 else cutlass.Boolean(0))"
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.bool, torch.int64]
)
def test_typed_frontier_layouts_have_exact_dense_row_major_spans(dtype):
    buffer = PreparationBuffer("image", "frontier", None, dtype, (17, 3))
    scratch = ScratchLayouts("xor")
    assert buffer_layout(buffer, scratch) == "cute.make_layout((17, 3), stride=(3, 1))"
    assert (
        buffer_layout(replace(buffer, shape=()), scratch)
        == "cute.make_layout((1,), stride=(1,))"
    )
    assert not scratch.xor_buffers


def test_role_local_carry_and_store_emission_preserves_default_and_two_phase_order():
    role = ChainedExecution(128, thread="consumer_thread", sync="consumer_sync()")
    original_store, original_carry = chain._emit_store, chained_loop.advance_carries
    captured = []

    def normalized(lines):
        names = {}

        def variable(match):
            return names.setdefault(match[0], f"temporary_{len(names)}")

        return re.sub(r"\b(?:v|chain_value)_\d+\b", variable, "\n".join(lines))

    def observe(original):
        def emit(*args, **kwargs):
            default = original(*args, **kwargs)
            # Re-emission consumes fresh names from the live codegen context.
            assert normalized(original(*args, **kwargs, execution=None)) == normalized(
                default
            )
            custom = original(*args, **kwargs, execution=role)
            captured.append((original.__name__, custom))
            return default

        return emit

    with (
        _cpu_codegen(),
        patch.object(chain, "_emit_store", observe(original_store)),
        patch.object(chained_loop, "advance_carries", observe(original_carry)),
    ):
        _paired_carries._bind_isolated(_args("cpu", swap=True)).to_code(_config())
    assert len(captured) == 4
    for name, lines in captured:
        source = "\n".join(lines)
        ast.parse(source)
        assert "chain_thread" not in source and "consumer_thread" in source
        assert "cute.arch.sync_threads()" not in source
        if name == "advance_carries":
            first, writes, tail = source.split("consumer_sync()")
            assert "chain_loop_carry_0_next" in first
            assert "chain_loop_carry_1_next" in first
            assert "chain_loop_carry_0[" in writes
            assert "chain_loop_carry_1[" in writes
            assert not tail.strip()


def test_actual_frontier_emission_preserves_materialized_reads_and_typed_scalar():
    kernel, args = _kda_fixture()
    original_plan, original_stage = (
        chain.plan_chained_matmul,
        chained_tcgen_stage.emit_stage,
    )
    frames, captures = [], []
    role = ChainedExecution(128, thread="prep_thread", sync="prep_sync()")

    def plan_with_frame(graphs):
        plan = original_plan(graphs)
        if plan is not None and plan.loop is not None:
            cut = plan_preparation_cut(graphs)
            assert cut is not None
            shapes = {
                node: chain._shape(node)
                for node in cut.region.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            }
            frame = plan_preparation_frame(plan, cut, shapes)
            assert frame is not None
            frames.append(frame)
        return plan

    def stage_with_images(cg, plan, *args, **kwargs):
        if not captures:
            frame = frames[-1]
            scratch = ScratchLayouts("xor")
            views = bind_frame_buffers(frame, "slot", scratch)
            ast.parse("\n".join(views))
            assert len(frontier_bindings(frame)) == 11
            boundaries = {
                **{node: f"chain_{index}_c" for index, node in enumerate(plan.dots)},
                **collective_bindings(plan),
                **{entry.node: entry.name for entry in plan.pointwise_cache.entries},
            }
            before = dict(boundaries)
            emissions = {}
            for buffer in frame.buffers:
                if buffer.kind == "frontier":
                    source = "\n".join(
                        emit_prepared_value(cg, plan, buffer, boundaries, role)
                    )
                    ast.parse(source)
                    assert "prep_thread" in source and source.endswith("prep_sync()")
                    assert "chain_thread" not in source
                    emissions[buffer.node] = source
                    if buffer.dtype == torch.bool:
                        assert "cutlass.Uint8(" in source
                    if not buffer.shape:
                        assert f"{buffer.name}[0]" in source
            assert boundaries == before
            cache = plan.pointwise_cache.entries[0]
            # The actual composite inverse reads its published cache. Also
            # exercise a frontier whose root itself is that materialized cache.
            assert any(cache.name in source for source in emissions.values())
            direct = PreparationBuffer(
                "direct_cache", "frontier", cache.node, cache.dtype, cache.shape
            )
            direct_source = "\n".join(
                emit_prepared_value(cg, plan, direct, boundaries, role)
            )
            assert cache.name in direct_source
            assert not any(
                f"chain_{index}_c[" in direct_source for index in range(len(plan.dots))
            )
            captures.append((views, emissions))
        return original_stage(cg, plan, *args, **kwargs)

    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
            patch.object(chain, "plan_chained_matmul", plan_with_frame),
            patch.object(chained_tcgen_stage, "emit_stage", stage_with_images),
        ):
            bound.to_code(
                helion.Config(
                    block_sizes=[128],
                    num_warps=16,
                    num_stages=2,
                    cute_chained_mma_schedule="tcgen05_tmem",
                    cute_chained_group_contractions=True,
                    cute_chained_scratch_layout="xor",
                    cute_chained_pointwise_vectorize=True,
                    cute_chained_scan_schedule="warp",
                    cute_chained_pointwise_cache_bytes=4096,
                    cute_chained_pointwise_unroll=8,
                    cute_chained_warp_mma_rows=32,
                )
            )
    assert captures
