from __future__ import annotations

from typing import Any

import pytest

from helion._compiler.cute.tcgen05_prologue import inline_floordiv_nonneg_expr
from helion._compiler.cute.tcgen05_prologue import inline_mod_nonneg_expr
from helion._compiler.cute.tcgen05_prologue import masked_cluster_rank_expr
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def test_inline_division_forms():
    assert inline_floordiv_nonneg_expr("x", "n", 1) == "(x)"
    assert inline_floordiv_nonneg_expr("x", "n", 2) == "((x) >> cutlass.Int32(1))"
    assert inline_floordiv_nonneg_expr("x", "n", 32) == "((x) >> cutlass.Int32(5))"
    # Non-power-of-two and dynamic counts keep the DSL's floor division.
    assert inline_floordiv_nonneg_expr("x", "n", 80) == "((x) // (n))"
    assert inline_floordiv_nonneg_expr("x", "n", None) == "((x) // (n))"
    assert inline_mod_nonneg_expr("x", "n", 2) == "((x) & cutlass.Int32(1))"
    assert inline_mod_nonneg_expr("x", "n", 32) == "((x) & cutlass.Int32(31))"
    assert inline_mod_nonneg_expr("x", "n", 1) == "((x) % (n))"
    assert inline_mod_nonneg_expr("x", "n", 80) == "((x) % (n))"
    assert inline_mod_nonneg_expr("x", "n", None) == "((x) % (n))"


def test_masked_cluster_rank():
    rank = "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())"
    assert masked_cluster_rank_expr(rank, 4) == f"({rank}) & cutlass.Int32(3)"
    assert masked_cluster_rank_expr(rank, 2) == f"({rank}) & cutlass.Int32(1)"
    assert masked_cluster_rank_expr(rank, 1) == rank
    assert masked_cluster_rank_expr(rank, 6) == rank


@pytest.mark.parametrize("divisor", [1, 2, 7, 8, 80])
def test_runtime_helpers_fold_python_ints(divisor):
    pytest.importorskip("cutlass")
    from helion._compiler.cute._tcgen05_prologue_runtime import floordiv_nonneg
    from helion._compiler.cute._tcgen05_prologue_runtime import mod_nonneg

    for value in (0, 1, 5, 79, 80, 81, 1023):
        assert floordiv_nonneg(value, divisor) == value // divisor
        assert mod_nonneg(value, divisor) == value % divisor


def test_split_init_surface_is_pinned_to_the_installed_dsl():
    """The no-init pipeline constructors mirror one DSL release's private
    surface; on it the fast path is enabled, and the pin's expected surface
    is the live one (a DSL bump fails here loudly while codegen falls back)."""
    pytest.importorskip("cutlass")
    import dataclasses
    import inspect

    from cutlass.pipeline import PipelineTmaUmma
    from cutlass.pipeline import PipelineUmmaAsync
    from cutlass.pipeline.helpers import MbarrierArray

    from helion._compiler.cute import _tcgen05_prologue_runtime as rt
    from helion._compiler.cute.cutedsl_compat import installed_cute_dsl_version
    from helion._compiler.cute.tcgen05_prologue import split_barrier_init_supported

    assert installed_cute_dsl_version() == rt.SPLIT_INIT_VALIDATED_DSL_VERSION
    assert (
        tuple(inspect.signature(inspect.unwrap(MbarrierArray.__init__)).parameters)
        == rt._MBARRIER_ARRAY_INIT_PARAMETERS
    )
    assert tuple(f.name for f in dataclasses.fields(PipelineTmaUmma)) == (
        rt._PIPELINE_TMA_UMMA_FIELDS
    )
    assert tuple(f.name for f in dataclasses.fields(PipelineUmmaAsync)) == (
        rt._PIPELINE_UMMA_ASYNC_FIELDS
    )
    assert rt.SPLIT_BARRIER_INIT_SUPPORTED is True
    assert split_barrier_init_supported() is True


def test_no_init_constructors_match_the_dsl_constructors():
    """Under a CPU MLIR context the runtime's constructors build the same
    pipeline objects as ``PipelineTmaUmma.create`` / ``PipelineUmmaAsync.create``
    with ``defer_sync=True`` (barrier arrays' stages, tx counts, arrive counts,
    op types, groups, layouts; masks, leader flag and CTA group of the same
    kinds) for the shapes the clustered kernels use; only the warp-0 init is
    left to ``init_pipeline_barriers``."""
    pytest.importorskip("cutlass")
    import cutlass
    from cutlass._mlir import ir as mlir_ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute
    from cutlass.pipeline import Agent
    from cutlass.pipeline import CooperativeGroup
    from cutlass.pipeline import PipelineTmaUmma
    from cutlass.pipeline import PipelineUmmaAsync

    from helion._compiler.cute import _tcgen05_prologue_runtime as rt

    ir: Any = mlir_ir

    def same_array(mine: Any, theirs: Any) -> None:
        assert mine.num_stages == theirs.num_stages
        assert mine.op_type is theirs.op_type
        assert mine.cg is theirs.cg
        assert mine.mbarrier_layout is theirs.mbarrier_layout
        assert mine.name == theirs.name
        assert type(mine.tx_count) is type(theirs.tx_count)
        if isinstance(theirs.tx_count, int):
            assert mine.tx_count == theirs.tx_count
        assert type(mine.arrive_count) is type(theirs.arrive_count)
        if isinstance(theirs.arrive_count, int):
            assert mine.arrive_count == theirs.arrive_count

    for cluster in ((2, 2, 1), (2, 1, 1), (1, 1, 1)):
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("pipelines", ([], []))
            with ir.InsertionPoint(function.add_entry_block()):
                cta_group = 2 if cluster[0] == 2 else 1
                cta_layout_vmnk = cute.tiled_divide(
                    cute.make_layout(cluster), (cute.make_layout(cta_group).shape,)
                )
                ab_bars = cute.make_ptr(
                    cutlass.Int64, 1024, cute.AddressSpace.smem, assumed_align=8
                )
                acc_bars = cute.make_ptr(
                    cutlass.Int64, 2048, cute.AddressSpace.smem, assumed_align=8
                )
                producer = CooperativeGroup(Agent.Thread, 1)
                consumer = CooperativeGroup(Agent.Thread, cutlass.Int32(cta_group))
                acc_consumer = CooperativeGroup(Agent.Thread, cutlass.Int32(8))
                tx_count = 26624
                theirs = PipelineTmaUmma.create(
                    num_stages=7,
                    producer_group=producer,
                    consumer_group=consumer,
                    tx_count=tx_count,
                    barrier_storage=ab_bars,
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
                mine = rt.make_tma_umma_pipeline(
                    num_stages=7,
                    producer_group=producer,
                    consumer_group=consumer,
                    tx_count=tx_count,
                    barrier_storage=ab_bars,
                    cta_layout_vmnk=cta_layout_vmnk,
                )
                same_array(mine.sync_object_full, theirs.sync_object_full)
                same_array(mine.sync_object_empty, theirs.sync_object_empty)
                assert mine.num_stages == theirs.num_stages == 7
                assert mine.cta_group is theirs.cta_group
                assert type(mine.producer_mask) is type(theirs.producer_mask)
                assert type(mine.consumer_mask) is type(theirs.consumer_mask)
                assert type(mine.is_leader_cta) is type(theirs.is_leader_cta)
                theirs_acc = PipelineUmmaAsync.create(
                    num_stages=2,
                    producer_group=producer,
                    consumer_group=acc_consumer,
                    barrier_storage=acc_bars,
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
                mine_acc = rt.make_umma_async_pipeline(
                    num_stages=2,
                    producer_group=producer,
                    consumer_group=acc_consumer,
                    barrier_storage=acc_bars,
                    cta_layout_vmnk=cta_layout_vmnk,
                )
                same_array(mine_acc.sync_object_full, theirs_acc.sync_object_full)
                same_array(mine_acc.sync_object_empty, theirs_acc.sync_object_empty)
                assert mine_acc.cta_group is theirs_acc.cta_group
                assert type(mine_acc.producer_mask) is type(theirs_acc.producer_mask)
                assert type(mine_acc.consumer_mask) is type(theirs_acc.consumer_mask)
                func.ReturnOp([])
            # The DSL constructors emitted the warp-0 inits; ours emitted none.
            text = str(module)
            assert text.count("mbarrier.init") == 2 * (7 + 2)
