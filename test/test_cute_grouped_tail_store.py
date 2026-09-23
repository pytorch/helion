"""CPU proof/codegen coverage for grouped stores after preserve-output folding."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test import test_cute_rank3_rhs_b_tma as grouped
from test.test_autotuner_heuristics import _grouped_worklist_bind_patches
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.cute_mma import _MmaSearchGraphView
from helion._compiler.cute.cute_mma import _tcgen05_tma_operand_is_aligned
from helion._compiler.cute.cute_mma import _trace_to_mma_operand
from helion._compiler.cute.cute_mma import (
    tcgen05_grouped_static_seed_has_common_k_proof,
)
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import skipUnlessBackends
from helion.language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.generate_ast import GenerateAST


@pytest.fixture(autouse=True)
def _target(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HELION_CUTE_MMA_IMPL", "tcgen05")
    monkeypatch.setattr(grouped, "DEVICE", torch.device("cpu"))
    with (
        _cpu_target(),
        _grouped_worklist_bind_patches(),
        patch(
            "helion._compiler.cute.tcgen05_config."
            "CuteTcgen05Config.per_cta_smem_capacity_bytes",
            return_value=232448,
        ),
    ):
        yield


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "k,block_k,ab_stages", [(16, 16, 2), (96, 32, 3), (192, 64, 3)]
)
@skipUnlessBackends(["cute"])
def test_folded_grouped_tail_store_keeps_native_seed_and_codegen(
    dtype: torch.dtype, k: int, block_k: int, ab_stages: int
) -> None:
    args = grouped._make_mn_tail_args(k=k, dtype=dtype)
    kernel = grouped._rank3_rhs_grouped_nt_with_mn_tails
    spec, raw, normalized = grouped._seed_configs(
        kernel, args, grouped.TCGEN05_GROUPED_MODE_STATIC
    )
    assert "cute_tcgen05_grouped_static_common_k" in spec.autotuner_heuristics
    assert len(raw) == len(normalized) == 1
    assert normalized[0]["block_sizes"] == [128, 64, block_k]
    assert normalized[0]["tcgen05_ab_stages"] == ab_stages
    code = grouped._code_for(kernel, args, helion.Config.from_dict(normalized[0]))
    plan = grouped._assert_group_scheduler(code)
    assert plan["m_tail_preserve"] is True
    assert plan["n_tail_preserve"] is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_folded_grouped_tail_store_keeps_dynamic_k_proof(dtype: torch.dtype) -> None:
    args = grouped._make_documented_mixed_k_args(dtype=dtype)
    kernel = grouped._rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes
    spec, raw, normalized = grouped._seed_configs(
        kernel, args, grouped.TCGEN05_GROUPED_MODE_DYNAMIC
    )
    assert "cute_tcgen05_grouped_dynamic_bk64" in spec.autotuner_heuristics
    assert len(raw) == len(normalized) == 1
    code = grouped._code_for(kernel, args, helion.Config.from_dict(normalized[0]))
    plan = grouped._assert_group_scheduler(code)
    assert plan["m_tail_preserve"] is True
    assert plan["n_tail_preserve"] is True
    assert plan["dynamic_ab_tensormaps"] is True


@pytest.mark.parametrize("axis", [0, 1, 3])
def test_grouped_tile_origin_keeps_role_and_cleanup_dependencies(axis: int) -> None:
    splitter = Tcgen05PersistentProgramIDs.__new__(Tcgen05PersistentProgramIDs)
    statements = ast.parse(
        "tile_offset_0 = pid_0 * block_size\n"
        f"tile_begin_{axis} = indices_{axis} - "
        f"(indices_{axis} - tile_offset_0)"
    ).body
    partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
        role_blocks_inline=[],
        role_blocks_extracted=[],
        shared_body_extracted=statements,
    )
    use = ast.parse(f"consume(tile_begin_{axis})").body
    with (
        patch.object(splitter, "_tcgen05_plan", return_value=None),
        patch.object(
            splitter, "_tcgen05_uses_grouped_static_persistent", return_value=True
        ),
    ):
        assert not splitter._tcgen05_shared_loop_has_meaningful_work(partition, [])
        splitter._assert_tcgen05_grouped_omit_shared_loop_safe(partition)
        # Omitting the shared copy cannot discard an origin read by a role:
        # extraction clones its complete coordinate dependency chain.
        assert splitter._role_local_dependency_stmts(statements, use) == statements
        assert splitter._tcgen05_shared_loop_has_meaningful_work(partition, use)
        assert splitter._tcgen05_shared_post_loop_dependencies(partition, use) == {
            f"tile_begin_{axis}"
        }


@pytest.mark.parametrize(
    "statement",
    [
        "tile_begin_0 = unknown(indices_0)",
        "tile_begin_0 = pointer.store(indices_0)",
        "tile_begin_0 = cute.arch.atomic_add(pointer, indices_0)",
        "tile_begin_0 = pipeline.consumer_wait(state)",
        "tile_begin_0[0] = indices_0",
        "tile_begin_other = indices_0 - (indices_0 - tile_offset_0)",
    ],
)
def test_grouped_tile_origin_omission_rejects_effects_and_other_writes(
    statement: str,
) -> None:
    splitter = Tcgen05PersistentProgramIDs.__new__(Tcgen05PersistentProgramIDs)
    partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
        role_blocks_inline=[],
        role_blocks_extracted=[],
        shared_body_extracted=ast.parse(statement).body,
    )
    with (
        patch.object(splitter, "_tcgen05_plan", return_value=None),
        patch.object(
            splitter, "_tcgen05_uses_grouped_static_persistent", return_value=True
        ),
    ):
        assert splitter._tcgen05_shared_loop_has_meaningful_work(partition, [])
        with pytest.raises(AssertionError, match="discard observable shared"):
            splitter._assert_tcgen05_grouped_omit_shared_loop_safe(partition)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("major", ["k", "n"])
@pytest.mark.parametrize(
    "padding,offset,aligned", [(0, 0, True), (1, 0, False), (8, 8, True), (0, 1, False)]
)
def test_grouped_operand_alignment_uses_physical_strides(
    dtype: torch.dtype, major: str, padding: int, offset: int, aligned: bool
) -> None:
    a, original_b, layout = grouped._make_full_args(dtype=dtype)
    groups, n, k = original_b.shape
    if major == "k":
        strides = (n * (k + padding), k + padding, 1)
    else:
        strides = (k * (n + padding), 1, n + padding)
    storage = torch.empty(groups * strides[0] + offset, dtype=dtype)
    b = storage.as_strided((groups, n, k), strides, offset)
    bound = grouped._rank3_rhs_grouped_nt._bind_isolated((a, b, layout))
    host = bound.host_function
    assert host is not None
    ir = host.device_ir
    product = next(
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    rhs = product.args[2]
    assert isinstance(rhs, torch.fx.Node)
    with bound.env, host:
        operand = _trace_to_mma_operand(
            rhs,
            role="rhs",
            allow_rank3_rhs_nt=True,
            cg=cast("GenerateAST", _MmaSearchGraphView(ir.graphs)),
            rank3_rhs_m_block_id=0,
            allow_rank3_rhs_mn_major=True,
        )
        assert operand is not None
        assert operand.source_fake.ndim == 3 and operand.logical_fake.ndim == 2
        assert _tcgen05_tma_operand_is_aligned(bound.env, operand) is aligned


@pytest.mark.parametrize("fold", [False, True])
def test_grouped_tail_proof_supports_both_preserve_output_forms(fold: bool) -> None:
    kernel = grouped._rank3_rhs_grouped_nt_with_mn_tails
    args = grouped._make_mn_tail_args(k=16)
    with (
        nullcontext()
        if fold
        else patch(
            "helion._compiler.cute.fold_noop_stores.fold_noop_stores", return_value=0
        )
    ):
        bound = kernel._bind_isolated(args)
    host = bound.host_function
    assert host is not None
    ir = host.device_ir
    assert tcgen05_grouped_static_seed_has_common_k_proof(
        bound.env, ir, bound.config_spec.matmul_facts[0]
    )
    stores = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    ]
    assert len(stores) == 1
    assert (len(stores[0].args) > 3 and stores[0].args[3] is not None) is fold


@pytest.mark.parametrize(
    "change", ["inverted_mask", "shared_mask", "different_coordinate", "output_dtype"]
)
def test_grouped_tail_proof_rejects_changed_mask_or_destination(change: str) -> None:
    bound = grouped._rank3_rhs_grouped_nt_with_mn_tails._bind_isolated(
        grouped._make_mn_tail_args(k=16)
    )
    host = bound.host_function
    assert host is not None
    ir = host.device_ir
    fact = bound.config_spec.matmul_facts[0]
    assert tcgen05_grouped_static_seed_has_common_k_proof(bound.env, ir, fact)
    store = next(
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    )
    mask = store.args[3]
    assert isinstance(mask, torch.fx.Node)
    with store.graph.inserting_before(store):
        if change == "inverted_mask":
            inverse = store.graph.call_function(
                torch.ops.aten.logical_not.default, (mask,)
            )
            inverse.meta = mask.meta.copy()
            store.args = (*store.args[:3], inverse)
        elif change == "shared_mask":
            store.graph.call_function(torch.ops.aten.clone.default, (mask,))
        elif change == "different_coordinate":
            indices = store.args[1]
            assert isinstance(indices, (list, tuple))
            store.args = (store.args[0], list(reversed(indices)), *store.args[2:])
        else:
            tensor = store.args[0]
            assert isinstance(tensor, torch.fx.Node)
            tensor.meta["val"] = torch.empty((256, 192), dtype=torch.float32)
    assert not tcgen05_grouped_static_seed_has_common_k_proof(bound.env, ir, fact)
