from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_batched_aux_tma import _code
from .test_cute_batched_aux_tma import _config
from .test_cute_batched_aux_tma import _cpu_codegen
from .test_cute_batched_aux_tma import _inputs
from .test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from .test_cute_grouped_gemm_split_sizes import _selected_config
from .test_cute_scheduler_mailbox import _plain_matmul
from helion import exc
from helion._compiler.cute import tcgen05_constants
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _literal_int(node: ast.expr) -> int:
    if isinstance(node, ast.Call):
        assert ast.unparse(node.func) == "cutlass.Int32"
        assert len(node.args) == 1 and not node.keywords
        node = node.args[0]
    assert isinstance(node, ast.Constant) and type(node.value) is int
    return node.value


def _assert_full_empty_storage(code: str) -> dict[str, int]:
    """Check each pipeline object against its declared storage, not padding."""
    tree = ast.parse(code)
    allocations = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.arch.alloc_smem"
            and ast.unparse(node.value.args[0]) == "cutlass.Int64"
        ):
            continue
        allocations[node.targets[0].id] = _literal_int(node.value.args[1])
    pipelines = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        constructor = ast.unparse(node.func)
        if constructor not in {
            f"cutlass.pipeline.{kind}.create"
            for kind in (
                "PipelineTmaUmma",
                "PipelineUmmaAsync",
                "PipelineAsync",
                "PipelineTmaAsync",
            )
        }:
            continue
        kwargs = {keyword.arg: keyword.value for keyword in node.keywords}
        storage = kwargs["barrier_storage"]
        assert isinstance(storage, ast.Name)
        stages = _literal_int(kwargs["num_stages"])
        assert allocations[storage.id] >= 2 * stages, (
            storage.id,
            allocations[storage.id],
            stages,
        )
        pipelines[storage.id] = stages
    assert pipelines
    return pipelines


def test_storage_check_rejects_old_underallocation_without_padding_credit() -> None:
    code = """
bars = cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(2))
unrelated_padding = cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(16))
pipe = cutlass.pipeline.PipelineTmaUmma.create(num_stages=2, barrier_storage=bars)
"""
    with pytest.raises(AssertionError):
        _assert_full_empty_storage(code)


@pytest.mark.parametrize("stages", [1, 2, 3])
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_ab_full_empty_allocation_all_stages_and_cta_groups(
    stages: int, cluster_m: int
) -> None:
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 1024), dtype=torch.bfloat16),
    )
    config = _config(
        block_sizes=[128 * cluster_m, 256, 64],
        tcgen05_cluster_m=cluster_m,
        tcgen05_ab_stages=stages,
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
    )
    with _cpu_codegen():
        code = _plain_matmul._bind_isolated(args).to_code(config)
    pipelines = _assert_full_empty_storage(code)
    assert pipelines["tcgen05_ab_pipeline_mbars"] == stages


@pytest.mark.parametrize("wait_mode", ["normal", "warp_leader"])
@pytest.mark.parametrize("sched_stages", [1, 2])
def test_clc_full_empty_allocations(wait_mode: str, sched_stages: int) -> None:
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 1024), dtype=torch.bfloat16),
    )
    config = _config(
        block_sizes=[256, 256, 128],
        tcgen05_cluster_m=2,
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
        tcgen05_persistence_model="clc_persistent",
        tcgen05_sched_stage_count=sched_stages,
        tcgen05_sched_consumer_wait_mode=wait_mode,
    )
    with _cpu_codegen():
        code = _plain_matmul._bind_isolated(args).to_code(config)
    pipelines = _assert_full_empty_storage(code)
    assert pipelines["tcgen05_sched_pipeline_mbars"] == sched_stages
    assert pipelines["tcgen05_ab_pipeline_mbars"] == 2


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_auxiliary_full_empty_allocations(stages: int) -> None:
    if stages == 3:
        # Productive AUX currently excludes this depth independently of the
        # AB storage correction. Preserve that public admission restriction.
        with pytest.raises(exc.BackendUnsupported, match="incompatible"):
            _code(
                _inputs(), _config(block_sizes=[128, 128, 64], tcgen05_ab_stages=stages)
            )
        return
    code = _code(
        _inputs(), _config(block_sizes=[128, 128, 64], tcgen05_ab_stages=stages)
    )
    pipelines = _assert_full_empty_storage(code)
    assert pipelines["tcgen05_ab_pipeline_mbars"] == stages
    assert "tcgen05_aux_pipeline_mbars" in pipelines


@pytest.mark.parametrize("block_k", [64, 128])
def test_grouped_full_empty_allocations(block_k: int) -> None:
    args = (
        torch.empty((2048, 128), dtype=torch.bfloat16),
        torch.empty((8, 224, 128), dtype=torch.bfloat16),
        torch.tensor((0, 1, 127, 224, 256, 449, 0, 991), dtype=torch.int32),
    )
    with _cpu_codegen():
        code = _device_split_sizes_kernel._bind_isolated(args).to_code(
            _selected_config(block_k)
        )
    pipelines = _assert_full_empty_storage(code)
    assert pipelines["tcgen05_ab_pipeline_mbars"] == (7 if block_k == 64 else 3)


@pytest.mark.parametrize("stages", [1, 2, 3, 7, 12])
def test_grouped_accounting_charges_both_barrier_arrays(stages: int) -> None:
    append = tcgen05_constants._append_aligned_tcgen05_smem
    with patch.object(
        tcgen05_constants, "_append_aligned_tcgen05_smem", wraps=append
    ) as allocations:
        tcgen05_constants.tcgen05_grouped_worklist_smem_bytes(
            group_count=8,
            device_split_sizes=True,
            sched_stage_count=1,
            bm=256,
            bn=224,
            bk=64,
            dtype_bytes=2,
            ab_stages=stages,
            acc_stages=2,
            c_stages=2,
            cluster_m=2,
        )
    # This precedes the output TensorMap and C ring. Their 128/1024-byte
    # alignment can hide a wrong AB size in the final total at common depths.
    assert allocations.call_args_list[-3].args[1:] == (2 * stages * 8, 8)


def test_existing_grouped_cap_boundary_is_preserved() -> None:
    def footprint(group_count: int) -> int:
        return tcgen05_constants.tcgen05_grouped_worklist_smem_bytes(
            group_count=group_count,
            device_split_sizes=True,
            sched_stage_count=1,
            bm=256,
            bn=224,
            bk=64,
            dtype_bytes=2,
            ab_stages=7,
            acc_stages=2,
            c_stages=2,
            cluster_m=2,
        )

    assert footprint(8) == 227 * 1024
    assert footprint(51) > 227 * 1024
