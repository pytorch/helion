from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grouped_fp32_output(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    groups, n, _ = b.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64, 128)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for group_tile, rows, columns in hl.tile(
        [groups, m, n], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < length
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, b[group, columns, inner].T)
        hl.store(out, [row_index, columns], acc, extra_mask=valid[:, None])
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_grouped_fp32_output_uses_worklist_and_charges_wider_ring(
    dtype: torch.dtype,
) -> None:
    args = (
        torch.empty((640, 128), dtype=dtype),
        torch.empty((4, 256, 128), dtype=dtype),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        facts = bound.config_spec._cute_tcgen05_config.grouped_worklist_smem_facts
        assert facts is not None and facts.output_dtype_bytes == 4
        config = helion.Config(
            block_sizes=[256, 128, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=5,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_grouped_mode="worklist_nm",
            tcgen05_grouped_worklist_source_m_tile=32,
        )
        source = bound.to_code(config)
    ast.parse(source)
    assert "tcgen05_grouped_static_persistent" in source
    assert "cutlass.Float32" in source
    assert "tcgen05_grouped_d_tensormap" in source


def test_fp32_grouped_output_ring_exceeds_the_old_bf16_capacity_boundary() -> None:
    arguments = {
        "group_count": 8,
        "device_split_sizes": True,
        "sched_stage_count": 1,
        "bm": 256,
        "bn": 224,
        "bk": 64,
        "dtype_bytes": 2,
        "ab_stages": 7,
        "acc_stages": 2,
        "c_stages": 2,
        "cluster_m": 2,
    }
    bf16 = tcgen05_grouped_worklist_smem_bytes(**arguments)
    fp32 = tcgen05_grouped_worklist_smem_bytes(**arguments, output_dtype_bytes=4)
    assert bf16 == 227 * 1024
    assert fp32 == bf16 + 16 * 1024
