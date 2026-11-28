"""CPU-only grouped worklist code generation and legality checks."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import TestCase
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_mma_support
from helion._testing.cute_codegen import tcgen05_target
from helion._testing.cute_grouped_gemm import _grouped_offsets_k64
from helion._testing.cute_grouped_gemm import _unsupported_k_loop
from helion._testing.cute_grouped_gemm import grouped_ab_k_args
from helion._testing.cute_grouped_gemm import grouped_ab_k_config
from helion._testing.cute_grouped_gemm import grouped_d_shift_args
from helion._testing.cute_grouped_gemm import grouped_d_shift_config
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("variant", [1, 2, 3, 5])
def test_noncanonical_k_loops_cannot_request_physical128(variant: int) -> None:
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        args = (*grouped_ab_k_args(), variant)
        bound = cpu_bind(_unsupported_k_loop, args)
        assert not bound.config_spec._cute_tcgen05_config.grouped_ab_k_fusion_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(grouped_ab_k_config(2))


@pytest.mark.parametrize("variant", [1, 2, 3, 5])
def test_noncanonical_reductions_cannot_request_bounded_d_shift(variant: int) -> None:
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        args = (*grouped_d_shift_args("fp32", torch.int32), variant)
        bound = cpu_bind(_unsupported_k_loop, args)
        assert not bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(grouped_d_shift_config("mn"))


class TestRoleLocalGroupedRenders(TestCase):
    def test_p2_panel_raster(self) -> None:
        pytest.importorskip("cutlass.cute")
        # The MN orientation with an L2 swizzle: the scheduler warp walks M,N
        # panels and publishes the panel-local index.
        config = helion.Config(
            block_sizes=[256, 128, 64],
            l2_groupings=[1],
            loop_orders=[[0, 1, 2]],
            num_stages=7,
            num_warps=8,
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=13,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_consumer_regs=256,
            tcgen05_grouped_mode="worklist_nm",
            tcgen05_grouped_worklist_orientation="mn",
            tcgen05_grouped_worklist_source_m_tile=128,
            tcgen05_grouped_worklist_mma_n=128,
            tcgen05_grouped_d_shift=True,
            tcgen05_grouped_full_allocation_a=True,
            tcgen05_l2_swizzle_size=4,
        )
        args = (
            torch.empty((2048, 2048), dtype=torch.bfloat16),
            torch.empty((16, 1536, 2048), dtype=torch.bfloat16),
            torch.arange(17, dtype=torch.int32) * 128,
        )
        with cpu_render_target(mma_support=tcgen05_mma_support()):
            bound = cpu_bind(_grouped_offsets_k64, args)
            code = bound.to_code(config, output_origin_lines=False)
        self.assertIn("tcgen05_grouped_panel_local_index", code)
        self.assertIn("StaticPersistentGroupTileScheduler.create", code)
        self.assertIn("tcgen05_role_local_0_valid", code)
        self.assertExpectedJournal(code)
