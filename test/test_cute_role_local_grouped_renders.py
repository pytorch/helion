"""CPU render pins for the grouped tcgen05 role-local persistent-loop families
(the plain-GEMM families and the shared binding live in
``test_cute_role_local_renders``).

One explicit ``helion.Config`` per family, each checked for the marker that
proves it reached its builder in
``helion/_compiler/cute/tcgen05/scheduler_pids.py`` and journaled whole in
``test_cute_role_local_grouped_renders.expected_cute``:

* F3 ``_build_generic_grouped_static_role_local_while`` and F4
  ``_build_specialized_grouped_static_role_local_while`` (static problem
  signature, generic fallback as the ``else``).
* F5 ``_build_grouped_worklist_nm_role_local_while``: the runtime tile table,
  the scheduler mailbox, runtime CLC record ids, and the dense ``local_full``
  branch; P2 ``_build_grouped_worklist_nm_scheduler_warp_role_local_while``:
  the M,N panel raster and the dense full-coverage loop.

The kernels and config builders belong to the tests that own those paths
(``test_cute_rank3_rhs_b_tma``, ``test_cute_deepgemm_selected``,
``test_cute_grouped_large_device_seeds``, ``examples.grouped_gemm``); their
argument builders allocate on ``DEVICE``, so the CPU copies live here.
"""

from __future__ import annotations

import pytest
import torch

from test.test_cute_deepgemm_selected import _selected_config
from test.test_cute_deepgemm_selected import _selected_kernel
from test.test_cute_grouped_large_device_seeds import _grouped_offsets_k64
from test.test_cute_rank3_rhs_b_tma import _dynamic_bk64_config
from test.test_cute_rank3_rhs_b_tma import _grouped_config
from test.test_cute_rank3_rhs_b_tma import _rank3_rhs_grouped_nt
from test.test_cute_rank3_rhs_b_tma import (
    _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
)
from test.test_cute_role_local_renders import CLC_BIDX
from test.test_cute_role_local_renders import CLC_PEER_BROADCAST
from test.test_cute_role_local_renders import F1_TILE_SCHEDULER
from test.test_cute_role_local_renders import MAILBOX_VALID
from test.test_cute_role_local_renders import RenderPins
from test.test_cute_role_local_renders import _render

import helion
from helion._compiler.cute.autotune.heuristics.cute import (
    _tcgen05_grouped_worklist_config,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_DENSE
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
)
from helion._testing import import_example
from helion._testing import matchesBackends
from helion._testing import onlyBackends

grouped_gemm_jagged = import_example("grouped_gemm").grouped_gemm_jagged

if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")

GROUP_TILE_SCHEDULER = "StaticPersistentGroupTileScheduler.create"
PANEL_LOCAL_INDEX = "tcgen05_grouped_panel_local_index"
P2_DENSE_INDEX = "tcgen05_grouped_dense_index"
LOCAL_FULL_DENSE_INDEX = "tcgen05_role_local_0_dense_index"
STATIC_LINEAR_INDEX = "_static_linear_idx"
STATIC_GROUP_QUOTA = "tcgen05_grouped_static_quota"
RUNTIME_DIRECT_LINEAR_INDEX = "_runtime_direct_linear_idx"
RUNTIME_DIRECT_GRID_STRIDE = "_runtime_direct_grid_stride"

_jagged_grouped_gemm = helion.kernel(
    grouped_gemm_jagged.fn,
    backend="cute",
    static_shapes=False,
    autotune_effort="none",
    cute_segmented_matmul_tiling=True,
)


def _rank3_grouped_args(
    groups: int = 4, m_per_group: int = 128, n: int = 256, k: int = 128
) -> tuple[torch.Tensor, ...]:
    """``test_cute_rank3_rhs_b_tma._make_full_args`` on the CPU."""
    return (
        torch.empty((groups * m_per_group, k), dtype=torch.bfloat16),
        torch.empty((groups, n, k), dtype=torch.bfloat16),
        torch.arange(groups, dtype=torch.int64).repeat_interleave(m_per_group),
    )


def _rank3_mixed_k_args() -> tuple[torch.Tensor, ...]:
    """``test_cute_rank3_rhs_b_tma._make_documented_mixed_k_args`` on the CPU:
    four groups of 128 padded rows with per-group M, N and K extents."""
    k_values = (32, 1536, 16, 16)
    m_values = (128, 16, 128, 16)
    n_values = (64, 128, 64, 80)
    groups = len(k_values)
    padded_m = groups * 128
    max_n = max(n_values)
    max_k = max(k_values)
    layout = torch.full((padded_m,), -1, dtype=torch.int32)
    for group, m_size in enumerate(m_values):
        layout[group * 128 : group * 128 + m_size] = group
    return (
        torch.empty((padded_m, max_k), dtype=torch.float16),
        torch.empty((groups, max_n, max_k), dtype=torch.float16),
        layout,
        torch.tensor(n_values, dtype=torch.int32),
        torch.tensor(k_values, dtype=torch.int32),
        torch.empty((padded_m, max_n), dtype=torch.float16),
    )


def _worklist_args(
    m_sizes: tuple[int, ...] = (1, 127, 224, 256), *, n: int = 224, k: int = 128
) -> tuple[torch.Tensor, ...]:
    """``test_cute_deepgemm_selected._make_args`` on the CPU: packed rows
    aligned to the large source tile and the ``[group, start, valid, store]``
    work-tile metadata."""
    tile = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
    rows: list[list[int]] = []
    cursor = 0
    for group, actual_m in enumerate(m_sizes):
        aligned_m = (actual_m + tile - 1) // tile * tile
        rows.append([group, cursor, actual_m, aligned_m])
        cursor += aligned_m
    return (
        torch.empty((cursor, k), dtype=torch.bfloat16),
        torch.empty((len(m_sizes), n, k), dtype=torch.bfloat16),
        torch.tensor(rows, dtype=torch.int32),
    )


def _jagged_args(m: int = 640, n: int = 128, k: int = 128) -> tuple[torch.Tensor, ...]:
    """``test_cute_grouped_full_coverage._inputs``: four groups, two of them
    one row long."""
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.tensor([0, 1, m // 2, m - 1, m], dtype=torch.int32),
    )


def _dense_config(mode: str) -> helion.Config:
    config = _tcgen05_grouped_worklist_config(
        32, 64, 2, 240, runtime_direct=False, l2_swizzle_size=1
    )
    config.config[TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY] = mode
    return config


@onlyBackends(["cute"])
class TestRoleLocalGroupedRenders(RenderPins):
    # F3 / F4: grouped static persistent.
    def test_f3_generic_grouped_static(self) -> None:
        code = _render(_rank3_rhs_grouped_nt, _rank3_grouped_args(), _grouped_config())
        self._pin(
            code,
            GROUP_TILE_SCHEDULER,
            ".group_search_result",
            absent=(F1_TILE_SCHEDULER, STATIC_LINEAR_INDEX, MAILBOX_VALID),
        )

    def test_f4_specialized_grouped_static(self) -> None:
        # Four static problem shapes: a per-group loop ladder under the quota
        # test, the generic group scheduler as the fallback branch.
        config = _dynamic_bk64_config(direct=True)
        config.config[TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY] = [
            4,
            *(128, 64, 32),
            *(16, 128, 1536),
            *(128, 64, 16),
            *(16, 80, 16),
        ]
        code = _render(
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _rank3_mixed_k_args(),
            config,
        )
        self._pin(
            code,
            STATIC_LINEAR_INDEX,
            STATIC_GROUP_QUOTA,
            "cute.arch.grid_dim()[2] >= cutlass.Int32(4)",
            GROUP_TILE_SCHEDULER,
        )

    # F5: the selected N,M worklist consumers.
    def test_f5_runtime_tile_table(self) -> None:
        code = _render(
            _selected_kernel, _worklist_args(), _selected_config(runtime_direct=True)
        )
        self._pin(
            code,
            RUNTIME_DIRECT_LINEAR_INDEX,
            RUNTIME_DIRECT_GRID_STRIDE,
            absent=(GROUP_TILE_SCHEDULER, MAILBOX_VALID, CLC_BIDX),
        )

    def test_f5_scheduler_mailbox(self) -> None:
        code = _render(
            _selected_kernel, _worklist_args(), _selected_config(runtime_direct=False)
        )
        self._pin(
            code,
            MAILBOX_VALID,
            GROUP_TILE_SCHEDULER,
            absent=(RUNTIME_DIRECT_LINEAR_INDEX, PANEL_LOCAL_INDEX),
        )

    def test_f5_runtime_clc(self) -> None:
        code = _render(
            _selected_kernel,
            _worklist_args((1, 127, 256, 256), n=512, k=128),
            _selected_config(
                block_k=64,
                source_m_tile=256,
                ab_stages=6,
                consumer_regs=256,
                runtime_direct=True,
                clc=True,
            ),
        )
        self._pin(
            code,
            RUNTIME_DIRECT_LINEAR_INDEX,
            CLC_BIDX,
            CLC_PEER_BROADCAST,
            MAILBOX_VALID,
            absent=(RUNTIME_DIRECT_GRID_STRIDE,),
        )

    # P2: the group-search scheduler warp's rasters.
    def test_p2_panel_raster(self) -> None:
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
        code = _render(_grouped_offsets_k64, args, config)
        self._pin(code, PANEL_LOCAL_INDEX, GROUP_TILE_SCHEDULER, MAILBOX_VALID)

    def test_dense_full_coverage(self) -> None:
        # P2's dense full-coverage loop: the scheduler warp publishes every
        # tile of the index domain.
        code = _render(
            _jagged_grouped_gemm,
            _jagged_args(),
            _dense_config(TCGEN05_GROUPED_FULL_COVERAGE_DENSE),
        )
        self._pin(
            code,
            P2_DENSE_INDEX,
            GROUP_TILE_SCHEDULER,
            MAILBOX_VALID,
            absent=(LOCAL_FULL_DENSE_INDEX,),
        )

    def test_dense_local_full_coverage(self) -> None:
        # F5's ``local_full`` branch: every consumer walks the dense index
        # domain itself under the full-coverage predicate.
        code = _render(
            _jagged_grouped_gemm,
            _jagged_args(),
            _dense_config(TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL),
        )
        self._pin(
            code,
            LOCAL_FULL_DENSE_INDEX,
            GROUP_TILE_SCHEDULER,
            MAILBOX_VALID,
            absent=(P2_DENSE_INDEX,),
        )
