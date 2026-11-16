"""Grouped GEMM split-size / device-offset kernels, their inputs and the selected
config, shared by the grouped split-size tests (``test_cute_grouped_*``)."""

from __future__ import annotations

import itertools

import pytest
import torch

import helion
from helion._compat import requires_cuda_version
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
)
from helion._testing import DEVICE
import helion.language as hl


def _selected_config(
    block_k: int = 128,
    *,
    grouped_mode: bool = True,
    source_m_tile: int | None = None,
) -> helion.Config:
    config = helion.Config(
        block_sizes=[256, 128, block_k],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages={64: 7, 128: 3}[block_k],
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )
    if grouped_mode:
        config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = (
            TCGEN05_GROUPED_MODE_WORKLIST_NM
        )
        config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = (
            source_m_tile
            if source_m_tile is not None
            else TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
            if block_k == 128
            else TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
        )
    return config


@helion.kernel(backend="cute", static_shapes=True)
def _device_split_sizes_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    split_sizes: torch.Tensor,
) -> torch.Tensor:
    m_total, k = a_packed.shape
    groups, n, k2 = b_grouped.shape
    assert k == k2
    assert groups == 8
    assert split_sizes.size(0) == 8
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for group_tile, tile_m, tile_n in hl.tile(
        [groups, m_total, n],
        block_size=[1, block_m, block_n],
    ):
        group = group_tile.index.sum()
        group_m = split_sizes[group]
        s0 = split_sizes[0]
        s1 = split_sizes[1]
        s2 = split_sizes[2]
        s3 = split_sizes[3]
        s4 = split_sizes[4]
        s5 = split_sizes[5]
        s6 = split_sizes[6]
        # Prefix addition is commutative; use a deliberately noncanonical order
        # to ensure the semantic matcher does not depend on source ordering.
        group_start = (
            torch.where(group > 4, s4, 0)
            + torch.where(group > 0, s0, 0)
            + torch.where(group > 6, s6, 0)
            + torch.where(group > 1, s1, 0)
            + torch.where(group > 5, s5, 0)
            + torch.where(group > 2, s2, 0)
            + torch.where(group > 3, s3, 0)
        )
        local_m = tile_m.index
        row_index = group_start + local_m
        valid_rows = local_m < group_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _device_offsets_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    m_total, k = a_packed.shape
    groups, n, k2 = b_grouped.shape
    assert k == k2
    assert groups == 8
    assert offsets.size(0) == 9
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for group_tile, tile_m, tile_n in hl.tile(
        [groups, m_total, n],
        block_size=[1, block_m, block_n],
    ):
        group = group_tile.index.sum()
        group_start = offsets[group]
        # Repeat the start load to ensure recognition follows tensor/index
        # provenance rather than requiring one shared FX load node.
        group_m = offsets[group + 1] - offsets[group]
        local_m = tile_m.index
        row_index = group_start + local_m
        valid_rows = local_m < group_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _make_device_split_sizes_args(
    m_sizes: tuple[int, ...],
    *,
    n: int = 128,
    k: int = 128,
    split_dtype: torch.dtype = torch.int32,
    split_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(m_sizes) >= 2
    assert split_stride > 0
    a_packed = torch.randn(
        (sum(m_sizes), k),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    b_grouped = torch.randn(
        (len(m_sizes), n, k),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    split_storage = torch.empty(
        len(m_sizes) * split_stride,
        device=DEVICE,
        dtype=split_dtype,
    )
    split_sizes = split_storage[::split_stride]
    split_sizes.copy_(torch.tensor(m_sizes, device=DEVICE, dtype=split_dtype))
    return a_packed, b_grouped, split_sizes


def _make_device_offsets_args(
    m_sizes: tuple[int, ...],
    *,
    n: int = 128,
    k: int = 128,
    offsets_dtype: torch.dtype = torch.int32,
    offsets_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_packed, b_grouped, _split_sizes = _make_device_split_sizes_args(
        m_sizes,
        n=n,
        k=k,
        split_dtype=offsets_dtype,
    )
    offsets_storage = torch.empty(
        (len(m_sizes) + 1) * offsets_stride,
        device=DEVICE,
        dtype=offsets_dtype,
    )
    offsets = offsets_storage[::offsets_stride]
    offset_values = [0]
    for size in m_sizes:
        offset_values.append(offset_values[-1] + size)
    offsets.copy_(torch.tensor(offset_values, device=DEVICE, dtype=offsets_dtype))
    return a_packed, b_grouped, offsets


def _assert_device_offsets_output(
    out: torch.Tensor,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a_packed, b_grouped, offsets = args
    offset_values = offsets.cpu().tolist()
    for group, (start, end) in enumerate(itertools.pairwise(offset_values)):
        expected = (a_packed[start:end].float() @ b_grouped[group].float().T).to(
            out.dtype
        )
        torch.testing.assert_close(
            out[start:end],
            expected,
            rtol=3e-2,
            atol=3e-2,
        )


def _require_codegen_cuda() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("tcgen05 grouped split-size codegen needs CUDA fake inputs")


def _require_runtime_cuda13_sm100() -> None:
    _require_codegen_cuda()
    if not requires_cuda_version("13"):
        pytest.skip("tcgen05 grouped split-size runtime needs CUDA >= 13")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")
