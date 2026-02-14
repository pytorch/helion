"""GPU architecture detection and selection utilities for CuteDSL backend.

Provides helpers to detect the CUDA compute capability and select
architecture-specific strategies for copy atoms, MMA atoms, and
memory operations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# Compute capability constants
SM80 = 80  # Ampere
SM86 = 86  # Ampere (GA10x)
SM89 = 89  # Ada Lovelace
SM90 = 90  # Hopper
SM100 = 100  # Blackwell


@lru_cache(maxsize=8)
def get_cuda_arch(device: torch.device | None = None) -> int:
    """Get the SM version as an integer (e.g. 80, 90, 100).

    Args:
        device: CUDA device. Defaults to ``torch.cuda.current_device()``.

    Returns:
        Integer SM version (major * 10 + minor).
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def supports_cp_async(arch: int | None = None) -> bool:
    """Check if the architecture supports cp.async (SM80+).

    Args:
        arch: SM version. If None, detects from current device.
    """
    if arch is None:
        arch = get_cuda_arch()
    return arch >= SM80


def supports_tma(arch: int | None = None) -> bool:
    """Check if the architecture supports TMA (SM90+).

    Args:
        arch: SM version. If None, detects from current device.
    """
    if arch is None:
        arch = get_cuda_arch()
    return arch >= SM90


def supports_tcgen05(arch: int | None = None) -> bool:
    """Check if the architecture supports tcgen05 (SM100+).

    Args:
        arch: SM version. If None, detects from current device.
    """
    if arch is None:
        arch = get_cuda_arch()
    return arch >= SM100


def select_copy_strategy(arch: int | None = None) -> str:
    """Select the optimal GMEM->SMEM copy strategy for the architecture.

    Returns:
        One of ``"cp_async"``, ``"tma"``, or ``"tcgen05"``.
    """
    if arch is None:
        arch = get_cuda_arch()
    if arch >= SM100:
        return "tcgen05"
    if arch >= SM90:
        return "tma"
    return "cp_async"


def select_mma_strategy(arch: int | None = None) -> str:
    """Select the optimal MMA strategy for the architecture.

    Returns:
        One of ``"warp_mma"``, ``"wgmma"``, or ``"tcgen05_mma"``.
    """
    if arch is None:
        arch = get_cuda_arch()
    if arch >= SM100:
        return "tcgen05_mma"
    if arch >= SM90:
        return "wgmma"
    return "warp_mma"


def get_max_smem_per_block(arch: int | None = None) -> int:
    """Get maximum shared memory per block in bytes.

    Args:
        arch: SM version. If None, detects from current device.

    Returns:
        Maximum shared memory in bytes.
    """
    if arch is None:
        arch = get_cuda_arch()
    # Conservative defaults per architecture family
    if arch >= SM100:
        return 232 * 1024  # 232 KB for Blackwell
    if arch >= SM90:
        return 228 * 1024  # 228 KB for Hopper
    if arch >= SM80:
        return 164 * 1024  # 164 KB for Ampere (with opt-in)
    return 48 * 1024  # Default for older architectures


def supports_warpgroup_mma(arch: int | None = None) -> bool:
    """Check if the architecture supports warpgroup MMA (SM90+).

    Warpgroup MMA uses 128-thread groups (4 warps) for larger MMA tiles.

    Args:
        arch: SM version. If None, detects from current device.
    """
    if arch is None:
        arch = get_cuda_arch()
    return arch >= SM90


def supports_tmem(arch: int | None = None) -> bool:
    """Check if the architecture supports Tensor Memory (SM100+).

    TMEM is a special on-chip memory for Blackwell tcgen05 MMA operands.

    Args:
        arch: SM version. If None, detects from current device.
    """
    if arch is None:
        arch = get_cuda_arch()
    return arch >= SM100


def get_warpgroup_size(arch: int | None = None) -> int:
    """Get the warpgroup size (number of threads per warpgroup).

    SM90+ uses warpgroups of 128 threads (4 warps).
    Earlier architectures do not have warpgroups.

    Args:
        arch: SM version. If None, detects from current device.

    Returns:
        128 for SM90+, 32 for earlier architectures (single warp).
    """
    if arch is None:
        arch = get_cuda_arch()
    return 128 if arch >= SM90 else 32


@lru_cache(maxsize=8)
def get_num_sms(device: torch.device | None = None) -> int:
    """Get the number of streaming multiprocessors on the device.

    Args:
        device: CUDA device. Defaults to current device.

    Returns:
        Number of SMs.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_properties(device).multi_processor_count


@lru_cache(maxsize=8)
def get_l2_cache_size(device: torch.device | None = None) -> int:
    """Get the L2 cache size in bytes.

    Args:
        device: CUDA device. Defaults to current device.

    Returns:
        L2 cache size in bytes.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_properties(device).L2_cache_size


def get_copy_bits(dtype_width: int) -> int:
    """Get optimal copy width in bits for the given dtype width.

    Selects 128-bit copies when possible for maximum memory bandwidth.

    Args:
        dtype_width: Width of the element type in bits (e.g. 16 for fp16).

    Returns:
        Copy width in bits (128, 64, or 32).
    """
    if dtype_width <= 0:
        raise ValueError(f"dtype_width must be positive, got {dtype_width}")
    # Use 128-bit copies when possible
    if 128 % dtype_width == 0:
        return 128
    if 64 % dtype_width == 0:
        return 64
    return 32


def select_num_stages(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_bytes: int,
    num_warps: int,
    arch: int | None = None,
) -> int:
    """Select the optimal number of pipeline stages based on SMEM budget.

    Chooses the largest num_stages such that SMEM usage does not exceed
    the per-block limit, capped at 4 for Ampere and 8 for Hopper+.

    Args:
        tile_m: M tile dimension.
        tile_n: N tile dimension.
        tile_k: K tile dimension.
        dtype_bytes: Element size in bytes.
        num_warps: Warps per block (unused currently, reserved for occupancy).
        arch: SM version. If None, detects from current device.

    Returns:
        Optimal number of pipeline stages (1, 2, 3, 4, or up to 8 for SM90+).
    """
    if arch is None:
        arch = get_cuda_arch()

    max_smem = get_max_smem_per_block(arch)
    # Leave some headroom for other SMEM usage (reductions, etc.)
    usable_smem = int(max_smem * 0.9)

    # SMEM per stage: A tile + B tile
    smem_per_stage = (tile_m * tile_k + tile_k * tile_n) * dtype_bytes

    if smem_per_stage == 0:
        return 1

    max_stages_by_smem = usable_smem // smem_per_stage

    # Architecture-dependent caps
    max_stages_by_arch = 8 if arch >= SM90 else 4

    stages = min(max_stages_by_smem, max_stages_by_arch)
    return max(1, stages)


def compute_occupancy_smem(
    smem_bytes: int,
    arch: int | None = None,
) -> int:
    """Estimate max blocks per SM given SMEM usage.

    Args:
        smem_bytes: Shared memory usage per block in bytes.
        arch: SM version. If None, detects from current device.

    Returns:
        Estimated max concurrent blocks per SM.
    """
    if arch is None:
        arch = get_cuda_arch()
    max_smem = get_max_smem_per_block(arch)
    if smem_bytes <= 0:
        return 32  # Max blocks per SM (theoretical limit)
    return max(1, max_smem // smem_bytes)
