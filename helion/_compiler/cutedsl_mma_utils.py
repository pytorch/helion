"""CuteDSL MMA (Matrix Multiply-Accumulate) utility code generation.

Provides parameterized helpers that select SMEM layouts, MMA atoms,
and copy atoms based on ``(tile_size, dtype, arch)``.  All selections
are deterministic — no pattern matching.

Reference patterns come from ``flash_attn/cute/ampere_helpers.py`` and
``flash_attn/cute/flash_fwd.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .cutedsl_arch_utils import SM80
from .cutedsl_arch_utils import SM90
from .cutedsl_arch_utils import SM100
from .cutedsl_arch_utils import get_copy_bits

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# SMEM layout atom selection (bank-conflict-free swizzled layouts)
# ---------------------------------------------------------------------------

def get_swizzle_params(
    k_dim: int,
    dtype_bytes: int,
) -> tuple[int, int, int]:
    """Compute swizzle parameters for a bank-conflict-free SMEM layout.

    Follows the pattern from ``flash_attn/cute/ampere_helpers.py``
    ``get_smem_layout_atom()``.

    Args:
        k_dim: Size of the contiguous (innermost) SMEM dimension.
        dtype_bytes: Size of each element in bytes.

    Returns:
        ``(swizzle_bits, swizzle_base, smem_k_block_size)`` tuple.
    """
    bytes_per_row = k_dim * dtype_bytes

    # Select swizzle block size (128/64/32/16 bytes based on alignment)
    if bytes_per_row % 128 == 0:
        smem_k_block = 128 // dtype_bytes
    elif bytes_per_row % 64 == 0:
        smem_k_block = 64 // dtype_bytes
    elif bytes_per_row % 32 == 0:
        smem_k_block = 32 // dtype_bytes
    else:
        smem_k_block = 16 // dtype_bytes

    # Swizzle bits based on block size (in elements)
    smem_k_bytes = smem_k_block * dtype_bytes
    if smem_k_bytes == 128:
        swizzle_bits = 4
    elif smem_k_bytes == 64:
        swizzle_bits = 3
    elif smem_k_bytes == 32:
        swizzle_bits = 2
    else:
        swizzle_bits = 1

    # Swizzle base depends on element size
    if dtype_bytes == 4:
        swizzle_base = 2
    elif dtype_bytes == 2:
        swizzle_base = 3
    else:
        swizzle_base = 4

    return swizzle_bits, swizzle_base, smem_k_block


def emit_smem_layout_atom(
    k_dim: int,
    dtype_bytes: int,
) -> str:
    """Emit CuteDSL code for creating a swizzled SMEM layout atom.

    Args:
        k_dim: Size of the K dimension for the SMEM tile.
        dtype_bytes: Size of each element in bytes.

    Returns:
        CuteDSL code string that creates a ``ComposedLayout``.
    """
    swizzle_bits, swizzle_base, smem_k_block = get_swizzle_params(k_dim, dtype_bytes)

    # Select M block size for the layout atom
    if k_dim % 32 == 0:
        m_block = 8
    else:
        m_block = 16

    return (
        f"cute.make_composed_layout("
        f"cute.make_swizzle({swizzle_bits}, {swizzle_base}, {swizzle_base}), "
        f"0, "
        f"cute.make_ordered_layout(({m_block}, {smem_k_block}), order=(1, 0)))"
    )


# ---------------------------------------------------------------------------
# MMA atom selection
# ---------------------------------------------------------------------------

def get_mma_shape(
    dtype_bytes: int,
    arch: int,
) -> tuple[int, int, int]:
    """Get the base MMA atom shape (M, N, K) for the given dtype and arch.

    Args:
        dtype_bytes: Element size in bytes (1 for int8/fp8, 2 for fp16/bf16, 4 for fp32).
        arch: SM version (80, 90, 100).

    Returns:
        ``(mma_m, mma_n, mma_k)`` tuple.
    """
    if arch >= SM90:
        # Hopper wgmma: larger tiles
        if dtype_bytes <= 2:
            return (64, 16, 16)
        return (64, 16, 8)
    # Ampere warp MMA
    if dtype_bytes <= 2:
        return (16, 8, 16)
    return (16, 8, 8)


def emit_tiled_mma(
    dtype_str: str,
    acc_dtype_str: str,
    num_warps: int,
    arch: int,
    dtype_bytes: int = 2,
) -> str:
    """Emit CuteDSL code for creating a tiled MMA operation.

    For SM80 (Ampere), uses ``warp.MmaF16BF16Op`` with warp-level tiling.
    For SM90+ (Hopper), uses warpgroup MMA with 128-thread groups.

    Args:
        dtype_str: CuteDSL dtype string for inputs (e.g. ``"cutlass.Float16"``).
        acc_dtype_str: CuteDSL dtype string for accumulator.
        num_warps: Number of warps per block.
        arch: SM version.
        dtype_bytes: Element size in bytes.

    Returns:
        CuteDSL code string for creating a tiled MMA.
    """
    mma_m, mma_n, mma_k = get_mma_shape(dtype_bytes, arch)

    if arch >= SM90:
        # Warpgroup MMA for Hopper
        tile_m = num_warps * 16  # Scale M tile with warps
        return (
            f"sm90_utils_basic.make_trivial_tiled_mma("
            f"{dtype_str}, {dtype_str}, "
            f"warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, "
            f"{acc_dtype_str}, "
            f"atom_layout_mnk=({tile_m // 64}, 1, 1), "
            f"tiler_mn=(64, {mma_n}))"
        )
    # Ampere warp MMA
    perm_m = num_warps * mma_m
    return (
        f"cute.make_tiled_mma("
        f"warp.MmaF16BF16Op({dtype_str}, {acc_dtype_str}, "
        f"({mma_m}, {mma_n}, {mma_k})), "
        f"({num_warps}, 1, 1), "
        f"permutation_mnk=({perm_m}, {mma_n}, {mma_k}))"
    )


# ---------------------------------------------------------------------------
# Copy atom selection (GMEM -> SMEM and SMEM -> REG)
# ---------------------------------------------------------------------------

def emit_gmem_copy_atom(
    dtype_str: str,
    dtype_bits: int,
    use_async: bool = True,
) -> str:
    """Emit CuteDSL code for a GMEM -> SMEM copy atom.

    Args:
        dtype_str: CuteDSL dtype string.
        dtype_bits: Width of the dtype in bits.
        use_async: Whether to use cp.async (SM80+).

    Returns:
        CuteDSL code string for the copy atom.
    """
    copy_bits = get_copy_bits(dtype_bits)
    if use_async:
        return (
            f"cute.make_copy_atom("
            f"cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), "
            f"{dtype_str}, num_bits_per_copy={copy_bits})"
        )
    return (
        f"cute.make_copy_atom("
        f"cute.nvgpu.CopyUniversalOp(), "
        f"{dtype_str}, num_bits_per_copy={copy_bits})"
    )


def emit_smem_copy_atom(
    dtype_str: str,
    dtype_bits: int,
    arch: int,
    transpose: bool = False,
) -> str:
    """Emit CuteDSL code for a SMEM -> REG copy atom.

    Uses architecture-specific instructions when available.

    Args:
        dtype_str: CuteDSL dtype string.
        dtype_bits: Width of the dtype in bits.
        arch: SM version.
        transpose: Whether to use a transposed store.

    Returns:
        CuteDSL code string for the copy atom.
    """
    if arch >= SM90 and dtype_bits == 16:
        # Hopper: use StMatrix instruction
        transpose_str = "True" if transpose else "False"
        return (
            f"cute.make_copy_atom("
            f"cute.nvgpu.warp.StMatrix8x8x16bOp("
            f"transpose={transpose_str}, num_matrices=4), "
            f"{dtype_str})"
        )
    # Ampere: universal copy
    copy_bits = min(128, 2 * dtype_bits)
    return (
        f"cute.make_copy_atom("
        f"cute.nvgpu.CopyUniversalOp(), "
        f"{dtype_str}, num_bits_per_copy={copy_bits})"
    )


# ---------------------------------------------------------------------------
# Tiled copy layout computation
# ---------------------------------------------------------------------------

def emit_tiled_copy_2d(
    dtype_str: str,
    dtype_bits: int,
    major_mode_size: int,
    num_threads: int,
    use_async: bool = True,
) -> str:
    """Emit CuteDSL code for a 2D tiled copy.

    Creates a tiled copy with thread and value layouts for 2D data,
    maximizing memory coalescing.

    Args:
        dtype_str: CuteDSL dtype string.
        dtype_bits: Width of the dtype in bits.
        major_mode_size: Size of the major (contiguous) mode.
        num_threads: Number of threads participating.
        use_async: Whether to use cp.async.

    Returns:
        CuteDSL code string for the tiled copy.
    """
    copy_bits = math.gcd(major_mode_size * dtype_bits, 128)
    copy_elems = copy_bits // dtype_bits
    threads_per_row = major_mode_size // copy_elems

    copy_atom = emit_gmem_copy_atom(dtype_str, dtype_bits, use_async)
    thr_rows = num_threads // threads_per_row

    return (
        f"cute.make_tiled_copy_tv("
        f"{copy_atom}, "
        f"cute.make_ordered_layout(({thr_rows}, {threads_per_row}), order=(1, 0)), "
        f"cute.make_layout((1, {copy_elems})))"
    )


# ---------------------------------------------------------------------------
# SMEM buffer size computation
# ---------------------------------------------------------------------------

def compute_smem_size(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_bytes: int,
    num_stages: int,
) -> int:
    """Compute total shared memory size for a tiled GEMM.

    Accounts for double/triple buffering of A and B SMEM tiles.

    Args:
        tile_m: M tile dimension.
        tile_n: N tile dimension.
        tile_k: K tile dimension.
        dtype_bytes: Size of each element in bytes.
        num_stages: Number of pipeline stages (1 = no pipelining).

    Returns:
        Total shared memory size in bytes.
    """
    a_size = tile_m * tile_k * dtype_bytes
    b_size = tile_k * tile_n * dtype_bytes
    return (a_size + b_size) * num_stages


# ---------------------------------------------------------------------------
# SM90 (Hopper) warpgroup MMA utilities
# ---------------------------------------------------------------------------

# Operand major modes for warpgroup MMA
OPERAND_MAJOR_K = "K"
OPERAND_MAJOR_MN = "MN"

# Operand sources for warpgroup MMA
OPERAND_SOURCE_SMEM = "SMEM"
OPERAND_SOURCE_RMEM = "RMEM"


def get_mma_shape_sm90(dtype_bytes: int) -> tuple[int, int, int]:
    """Get the warpgroup MMA atom shape (M, N, K) for SM90.

    SM90 uses warpgroup MMA (128-thread groups = 4 warps).

    Args:
        dtype_bytes: Element size in bytes.

    Returns:
        ``(mma_m, mma_n, mma_k)`` tuple.
    """
    if dtype_bytes <= 1:
        return (64, 16, 32)  # FP8/INT8
    if dtype_bytes <= 2:
        return (64, 16, 16)  # FP16/BF16
    return (64, 16, 8)  # FP32/TF32


def emit_tiled_mma_sm90(
    dtype_str: str,
    acc_dtype_str: str,
    num_warps: int,
    dtype_bytes: int = 2,
    *,
    a_major: str = OPERAND_MAJOR_K,
    b_major: str = OPERAND_MAJOR_K,
    a_source: str = OPERAND_SOURCE_SMEM,
    tile_n_override: int | None = None,
) -> str:
    """Emit CuteDSL code for SM90 warpgroup MMA.

    SM90 uses warpgroup MMA with 128-thread groups (4 warps each).
    The atom_layout_mnk controls how warpgroups tile the M dimension.

    Args:
        dtype_str: CuteDSL dtype string for inputs.
        acc_dtype_str: CuteDSL dtype string for accumulator.
        num_warps: Number of warps per block.
        dtype_bytes: Element size in bytes.
        a_major: Operand A major mode ("K" or "MN").
        b_major: Operand B major mode ("K" or "MN").
        a_source: Operand A source ("SMEM" or "RMEM").
        tile_n_override: Override for the N tile in the tiler_mn.

    Returns:
        CuteDSL code string for creating a warpgroup tiled MMA.
    """
    mma_m, mma_n, mma_k = get_mma_shape_sm90(dtype_bytes)
    # Each warpgroup is 4 warps; scale M dimension by number of warpgroups
    num_warpgroups = max(1, num_warps // 4)
    tile_n = tile_n_override if tile_n_override is not None else mma_n

    parts = [
        f"sm90_utils_basic.make_trivial_tiled_mma(",
        f"    {dtype_str}, {dtype_str},",
        f"    warpgroup.OperandMajorMode.{a_major},",
        f"    warpgroup.OperandMajorMode.{b_major},",
        f"    {acc_dtype_str},",
        f"    atom_layout_mnk=({num_warpgroups}, 1, 1),",
        f"    tiler_mn=({mma_m}, {tile_n}),",
    ]
    if a_source == OPERAND_SOURCE_RMEM:
        parts.append(f"    a_source=warpgroup.OperandSource.RMEM,")
    parts.append(")")
    return "\n".join(parts)


def emit_smem_layout_atom_sm90(
    k_dim: int,
    dtype_bytes: int,
    dtype_str: str,
) -> str:
    """Emit CuteDSL code for SM90 warpgroup-compatible SMEM layout atom.

    Wraps the base SMEM layout atom with warpgroup.make_smem_layout_atom()
    for compatibility with SM90 warpgroup MMA.

    Args:
        k_dim: Size of the K dimension for the SMEM tile.
        dtype_bytes: Size of each element in bytes.
        dtype_str: CuteDSL dtype string.

    Returns:
        CuteDSL code string for the SM90 SMEM layout atom.
    """
    base_layout = emit_smem_layout_atom(k_dim, dtype_bytes)
    return (
        f"warpgroup.make_smem_layout_atom("
        f"sm90_utils_basic.get_smem_layout_atom("
        f"LayoutEnum.ROW_MAJOR, {dtype_str}, {k_dim}), "
        f"{dtype_str})"
    )


# ---------------------------------------------------------------------------
# SM100 (Blackwell) tcgen05 MMA utilities
# ---------------------------------------------------------------------------

# TMEM capacity per CTA
SM100_TMEM_CAPACITY_COLUMNS = 512


def get_mma_shape_sm100(dtype_bytes: int) -> tuple[int, int, int]:
    """Get the tcgen05 MMA atom shape (M, N, K) for SM100.

    SM100 uses tcgen05 MMA with larger tile sizes.

    Args:
        dtype_bytes: Element size in bytes.

    Returns:
        ``(mma_m, mma_n, mma_k)`` tuple.
    """
    if dtype_bytes <= 1:
        return (128, 256, 64)  # FP8/INT8
    if dtype_bytes <= 2:
        return (128, 256, 32)  # FP16/BF16
    return (128, 256, 16)  # FP32/TF32


def emit_tiled_mma_sm100(
    dtype_str: str,
    acc_dtype_str: str,
    dtype_bytes: int = 2,
    *,
    a_major: str = OPERAND_MAJOR_K,
    b_major: str = OPERAND_MAJOR_K,
    a_source: str = OPERAND_SOURCE_SMEM,
) -> str:
    """Emit CuteDSL code for SM100 tcgen05 MMA.

    SM100 uses tcgen05 instructions with SMEM descriptor-based operands
    and optional TMEM source for operand A.

    Args:
        dtype_str: CuteDSL dtype string for inputs.
        acc_dtype_str: CuteDSL dtype string for accumulator.
        dtype_bytes: Element size in bytes.
        a_major: Operand A major mode ("K" or "MN").
        b_major: Operand B major mode ("K" or "MN").
        a_source: Operand A source ("SMEM" or "TMEM").

    Returns:
        CuteDSL code string for creating a tcgen05 tiled MMA.
    """
    mma_m, mma_n, mma_k = get_mma_shape_sm100(dtype_bytes)
    src_str = "tcgen05.OperandSource.TMEM" if a_source == "TMEM" else "tcgen05.OperandSource.SMEM"
    return (
        f"tcgen05.mma.MmaOp("
        f"shape_mnk=({mma_m}, {mma_n}, {mma_k}), "
        f"a_dtype={dtype_str}, b_dtype={dtype_str}, "
        f"acc_dtype={acc_dtype_str}, "
        f"a_major_mode=tcgen05.mma.OperandMajorMode.{a_major}, "
        f"b_major_mode=tcgen05.mma.OperandMajorMode.{b_major}, "
        f"a_src={src_str})"
    )


# ---------------------------------------------------------------------------
# TMA copy atoms (SM90+)
# ---------------------------------------------------------------------------

def emit_tma_copy_atom(
    dtype_str: str,
    ndim: int = 2,
) -> str:
    """Emit CuteDSL code for a TMA (Tensor Memory Accelerator) copy atom.

    TMA is available on SM90+ and provides hardware-accelerated
    multi-dimensional async copies with automatic padding and swizzling.

    Args:
        dtype_str: CuteDSL dtype string.
        ndim: Number of tensor dimensions (2-5).

    Returns:
        CuteDSL code string for a TMA copy atom.
    """
    return (
        f"cpasync.tma_partition("
        f"cute.make_copy_atom(cpasync.CopyTmaOp(), {dtype_str}))"
    )


def emit_tma_store_atom(
    dtype_str: str,
) -> str:
    """Emit CuteDSL code for a TMA store atom (SM90+).

    Args:
        dtype_str: CuteDSL dtype string.

    Returns:
        CuteDSL code string for a TMA store atom.
    """
    return f"cute.make_copy_atom(cpasync.CopyTmaStoreOp(), {dtype_str})"


# ---------------------------------------------------------------------------
# SMEM descriptor layout types (SM100)
# ---------------------------------------------------------------------------

# Layout types for SM100 UMMA SMEM descriptors
SMEM_LAYOUT_SWIZZLE_NONE = 0
SMEM_LAYOUT_SWIZZLE_128B_BASE32B = 1
SMEM_LAYOUT_SWIZZLE_128B = 2
SMEM_LAYOUT_SWIZZLE_64B = 4
SMEM_LAYOUT_SWIZZLE_32B = 6


def get_smem_descriptor_layout_type(swizzle_bits: int) -> int:
    """Map swizzle bits to SM100 SMEM descriptor layout type.

    Args:
        swizzle_bits: Number of swizzle bits (0-4).

    Returns:
        SM100 LayoutType integer value.
    """
    mapping = {
        0: SMEM_LAYOUT_SWIZZLE_NONE,
        1: SMEM_LAYOUT_SWIZZLE_32B,
        2: SMEM_LAYOUT_SWIZZLE_64B,
        3: SMEM_LAYOUT_SWIZZLE_128B_BASE32B,
        4: SMEM_LAYOUT_SWIZZLE_128B,
    }
    return mapping.get(swizzle_bits, SMEM_LAYOUT_SWIZZLE_NONE)


# ---------------------------------------------------------------------------
# Architecture-aware MMA dispatch
# ---------------------------------------------------------------------------

def emit_tiled_mma_for_arch(
    dtype_str: str,
    acc_dtype_str: str,
    num_warps: int,
    arch: int,
    dtype_bytes: int = 2,
) -> str:
    """Emit CuteDSL code for the optimal MMA for the given architecture.

    Dispatches to the appropriate MMA emission function based on arch.

    Args:
        dtype_str: CuteDSL dtype string for inputs.
        acc_dtype_str: CuteDSL dtype string for accumulator.
        num_warps: Number of warps per block.
        arch: SM version.
        dtype_bytes: Element size in bytes.

    Returns:
        CuteDSL code string for creating the architecture-optimal tiled MMA.
    """
    if arch >= SM100:
        return emit_tiled_mma_sm100(
            dtype_str, acc_dtype_str, dtype_bytes,
        )
    if arch >= SM90:
        return emit_tiled_mma_sm90(
            dtype_str, acc_dtype_str, num_warps, dtype_bytes,
        )
    # SM80: use the existing Ampere emit_tiled_mma
    return emit_tiled_mma(
        dtype_str, acc_dtype_str, num_warps, arch, dtype_bytes,
    )


def emit_gmem_copy_atom_for_arch(
    dtype_str: str,
    dtype_bits: int,
    arch: int,
) -> str:
    """Emit the optimal GMEM->SMEM copy atom for the given architecture.

    SM90+ uses TMA, SM80 uses cp.async, earlier uses universal copy.

    Args:
        dtype_str: CuteDSL dtype string.
        dtype_bits: Width of the dtype in bits.
        arch: SM version.

    Returns:
        CuteDSL code string for the copy atom.
    """
    if arch >= SM90:
        return emit_tma_copy_atom(dtype_str)
    return emit_gmem_copy_atom(dtype_str, dtype_bits, use_async=arch >= SM80)
