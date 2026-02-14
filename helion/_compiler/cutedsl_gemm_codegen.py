"""CuteDSL GEMM (General Matrix Multiply) codegen utilities.

Provides helpers for generating pipelined matrix multiplication kernels
using CuteDSL. The generated code follows the pattern:

1. **Preamble**: Allocate SMEM, set up copy atoms, MMA atoms.
2. **Main loop**: Pipelined K-loop with async GMEM->SMEM copies,
   SMEM->REG copies, and MMA accumulation.
3. **Epilogue**: Write accumulated results from registers to GMEM.

All choices are parameterized by ``(tile_m, tile_n, tile_k, dtype, num_stages, arch)``
and are deterministic — no pattern matching.

Reference: ``flash_attn/cute/flash_fwd.py`` ``_setup_attributes()``,
``emit_gemm_kernel()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cutedsl_arch_utils import SM80
from .cutedsl_arch_utils import SM90
from .cutedsl_arch_utils import SM100
from .cutedsl_mma_utils import compute_smem_size
from .cutedsl_mma_utils import emit_gmem_copy_atom
from .cutedsl_mma_utils import emit_gmem_copy_atom_for_arch
from .cutedsl_mma_utils import emit_smem_copy_atom
from .cutedsl_mma_utils import emit_smem_layout_atom
from .cutedsl_mma_utils import emit_tiled_mma_for_arch
from .cutedsl_pipeline import PipelineConfig
from .cutedsl_pipeline import emit_pipelined_mainloop

if TYPE_CHECKING:
    pass


# Alignment for SMEM allocations (bytes)
SMEM_ALIGNMENT = 1024


def emit_gemm_smem_alloc(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_str: str,
    dtype_bytes: int,
    num_stages: int,
    var_prefix: str = "",
) -> list[str]:
    """Emit shared memory allocation code for GEMM tiles.

    Allocates swizzled SMEM buffers for A and B operand tiles with
    multi-stage buffering.

    Args:
        tile_m: M tile dimension.
        tile_n: N tile dimension.
        tile_k: K tile dimension.
        dtype_str: CuteDSL dtype string.
        dtype_bytes: Element size in bytes.
        num_stages: Pipeline stages.
        var_prefix: Optional prefix for variable names.

    Returns:
        List of CuteDSL Python code lines.
    """
    lines: list[str] = []
    p = var_prefix

    smem_a_layout = emit_smem_layout_atom(tile_k, dtype_bytes)
    smem_b_layout = emit_smem_layout_atom(tile_n, dtype_bytes)

    total_bytes = compute_smem_size(tile_m, tile_n, tile_k, dtype_bytes, num_stages)
    lines.append(f"# Shared memory allocation: {total_bytes} bytes")
    lines.append(
        f"{p}sA_layout_atom = {smem_a_layout}"
    )
    lines.append(
        f"{p}sB_layout_atom = {smem_b_layout}"
    )

    # Multi-stage tile shape
    lines.append(
        f"{p}sA = cute.make_tensor("
        f"{dtype_str}, "
        f"cute.make_layout(({tile_m}, {tile_k}, {num_stages})), "
        f"smem=True)"
    )
    lines.append(
        f"{p}sB = cute.make_tensor("
        f"{dtype_str}, "
        f"cute.make_layout(({tile_k}, {tile_n}, {num_stages})), "
        f"smem=True)"
    )

    return lines


def emit_gemm_copy_setup(
    dtype_str: str,
    dtype_bits: int,
    num_threads: int,
    arch: int,
    var_prefix: str = "",
) -> list[str]:
    """Emit copy atom setup code for GMEM->SMEM transfers.

    Uses architecture-optimal copy strategy: TMA for SM90+, cp.async for SM80.

    Args:
        dtype_str: CuteDSL dtype string.
        dtype_bits: Element width in bits.
        num_threads: Number of threads per block.
        arch: SM version.
        var_prefix: Optional prefix for variable names.

    Returns:
        List of CuteDSL Python code lines.
    """
    lines: list[str] = []
    p = var_prefix

    copy_atom_code = emit_gmem_copy_atom_for_arch(dtype_str, dtype_bits, arch)
    if arch >= SM90:
        lines.append(f"# GMEM -> SMEM copy (TMA)")
    elif arch >= SM80:
        lines.append(f"# GMEM -> SMEM copy (cp.async)")
    else:
        lines.append(f"# GMEM -> SMEM copy (universal)")
    lines.append(f"{p}gmem_copy_atom = {copy_atom_code}")

    # Also set up SMEM -> REG copy atom
    smem_copy = emit_smem_copy_atom(dtype_str, dtype_bits, arch)
    lines.append(f"{p}smem_copy_atom = {smem_copy}")

    return lines


def emit_gemm_mma_setup(
    dtype_str: str,
    acc_dtype_str: str,
    num_warps: int,
    arch: int,
    dtype_bytes: int = 2,
    var_prefix: str = "",
) -> list[str]:
    """Emit MMA atom setup code.

    Uses architecture-optimal MMA: tcgen05 for SM100, warpgroup for SM90,
    warp MMA for SM80.

    Args:
        dtype_str: Input dtype string.
        acc_dtype_str: Accumulator dtype string.
        num_warps: Warps per block.
        arch: SM version.
        dtype_bytes: Element size in bytes.
        var_prefix: Optional prefix for variable names.

    Returns:
        List of CuteDSL Python code lines.
    """
    lines: list[str] = []
    p = var_prefix

    mma_code = emit_tiled_mma_for_arch(dtype_str, acc_dtype_str, num_warps, arch, dtype_bytes)

    if arch >= SM100:
        lines.append(f"# MMA atom (tcgen05)")
    elif arch >= SM90:
        lines.append(f"# MMA atom (warpgroup MMA)")
    else:
        lines.append(f"# MMA atom (warp MMA)")
    lines.append(f"{p}tiled_mma = {mma_code}")

    return lines


def emit_gemm_preamble(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_str: str,
    acc_dtype_str: str,
    dtype_bytes: int,
    num_stages: int,
    num_warps: int,
    arch: int,
    var_prefix: str = "",
) -> list[str]:
    """Emit the full GEMM preamble: SMEM alloc + copy setup + MMA setup.

    Args:
        tile_m: M tile dimension.
        tile_n: N tile dimension.
        tile_k: K tile dimension.
        dtype_str: Input element dtype string.
        acc_dtype_str: Accumulator dtype string.
        dtype_bytes: Element size in bytes.
        num_stages: Pipeline stages.
        num_warps: Warps per block.
        arch: SM version.
        var_prefix: Optional prefix for variable names.

    Returns:
        List of CuteDSL Python code lines.
    """
    lines: list[str] = []
    dtype_bits = dtype_bytes * 8
    num_threads = num_warps * 32

    lines.extend(
        emit_gemm_smem_alloc(
            tile_m, tile_n, tile_k, dtype_str, dtype_bytes, num_stages, var_prefix
        )
    )
    lines.append("")
    lines.extend(
        emit_gemm_copy_setup(dtype_str, dtype_bits, num_threads, arch, var_prefix)
    )
    lines.append("")
    lines.extend(
        emit_gemm_mma_setup(
            dtype_str, acc_dtype_str, num_warps, arch, dtype_bytes, var_prefix
        )
    )

    return lines


def emit_gemm_mainloop(
    tile_k: int,
    k_extent: str,
    num_stages: int,
    arch: int,
    var_prefix: str = "",
) -> list[str]:
    """Emit the pipelined GEMM main loop.

    Uses the PipelineConfig infrastructure for arch-aware pipeline generation:
    - SM100: TMA pipeline with mbarriers
    - SM90: TMA pipeline with warpgroup synchronization
    - SM80: cp.async pipeline with commit/wait groups
    - Single stage: simple synchronous loop

    Args:
        tile_k: K tile dimension.
        k_extent: Expression for the total K extent.
        num_stages: Pipeline stages.
        arch: SM version.
        var_prefix: Optional prefix for variable names.

    Returns:
        List of CuteDSL Python code lines.
    """
    config = PipelineConfig.from_config(num_stages, arch)
    lines: list[str] = []
    lines.append(f"# Main GEMM loop (K dimension, pipeline={config.pipeline_mode})")
    lines.extend(
        emit_pipelined_mainloop(config, tile_k, k_extent, var_prefix)
    )
    return lines


def emit_gemm_epilogue(
    arch: int,
    acc_dtype_str: str | None = None,
    out_dtype_str: str | None = None,
    var_prefix: str = "",
) -> list[str]:
    """Emit the GEMM epilogue: write results from registers to GMEM.

    Handles type casting from accumulator dtype to output dtype if needed,
    and uses architecture-optimal write-back strategy.

    Args:
        arch: SM version.
        acc_dtype_str: Accumulator dtype string (e.g. "cutlass.Float32").
        out_dtype_str: Output dtype string (e.g. "cutlass.Float16").
        var_prefix: Optional prefix for variable names.

    Returns:
        List of CuteDSL Python code lines.
    """
    lines: list[str] = []
    p = var_prefix

    lines.append(f"# Epilogue: write accumulator to GMEM")

    # Type casting if accumulator and output dtypes differ
    needs_cast = (
        acc_dtype_str is not None
        and out_dtype_str is not None
        and acc_dtype_str != out_dtype_str
    )
    if needs_cast:
        lines.append(f"{p}# Cast accumulator from {acc_dtype_str} to {out_dtype_str}")
        lines.append(f"{p}acc_out = tl.cast(acc, {out_dtype_str})")
    else:
        lines.append(f"{p}acc_out = acc")

    if arch >= SM100:
        lines.append(f"{p}# SM100: TMEM -> SMEM -> GMEM writeback")
        lines.append(f"{p}cute.arch.syncthreads()")
        lines.append(f"{p}# Store acc_out to output GMEM via SMEM staging")
    elif arch >= SM90:
        lines.append(f"{p}# SM90: StMatrix for register -> SMEM -> GMEM")
        lines.append(f"{p}cute.arch.syncthreads()")
        lines.append(f"{p}# cute.copy(smem_copy_atom, acc_out, sC)")
        lines.append(f"{p}# cute.copy(gmem_copy_atom, sC, gC)")
    else:
        lines.append(f"{p}# SM80: Direct register -> GMEM store")
        lines.append(f"{p}cute.arch.syncthreads()")
        lines.append(f"{p}# Store acc_out directly to output GMEM")

    return lines


def emit_gemm_full(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_str: str,
    acc_dtype_str: str,
    out_dtype_str: str,
    dtype_bytes: int,
    num_stages: int,
    num_warps: int,
    arch: int,
    k_extent: str = "K",
    var_prefix: str = "",
) -> list[str]:
    """Emit a complete GEMM kernel body: preamble + mainloop + epilogue.

    This is a convenience function that composes all GEMM phases.

    Args:
        tile_m: M tile dimension.
        tile_n: N tile dimension.
        tile_k: K tile dimension.
        dtype_str: Input element dtype string.
        acc_dtype_str: Accumulator dtype string.
        out_dtype_str: Output dtype string.
        dtype_bytes: Element size in bytes.
        num_stages: Pipeline stages.
        num_warps: Warps per block.
        arch: SM version.
        k_extent: Expression for total K extent.
        var_prefix: Optional variable name prefix.

    Returns:
        List of CuteDSL Python code lines for the full GEMM.
    """
    lines: list[str] = []

    # Preamble
    lines.extend(
        emit_gemm_preamble(
            tile_m, tile_n, tile_k, dtype_str, acc_dtype_str,
            dtype_bytes, num_stages, num_warps, arch, var_prefix,
        )
    )
    lines.append("")

    # Accumulator initialization
    p = var_prefix
    lines.append(f"{p}# Initialize accumulator")
    lines.append(f"{p}acc = cute.zeros(({tile_m}, {tile_n}), {acc_dtype_str})")
    lines.append("")

    # Main loop
    lines.extend(
        emit_gemm_mainloop(tile_k, k_extent, num_stages, arch, var_prefix)
    )
    lines.append("")

    # Epilogue
    lines.extend(
        emit_gemm_epilogue(arch, acc_dtype_str, out_dtype_str, var_prefix)
    )

    return lines
