"""CuteDSL reduction code generation utilities.

This module provides reduction codegen helpers for the CuteDSL backend.
CuteDSL does not have a built-in ``tl.reduce()`` equivalent, so reductions
must be explicitly implemented using warp shuffle + shared memory patterns.

Utilities provided:
- Warp-level reductions via butterfly shuffle instructions
- Block-level reductions via shared memory cross-warp reduction
- Support for common reduction ops (sum, max, min, prod)
- Architecture-aware optimizations (SM80 vs SM90+)

These are parameterized by (op, tile_shape, num_warps, dtype) and
generate deterministic CuteDSL code snippets.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Warp size is 32 threads on all NVIDIA GPUs
WARP_SIZE = 32


def get_warp_reduce_op(op_name: str) -> str:
    """Get the CuteDSL warp shuffle reduction pattern for a given op.

    Args:
        op_name: One of 'sum', 'max', 'min', 'prod'.

    Returns:
        A string template for the warp-level reduction operation.
        Use ``.format(a=..., b=...)`` to fill in operand names.
    """
    ops = {
        "sum": "{a} + {b}",
        "max": "max({a}, {b})",
        "min": "min({a}, {b})",
        "prod": "{a} * {b}",
    }
    if op_name not in ops:
        raise ValueError(f"Unsupported reduction op: {op_name}")
    return ops[op_name]


def compute_reduction_smem_size(
    num_warps: int,
    dtype_bytes: int,
    elements_per_warp: int = 1,
) -> int:
    """Compute shared memory size needed for cross-warp reduction.

    For a block-level reduction, each warp produces one partial result per
    reduction element.  These are written to shared memory so that a single
    warp can finish the cross-warp reduction.

    Args:
        num_warps: Number of warps per block.
        dtype_bytes: Size of each element in bytes.
        elements_per_warp: Number of reduction elements per warp.

    Returns:
        Shared memory size in bytes.
    """
    return num_warps * elements_per_warp * dtype_bytes


def get_reduction_identity(op_name: str, dtype_str: str) -> str:
    """Get the identity element for a reduction operation.

    Args:
        op_name: Reduction operation name.
        dtype_str: CuteDSL dtype string (e.g., 'cutlass.Float32').

    Returns:
        String representation of the identity element.
    """
    identities = {
        "sum": "0.0",
        "max": "float('-inf')",
        "min": "float('inf')",
        "prod": "1.0",
    }
    if op_name not in identities:
        raise ValueError(f"Unsupported reduction op: {op_name}")
    return identities[op_name]


def num_shuffle_rounds(width: int = WARP_SIZE) -> int:
    """Number of butterfly shuffle rounds for a given reduction width.

    Args:
        width: The number of lanes participating (must be power of 2).

    Returns:
        ``log2(width)`` — the number of shuffle rounds.
    """
    assert width > 0 and (width & (width - 1)) == 0, (
        f"width must be a power of 2, got {width}"
    )
    return int(math.log2(width))


def emit_warp_reduce_snippet(
    val_name: str,
    op_name: str,
    width: int = WARP_SIZE,
) -> list[str]:
    """Emit CuteDSL code lines for a butterfly warp-level reduction.

    Generates a sequence of ``cute.arch.shuffle_sync_bfly`` calls that
    reduce *val_name* in-place using the given binary *op_name*.

    Args:
        val_name: Name of the register variable holding the value to reduce.
        op_name: One of 'sum', 'max', 'min', 'prod'.
        width: Number of lanes participating (must be power of 2, <= 32).

    Returns:
        List of Python code lines (without leading indentation).

    Example output for ``emit_warp_reduce_snippet("v", "sum", 32)``::

        v = v + cute.arch.shuffle_sync_bfly(v, offset=1)
        v = v + cute.arch.shuffle_sync_bfly(v, offset=2)
        v = v + cute.arch.shuffle_sync_bfly(v, offset=4)
        v = v + cute.arch.shuffle_sync_bfly(v, offset=8)
        v = v + cute.arch.shuffle_sync_bfly(v, offset=16)
    """
    op_template = get_warp_reduce_op(op_name)
    lines: list[str] = []
    for i in range(num_shuffle_rounds(width)):
        offset = 1 << i
        shuffled = f"cute.arch.shuffle_sync_bfly({val_name}, offset={offset})"
        rhs = op_template.format(a=val_name, b=shuffled)
        lines.append(f"{val_name} = {rhs}")
    return lines


def emit_block_reduce_snippet(
    val_name: str,
    op_name: str,
    num_warps: int,
    smem_name: str = "smem_reduce",
    thread_idx_expr: str = "cute.arch.thread_idx()[0]",
) -> list[str]:
    """Emit CuteDSL code lines for a full block-level reduction.

    The pattern is:
    1. Each warp does a warp-level butterfly reduction.
    2. Lane 0 of each warp writes its result to shared memory.
    3. A ``__syncthreads()`` barrier.
    4. The first warp loads all partial results and does a final reduction.

    Args:
        val_name: Name of the register variable to reduce.
        op_name: One of 'sum', 'max', 'min', 'prod'.
        num_warps: Number of warps in the block.
        smem_name: Name of the shared memory buffer.
        thread_idx_expr: Expression for the thread index.

    Returns:
        List of Python code lines.
    """
    identity = get_reduction_identity(op_name, "")
    op_template = get_warp_reduce_op(op_name)

    lines: list[str] = []

    # Step 1: intra-warp reduction
    lines.append(f"# Warp-level reduction for {op_name}")
    lines.extend(emit_warp_reduce_snippet(val_name, op_name, WARP_SIZE))

    # Step 2: lane 0 writes to SMEM
    lines.append(f"warp_id = {thread_idx_expr} // {WARP_SIZE}")
    lines.append(f"lane_id = {thread_idx_expr} % {WARP_SIZE}")
    lines.append(f"if lane_id == 0:")
    lines.append(f"    {smem_name}[warp_id] = {val_name}")

    # Step 3: barrier
    lines.append("cute.arch.syncthreads()")

    # Step 4: first warp reads all partial results and reduces
    lines.append(f"if warp_id == 0:")
    lines.append(
        f"    {val_name} = {smem_name}[lane_id] if lane_id < {num_warps} else {identity}"
    )
    # Final warp-level reduction across partial results
    # Use next power of 2 >= num_warps for the shuffle width
    final_width = 1
    while final_width < num_warps:
        final_width *= 2
    final_width = min(final_width, WARP_SIZE)
    for line in emit_warp_reduce_snippet(val_name, op_name, final_width):
        lines.append(f"    {line}")

    return lines


def get_cutedsl_reduce_fn_name(op_name: str) -> str:
    """Get the CuteDSL utility function name for a standard reduction op.

    Maps Helion reduce op names to the corresponding CuteDSL utility
    function names from the ``flash_attn/cute/utils.py`` patterns.

    Args:
        op_name: One of 'sum', 'max', 'min'.

    Returns:
        CuteDSL utility function name (e.g., ``"utils.fadd_reduce"``).
    """
    mapping = {
        "sum": "fadd_reduce",
        "max": "fmax_reduce",
        "min": "fmin_reduce",
    }
    if op_name not in mapping:
        raise ValueError(f"No CuteDSL reduce function for op: {op_name}")
    return mapping[op_name]
