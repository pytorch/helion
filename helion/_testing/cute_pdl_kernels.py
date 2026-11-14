"""Triton producers and consumers for dependent-launch numerical tests.

Imported lazily by CPU-collected modules so CPU-only environments do not need
Triton just to inspect CuTe source.
"""

# Non-constexpr parameters are Triton DSL values, not Python host type contracts.
# ruff: noqa: ANN001

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents
from triton.language.extra.cuda import gdc_wait


@triton.jit
def _late_writer(
    src_ptr,
    dst_ptr,
    n,
    per_program,
    delay,
    RELEASE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Copies ``src`` to ``dst`` late: releases its dependents at entry (when
    ``RELEASE``) and spins ``delay`` rounds ahead of every store."""
    if RELEASE:
        gdc_launch_dependents()
    base = tl.program_id(0) * per_program
    for chunk in range(0, per_program, BLOCK):
        offsets = base + chunk + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(src_ptr + offsets, mask=mask)
        spin = tl.zeros([BLOCK], dtype=tl.float32)
        for _ in range(delay):
            spin = tl.sin(spin + 0.5)
        values = values + (spin * 0.0).to(values.dtype)
        tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _middle(flag_ptr, RELEASE_FIRST: tl.constexpr) -> None:
    """Waits on its predecessor and releases its dependents, in either order."""
    if RELEASE_FIRST:
        gdc_launch_dependents()
    gdc_wait()
    tl.atomic_add(flag_ptr, 1)
    if not RELEASE_FIRST:
        gdc_launch_dependents()


@triton.jit
def _poison_tail_without_wait(x_ptr, first, n, BLOCK: tl.constexpr) -> None:
    """Writes ones over ``x[first:n]`` right away: launched as a programmatic
    dependent, it runs as soon as its predecessor releases it."""
    offsets = first + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(x_ptr + offsets, tl.full([BLOCK], 1, tl.bfloat16), mask=mask)


@triton.jit
def _copy_without_wait(x_ptr, out_ptr, n, BLOCK: tl.constexpr) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)
