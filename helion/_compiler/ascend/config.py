"""Ascend NPU autotuner config helpers (env-tunable caps)."""

from __future__ import annotations

import os

import torch

_is_npu_latch: bool | None = None


def is_npu() -> bool:
    """Runtime Ascend NPU availability (torch_npu present and a device ready).

    ``True`` is latched for the process: an NPU never disappears mid-run, so
    hot paths avoid repeated ``torch.npu.is_available()`` calls.  ``False``
    keeps re-checking so a later lazy ``import torch_npu`` is still detected.
    Tests simulating NPU-absent hosts call :func:`reset_is_npu` (see
    ``test_ascend_config._force_npu_absent``).
    """
    global _is_npu_latch
    if _is_npu_latch is True:
        return True
    if hasattr(torch, "npu") and torch.npu.is_available():
        _is_npu_latch = True
        return True
    return False


def reset_is_npu() -> None:
    """Clear the latched NPU-availability result (test hook)."""
    global _is_npu_latch
    _is_npu_latch = None


def _npu_ub_budget_elements() -> int:
    """Max prod(block_sizes)*reduction_loops on Ascend (UB=192KB). Env: HELION_NPU_UB_BUDGET_ELEMENTS."""
    v = os.environ.get("HELION_NPU_UB_BUDGET_ELEMENTS", "").strip()
    try:
        return int(v) if v else 2048
    except ValueError:
        return 2048


def _npu_max_tensor_numel() -> int:
    """Per-tile max tensor numel on Ascend (UB=192KB). Env: HELION_NPU_MAX_TENSOR_NUMEL."""
    v = os.environ.get("HELION_NPU_MAX_TENSOR_NUMEL", "").strip()
    try:
        return int(v) if v else 8192
    except ValueError:
        return 8192


def _npu_default_reduction_loop() -> int:
    """Default reduction chunk on Ascend (must compile as baseline). Env: HELION_NPU_DEFAULT_REDUCTION_LOOP."""
    v = os.environ.get("HELION_NPU_DEFAULT_REDUCTION_LOOP", "").strip()
    try:
        return int(v) if v else 16
    except ValueError:
        return 16


def _npu_cap_reduction_loops(
    reduction_loops: list[object], block_sizes: object
) -> list[object] | None:
    """Cap ``reduction_loops`` to fit the Ascend UB (192 KB) budget.

    The UB bounds ``prod(block_sizes) * reduction_loops``, so an over-large
    reduction loop is floored to the largest power of two that fits the budget,
    and a ``None`` entry is materialized as the NPU default reduction loop.

    Returns the rewritten list, or ``None`` when no change was needed (so callers
    can skip the assignment and preserve the original list identity).
    """
    new_loops = list(reduction_loops)
    changed = False
    default_rl = _npu_default_reduction_loop()
    for i, rl in enumerate(new_loops):
        if rl is None:
            new_loops[i] = default_rl
            changed = True
    tile_product = 1
    if isinstance(block_sizes, list):
        for bs in block_sizes:
            if isinstance(bs, int) and bs > 0:
                tile_product *= bs
    if tile_product > 0:
        budget = _npu_ub_budget_elements()
        max_reduction = max(1, budget // tile_product)
        capped = 1 << (max_reduction.bit_length() - 1)
        for i, rl in enumerate(new_loops):
            if isinstance(rl, int) and rl > capped:
                new_loops[i] = capped
                changed = True
    return new_loops if changed else None
