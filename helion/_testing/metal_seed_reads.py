"""A value that seeds an MPP K loop's accumulator, read again after the loop:
the op updates its storage in place, so such a read would see the op's
result (``mpp_matmul._check_read_before_the_write``).  Shared by
``test/test_metal_seed_reads.py`` (CPU codegen) and
``test/test_metal_seed_reads_numerics.py`` (MPS)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", static_shapes=True)
def seed_read_after_its_loop(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """``x @ y`` plus 4 ``b``, both loops starting from one ``hl.zeros``:
    the second, a grid of 4 (no lanes), reads it after the addmm loop."""
    m, k = x.shape
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        zero = hl.zeros([tm, tn], dtype=torch.float32)
        acc1 = zero
        for tk in hl.tile(k):
            acc1 = torch.addmm(acc1, x[tm, tk], y[tk, tn])
        acc2 = zero
        for _i in hl.grid(4):
            acc2 = acc2 + b[tm, tn]
        out[tm, tn] = acc1 + acc2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def carry_read_after_its_update(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x @ y`` and, beside it, the accumulator as it stood before the last
    K tile's addmm: ``a`` aliases the carry ahead of the op in the loop's
    body and is read after it."""
    m, k = x.shape
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    prev_out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        prev = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = acc
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            prev = a
        out[tm, tn] = acc
        prev_out[tm, tn] = prev
    return out, prev_out


@helion.kernel(backend="metal", static_shapes=True)
def carry_read_after_its_update_op(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``carry_read_after_its_update`` with an in-place update ahead of the
    op: ``a`` copies the carry from before the update, after it."""
    m, k = x.shape
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    prev_out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        prev = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a0 = acc
            acc = acc * 0.5
            a = a0
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            prev = a * 1.0
        out[tm, tn] = acc
        prev_out[tm, tn] = prev
    return out, prev_out


SEED_READ_CONFIG = helion.Config(block_sizes=[16, 16, 16])
