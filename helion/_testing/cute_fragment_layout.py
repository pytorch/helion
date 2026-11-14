"""Computed-fragment layout and scan kernels shared by tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", autotune_effort="none")
def expanded_mask_counts(
    value: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(value)
    counts_output = torch.empty_like(value, dtype=torch.int64)
    for rows in hl.tile(value.size(0)):
        viewed = mask[:].reshape(-1, 1)
        expanded = viewed.expand(viewed.shape[0], 2)
        counts = expanded.sum(-1)
        output[rows, :] = torch.where(counts > 1, value[rows, :], 0)
        counts_output[rows, :] = counts[None, :]
    return output, counts_output


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_singleton_scan(x: torch.Tensor, steps: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        value = (x[row, :, :] + 1).transpose(-2, -1)
        value = value.unsqueeze(-1).squeeze(-1)
        value = value.unsqueeze(1).unsqueeze(-1).squeeze((1, -1))
        # Restore the logical shape after testing singleton and non-singleton
        # last axes. Default squeeze also removes the temporary leading one.
        shape = value.shape
        value = value.squeeze(-1).reshape(shape)
        value = value.unsqueeze(0).squeeze().reshape(shape)
        value = value.unsqueeze(0).squeeze(0)
        carry = hl.cumsum(value, dim=-1)
        for _ in range(steps):
            carry = hl.cumsum(carry.unsqueeze(-2).squeeze(-2) + 1, dim=-1)
        out[row, :, :] = carry.transpose(-2, -1)
    return out
