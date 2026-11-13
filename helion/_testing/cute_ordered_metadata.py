"""Ordered phase kernels with pure host metadata and captured storage views."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def computed_metadata_phases(
    x: torch.Tensor, y: torch.Tensor, shift: hl.constexpr
) -> torch.Tensor:
    batches, rows, reduction = x.shape
    cols = y.size(2)
    assert reduction == y.shape[-2]
    assert rows > 0 and cols > 0
    if shift is None:
        bias = 1.0
    else:
        bias = float(shift)
    scale = reduction**-0.5
    span = (
        1
        + (batches - 1) * x.stride(0)
        + (rows - 1) * x.stride(1)
        + (reduction - 1) * x.stride(2)
    )
    storage = x.as_strided((span,), (1,))
    temporary = torch.empty((batches, rows, cols), device=x.device, dtype=torch.float32)
    temporary_flat = temporary.flatten()
    output = torch.empty_like(temporary)
    output_flat = output.flatten()
    for batch in hl.grid(batches):
        r, k, c = hl.arange(rows), hl.arange(reduction), hl.arange(cols)
        # PyTorch wheel stubs omit this supported SymInt/Tensor sum.
        # pyrefly: ignore[unsupported-operation]
        indices = batch * x.stride(0) + r[:, None] * x.stride(1) + k[None, :]
        left = (storage[indices].float() * scale).to(x.dtype)
        value = hl.dot(left, y[batch, k, c])
        temporary[batch, r, c] = value + torch.amax(value, dim=0)[None, :]
    hl.barrier()
    for index in hl.tile(output.numel()):
        output_flat[index] = temporary_flat[index] + bias
    return output
