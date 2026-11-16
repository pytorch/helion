"""Root-tile scalar expressions beside and inside MMA recipes."""

from __future__ import annotations

import torch

import helion.language as hl


def root_coordinate_matmul(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    m = a.size(0)
    n = hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row in hl.tile(m):
        if kind == "mixed":
            acc = hl.full([row, n], 0, torch.float32)
            for step in hl.grid(2):
                dot = hl.dot(a[row, :], b[:, :])
                # PyTorch wheel stubs omit this supported Tensor/SymInt product.
                # pyrefly: ignore[unsupported-operation]
                acc = acc + dot * (row.begin + step + 1)
            out[row, :] = acc
        else:
            factor = row.begin
            acc = hl.full([row, n], 0, torch.float32)
            for _step in hl.grid(2):
                if kind == "captured":
                    acc = hl.dot(a[row, :], b[:, :], acc=acc * factor)
                else:
                    acc = hl.dot(a[row, :], b[:, :], acc=acc)
            if kind == "begin":
                out[row, :] = acc * row.begin
            elif kind == "id":
                out[row, :] = acc * row.id
            else:
                out[row, :] = acc
    return out
