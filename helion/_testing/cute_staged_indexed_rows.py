"""Small communicating pair of independently selected, BF16 row projections."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def selected_projection_pair(
    x: torch.Tensor,
    pair: torch.Tensor,
    down: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batches, width = x.shape
    _, slots = ids.shape
    groups, twice, _ = pair.shape
    rows = twice // 2
    output_width = down.size(1)
    matrix = pair.view(groups * twice, width)
    matrix2 = down.view(groups * output_width, twice)
    picked = torch.empty_like(ids)
    values = torch.empty_like(scores)
    mid = torch.empty((batches * slots, twice), device=x.device, dtype=x.dtype)
    mid3 = mid.view(batches, slots, twice)
    output = torch.empty((batches, slots, output_width), device=x.device, dtype=x.dtype)
    for b, s, m in hl.tile([batches, slots, rows], block_size=[1, 1, None]):
        bi = b.begin
        si = s.begin
        selected = ids[bi, si]
        low = selected * twice + m.index
        high = low + rows
        full = x[bi, :].float()
        inv = torch.rsqrt(torch.mean(full * full, dim=-1) + eps)
        a = hl.zeros([m], dtype=torch.float32)
        h = hl.zeros([m], dtype=torch.float32)
        for k in hl.tile(width):
            vector = (x[bi, k].float() * inv).to(x.dtype)
            normal = (vector * scale[k]).to(x.dtype).float()
            a = a + torch.sum(matrix[low, k].float() * normal, dim=-1)
            h = h + torch.sum(matrix[high, k].float() * normal, dim=-1)
        mid[bi * slots + si, m] = a.to(x.dtype)
        mid[bi * slots + si, m.index + rows] = h.to(x.dtype)
        if m.begin == 0:
            picked[bi, si] = selected
            values[bi, si] = scores[bi, si]
    for b, s, m in hl.tile([batches, slots, output_width], block_size=[1, 1, None]):
        bi = b.begin
        si = s.begin
        selection = picked[bi, si] * output_width + m.index
        acc = hl.zeros([m], dtype=torch.float32)
        for k in hl.tile(twice):
            v = mid3[bi, si, k].float()
            acc = acc + torch.sum(matrix2[selection, k].float() * v, dim=-1)
        output[bi, si, m] = (acc * values[bi, si]).to(x.dtype)
    return mid, picked, values, output


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def selected_projection_with_epilogue(
    x: torch.Tensor,
    pair: torch.Tensor,
    down: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batches, width = x.shape
    _, slots = ids.shape
    groups, twice, _ = pair.shape
    rows = twice // 2
    output_width = down.size(1)
    matrix = pair.view(groups * twice, width)
    k2 = down.size(2)
    matrix2 = down.view(groups * output_width, k2)
    picked = torch.empty_like(ids)
    values = torch.empty_like(scores)
    mid = torch.empty((batches * slots, twice), device=x.device, dtype=x.dtype)
    output = torch.empty((batches, slots, output_width), device=x.device, dtype=x.dtype)
    activation = torch.empty((batches * slots, rows), device=x.device, dtype=x.dtype)
    mid3 = activation.view(batches, slots, k2)
    for b, s, m in hl.tile([batches, slots, rows], block_size=[1, 1, None]):
        bi = b.begin
        si = s.begin
        selected = ids[bi, si]
        low = selected * twice + m.index
        high = low + rows
        full = x[bi, :].float()
        inv = torch.rsqrt(torch.mean(full * full, dim=-1) + eps)
        a = hl.zeros([m], dtype=torch.float32)
        h = hl.zeros([m], dtype=torch.float32)
        for k in hl.tile(width):
            vector = (x[bi, k].float() * inv).to(x.dtype)
            normal = (vector * scale[k]).to(x.dtype).float()
            a = a + torch.sum(matrix[low, k].float() * normal, dim=-1)
            h = h + torch.sum(matrix[high, k].float() * normal, dim=-1)
        mid[bi * slots + si, m] = a.to(x.dtype)
        mid[bi * slots + si, m.index + rows] = h.to(x.dtype)
        if m.begin == 0:
            picked[bi, si] = selected
            values[bi, si] = scores[bi, si]
    for token, col in hl.tile([batches * slots, rows]):
        lo = mid[token, col].float()
        hi = mid[token, col + rows].float()
        activation[token, col] = (lo * 0.25 + hi * -0.5).to(x.dtype)
    for b, s, m in hl.tile([batches, slots, output_width], block_size=[1, 1, None]):
        bi = b.begin
        si = s.begin
        selection = picked[bi, si] * output_width + m.index
        acc = hl.zeros([m], dtype=torch.float32)
        for k in hl.tile(k2):
            v = mid3[bi, si, k].float()
            acc = acc + torch.sum(matrix2[selection, k].float() * v, dim=-1)
        output[bi, si, m] = (acc * values[bi, si]).to(x.dtype)
    return mid, picked, values, output
