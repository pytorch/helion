"""A kernel, its config and a generated-code check shared by the
``cute_defer_tile_reduce`` test modules and the ``cute_tail_peel`` tests of
sunk reduction nests: per-split column sums walked by an explicit row loop
with masked loads (better-benchmark's BatchNorm backward partial sums, at
test sizes), and the count of the cross-thread combines its device loops
hold."""

from __future__ import annotations

import ast

import torch

import helion
import helion.language as hl

# The split column reduction's first seed: 128-row tiles on one warp (16 row
# threads beside 2 column threads), eight bf16 columns per thread as one
# packet, eight rows per thread and tile.
SPLIT_CONFIG = {
    "block_sizes": [1, 16, 128],
    "num_threads": [1, 2, 16],
    "cute_vector_widths": [1, 8, 1],
    "cute_lane_layouts": ["blocked", "blocked", "strided"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def split_column_sums(
    x: torch.Tensor, y: torch.Tensor, rows_per_split: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each block of ``rows_per_split`` rows (the last one partial), the
    column sums of ``x`` and of ``x * (y + 1)``, the way Inductor's looped
    split reductions walk a split's rows."""
    rows, n = x.size()
    # pyrefly: ignore [unsupported-operation]
    splits = (rows + rows_per_split - 1) // rows_per_split
    total = torch.empty([splits, n], dtype=torch.float32, device=x.device)
    dot = torch.empty([splits, n], dtype=torch.float32, device=x.device)
    for tile_s, tile_n in hl.tile([splits, n]):
        acc = hl.zeros([tile_s, tile_n], dtype=torch.float32)
        acc_dot = hl.zeros([tile_s, tile_n], dtype=torch.float32)
        # pyrefly: ignore [no-matching-overload]
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            mask = valid[:, :, None]
            a = hl.load(x, [row, tile_n], extra_mask=mask).to(torch.float32)
            b = hl.load(y, [row, tile_n], extra_mask=mask).to(torch.float32)
            # The port's own select of the valid rows, as the repro writes it.
            b = torch.where(mask, b + 1.0, 0.0)
            acc = acc + torch.sum(a, dim=1)
            acc_dot = acc_dot + torch.sum(a * b, dim=1)
        total[tile_s, tile_n] = acc
        dot[tile_s, tile_n] = acc_dot
    return total, dot


def split_column_sums_reference(
    x: torch.Tensor, y: torch.Tensor, rows_per_split: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``split_column_sums`` in float64."""
    rows, n = x.shape
    splits = -(-rows // rows_per_split)
    a = torch.zeros(splits * rows_per_split, n, dtype=torch.float64, device=x.device)
    b = torch.zeros_like(a)
    a[:rows] = x
    b[:rows] = y
    a = a.view(splits, rows_per_split, n)
    b = b.view(splits, rows_per_split, n)
    return a.sum(1), (a * (b + 1.0)).sum(1)


def _is_combine(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        "_cute_grouped_reduce" in (name := ast.unparse(node.func))
        or "warp_reduction" in name
    )


def cross_thread_combines(code: str) -> tuple[int, int]:
    """The cross-thread combines of the generated kernel ``code`` (warp
    reductions and grouped reduces): ``(inside a device loop, in all)``."""
    tree = ast.parse(code)
    inside = sum(
        _is_combine(node)
        for loop in ast.walk(tree)
        if isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and loop.target.id.startswith("tile_offset_")
        for node in ast.walk(loop)
    )
    return inside, sum(_is_combine(node) for node in ast.walk(tree))
