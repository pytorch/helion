"""Shared kernels for the mixed-radix (fused view/permute) store tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# The kernels spell their constants as literals, as Inductor does (a global
# int would trace as a runtime symbol the store proof cannot read).
SEQ = 512
HEAD = 64

# A 256x256 two-CTA pair tile, and a 128x256 one-CTA tile.
PAIR_CONFIG = helion.Config(
    block_sizes=[256, 256, 64],
    tcgen05_cta_group="two",
    tcgen05_cluster_m=2,
    tcgen05_ab_stages=6,
    pid_type="persistent_blocked",
    l2_groupings=[1],
)
ONE_CTA_CONFIG = helion.Config(
    block_sizes=[128, 256, 64],
    tcgen05_cluster_m=1,
    tcgen05_ab_stages=4,
    pid_type="persistent_blocked",
    l2_groupings=[1],
    tcgen05_layout_strategy="explicit_epi_tile",
    tcgen05_layout_overrides_epi_tile_m=128,
    tcgen05_layout_overrides_epi_tile_n=32,
    tcgen05_layout_overrides_d_store_box_n=32,
)


def _rows(tile: hl.Tile) -> torch.Tensor:
    """The tile's row coordinates as a column, as Inductor spells them."""
    return tile.index.reshape([tile, 1])  # pyrefly: ignore[no-matching-overload]


def _cols(tile: hl.Tile) -> torch.Tensor:
    """The tile's column coordinates as a row, as Inductor spells them."""
    return tile.index.reshape([1, tile])  # pyrefly: ignore[no-matching-overload]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def head_split_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """``out[b, h, s, d] = (a @ b)[b * 512 + s, h * 64 + d]``."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 512, x // 64, y % 512, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def flattened_head_split_mm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """The head split in Inductor's flattened-index spelling."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        flat = x + n * y
        out[y // 512, (flat // 64) % (n // 64), y % 512, flat % 64] = acc.to(
            torch.bfloat16
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def seq384_head_split_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """A 384-token head split (BERT SQuAD): 256-row tiles straddle runs."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 384, x // 64, y % 384, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def short_head_split_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """A 64-token head split: a 128-row TMA box spans two runs of rows."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 64, x // 64, y % 64, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pure_view_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """A fused view without a permute: the destination is row-major [M, N]."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 512, y % 512, x // 64, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def colliding_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Every head lands on head 0: not injective."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 512, x * 0, y % 512, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def skewed_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """The column a row lands in depends on the row: not separable."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 512, x // 64, y % 512, (x + y) % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def literal_reshape_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """A coordinate reshape with a literal extent instead of the block size."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n], block_size=[256, 256]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = tile_y.index.reshape([256, 1])
        x = tile_x.index.reshape([1, 256])
        out[y // 512, x // 64, y % 512, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def block_size_index_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """The head index reads the tile's block size: its value is config-dependent."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        head = x // 64 + (x // tile_x.block_size) * 8
        out[y // 512, head, y % 512, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def overflowing_head_split_mm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """``y * 2**24`` wraps in the kernel's int32 index arithmetic."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        y = _rows(tile_y)
        x = _cols(tile_x)
        seq = (y * 16777216 // 16777216) % 512
        out[y // 512, x // 64, seq, x % 64] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def transposed_store_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """``out[x, y] = acc[y, x]`` into the transpose of a row-major [M, N]."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        out[_cols(tile_x), _rows(tile_y)] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def fanout_head_split_mm(
    a: torch.Tensor, b: torch.Tensor, plain: torch.Tensor, out: torch.Tensor
) -> None:
    """One accumulator stored plainly and through the head split."""
    m, k = a.shape
    n = b.shape[1]
    for tile_y, tile_x in hl.tile([m, n]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.addmm(acc, a[tile_y, tile_r], b[tile_r, tile_x])
        plain[tile_y, tile_x] = acc.to(torch.bfloat16)
        y = _rows(tile_y)
        x = _cols(tile_x)
        out[y // 512, x // 64, y % 512, x % 64] = acc.to(torch.bfloat16)


# A [B * 4, M, K] x [B * 4, K, N] attention bmm whose output merges its four
# heads into the destination's last axis, scaled in the epilogue: a
# row-major [128, 64] tile of the row-major merge (N = 64), a 64-row head
# (M = 64) stored transposed through 64-row tiles.
HEAD_MERGE_CONFIG = helion.Config(
    block_sizes=[1, 128, 64, 64],
    tcgen05_cluster_m=1,
    tcgen05_ab_stages=2,
    pid_type="persistent_blocked",
    l2_groupings=[1],
    tcgen05_layout_strategy="explicit_epi_tile",
    tcgen05_layout_overrides_epi_tile_m=128,
    tcgen05_layout_overrides_epi_tile_n=32,
    tcgen05_layout_overrides_d_store_box_n=32,
)
TRANSPOSED_HEAD_MERGE_CONFIG = helion.Config(
    block_sizes=[1, 64, 128, 64],
    tcgen05_cluster_m=1,
    tcgen05_ab_stages=2,
    pid_type="flat",
    l2_groupings=[1],
    tcgen05_layout_strategy="explicit_epi_tile",
    tcgen05_layout_overrides_epi_tile_m=64,
    tcgen05_layout_overrides_epi_tile_n=32,
    tcgen05_layout_overrides_d_store_box_n=32,
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def head_merge_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """``out[z // 4, y, (z % 4) * N + x] = 0.125 * (a @ b)[z, y, x]``: the
    ``[B, S, 4 * N]`` merge of a ``[B * 4, S, N]`` attention output."""
    batch, m, k = a.shape
    n = b.shape[2]
    for tile_z, tile_y, tile_x in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_z, tile_y, tile_r], b[tile_z, tile_r, tile_x]
            )
        z = tile_z.index.reshape([tile_z, 1, 1])  # pyrefly: ignore[no-matching-overload]
        y = tile_y.index.reshape([1, tile_y, 1])  # pyrefly: ignore[no-matching-overload]
        x = tile_x.index.reshape([1, 1, tile_x])  # pyrefly: ignore[no-matching-overload]
        out[z // 4, y, (z % 4) * n + x] = (acc * 0.125).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def transposed_head_merge_bmm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """``out[z // 4, x, (z % 4) * M + y] = 0.125 * (a @ b)[z, y, x]``: each
    M-row head lands transposed, its rows contiguous."""
    batch, m, k = a.shape
    n = b.shape[2]
    for tile_z, tile_y, tile_x in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_z, tile_y, tile_r], b[tile_z, tile_r, tile_x]
            )
        z = tile_z.index.reshape([tile_z, 1, 1])  # pyrefly: ignore[no-matching-overload]
        y = tile_y.index.reshape([1, tile_y, 1])  # pyrefly: ignore[no-matching-overload]
        x = tile_x.index.reshape([1, 1, tile_x])  # pyrefly: ignore[no-matching-overload]
        out[z // 4, x, (z % 4) * m + y] = (acc * 0.125).to(torch.bfloat16)


def head_merge_args(
    batch: int = 2,
    m: int = 256,
    n: int = 64,
    k: int = 128,
    *,
    transposed: bool = False,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``a`` [B * 4, M, K], ``b`` [B * 4, K, N] and the merged destination:
    ``[B, M, 4 * N]``, or ``[B, N, 4 * M]`` for the transposed merge."""
    rows, cols = (n, 4 * m) if transposed else (m, 4 * n)
    return (
        torch.randn((batch * 4, m, k), dtype=torch.bfloat16, device=device),
        torch.randn((batch * 4, k, n), dtype=torch.bfloat16, device=device),
        torch.empty((batch, rows, cols), dtype=torch.bfloat16, device=device),
    )


def head_merge_reference(
    a: torch.Tensor, b: torch.Tensor, *, transposed: bool = False
) -> torch.Tensor:
    product = (0.125 * torch.bmm(a.float(), b.float())).to(torch.bfloat16)
    heads, m, n = product.shape
    split = product.view(heads // 4, 4, m, n)
    if transposed:
        return split.permute(0, 3, 1, 2).reshape(heads // 4, n, 4 * m)
    return split.transpose(1, 2).reshape(heads // 4, m, 4 * n)


def head_split_args(
    batch: int = 2,
    heads: int = 4,
    k: int = 256,
    *,
    seq: int = SEQ,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``a`` K-major, ``b`` K-major (NT) and the [batch, heads, seq, 64] output."""
    m, n = batch * seq, heads * HEAD
    return (
        torch.randn((m, k), dtype=torch.bfloat16, device=device),
        torch.randn((n, k), dtype=torch.bfloat16, device=device).T,
        torch.empty((batch, heads, seq, HEAD), dtype=torch.bfloat16, device=device),
    )


def head_split_reference(
    a: torch.Tensor, b: torch.Tensor, out_shape: torch.Size
) -> torch.Tensor:
    batch, heads, seq, head = out_shape
    return (
        (a.float() @ b.float())
        .to(torch.bfloat16)
        .view(batch, seq, heads, head)
        .permute(0, 2, 1, 3)
    )
