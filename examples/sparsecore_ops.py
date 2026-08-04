"""Representative SparseCore workloads.

Dynamic jagged tiles, quantized cached inputs, and matrix products remain
unsupported.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(backend="pallas", static_shapes=True)
def gather_reduce(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather and sum several table entries per item."""
    B, K = idx.size()
    out = torch.empty([B, table.size(1)], dtype=table.dtype, device=table.device)
    for tile_b in hl.tile(B):
        out[tile_b, :] = table[idx[tile_b, :], :].sum(dim=1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def moe_combine(
    expert_out: torch.Tensor, idx: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Top-K weighted combine (tokamax ragged_gather_reduce core math)."""
    B, K = idx.size()
    out = torch.empty(
        [B, expert_out.size(1)], dtype=expert_out.dtype, device=expert_out.device
    )
    for tile_b in hl.tile(B):
        values = expert_out[idx[tile_b, :], :]
        out[tile_b, :] = (values * w[tile_b, :, None]).sum(dim=1)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def ragged_gather(tokens: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    """Gather tokens using precomputed routing IDs."""
    N = item_ids.size(0)
    out = torch.empty([N, tokens.size(1)], dtype=tokens.dtype, device=tokens.device)
    for tile in hl.tile(N):
        out[tile, :] = tokens[item_ids[tile], :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def ragged_scatter(x: torch.Tensor, dest: torch.Tensor, out_items: int) -> torch.Tensor:
    """Covering token permutation scatter."""
    N, D = x.size()
    out = torch.empty([out_items, D], dtype=x.dtype, device=x.device)
    for tile in hl.tile(N):
        out[dest[tile], :] = x[tile, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def gelu_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """GELU and multiply over direct streams."""
    out = torch.empty_like(x)
    N = x.size(0)
    for tile in hl.tile(N):
        out[tile, :] = (
            torch.nn.functional.gelu(x[tile, :], approximate="tanh") * y[tile, :]
        )
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmax int8 quantization with a scale per item."""
    N, D = x.size()
    q = torch.empty([N, D], dtype=torch.int8, device=x.device)
    scales = torch.empty([N, 1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(N):
        values = x[tile, :]
        s = values.abs().amax(dim=1, keepdim=True) / 127.0
        q[tile, :] = torch.clamp(torch.round(values / s), -128, 127).to(torch.int8)
        scales[tile, :] = s
    return q, scales


@helion.kernel(backend="pallas", static_shapes=True)
def block_mask(scores: torch.Tensor, tau: hl.constexpr) -> torch.Tensor:
    """Build a routing mask from each item's maximum score."""
    NB = scores.size(0)
    mask = torch.empty([NB], dtype=torch.bool, device=scores.device)
    for tile in hl.tile(NB):
        mask[tile] = scores[tile, :].amax(dim=1) > tau  # pyrefly: ignore [unsupported-operation]
    return mask


@helion.kernel(backend="pallas", static_shapes=True)
def masked_gather_sum(
    x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Gather and reduce using host-prepared IDs and masks."""
    S, L = ids.size()
    out = torch.empty([S, x.size(1)], dtype=x.dtype, device=x.device)
    for tile_s in hl.tile(S):
        values = x[ids[tile_s, :], :]
        out[tile_s, :] = (values * w[tile_s, :, None]).sum(dim=1)
    return out


def jagged_sum(
    x_data: torch.Tensor, x_offsets: torch.Tensor, max_len: int
) -> torch.Tensor:
    """Prepare IDs and masks, then run the static gather kernel."""
    counts = x_offsets[1:] - x_offsets[:-1]
    pos = torch.arange(max_len, dtype=torch.int32, device=x_offsets.device)
    ids = torch.clamp(x_offsets[:-1, None] + pos[None, :], max=x_data.size(0) - 1).to(
        torch.int32
    )
    w = (pos[None, :] < counts[:, None]).to(x_data.dtype)
    return masked_gather_sum(x_data, ids, w)


def main() -> None:
    g = torch.Generator().manual_seed(0)
    V, D, B, K, N = 2048, 64, 1024, 8, 1024
    table = torch.randn(V, D, generator=g)
    idx = torch.randint(0, V, (B, K), dtype=torch.int32, generator=g)
    w = torch.rand(B, K, generator=g)
    tokens = torch.randn(N, D, generator=g)
    item_ids = torch.randperm(N, generator=g).to(torch.int32)
    x = torch.randn(N, 128, generator=g)
    y = torch.randn(N, 128, generator=g)

    run_example(gather_reduce, lambda t, i: t[i.long()].sum(dim=1), (table, idx))
    run_example(
        moe_combine,
        lambda t, i, ww: (t[i.long()] * ww[..., None]).sum(dim=1),
        (table, idx, w),
    )
    run_example(ragged_gather, lambda t, r: t[r.long()], (tokens, item_ids))
    run_example(
        gelu_multiply,
        lambda a, b: torch.nn.functional.gelu(a, approximate="tanh") * b,
        (x, y),
    )


if __name__ == "__main__":
    main()
