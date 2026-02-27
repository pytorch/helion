from __future__ import annotations

import os

import torch

import helion
import helion.language as hl


def _make_offsets(
    batch_size: int,
    max_seq_len: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    lengths = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(lengths, dim=0),
        ]
    )


@helion.kernel(autotune_effort="none")
def jagged_dense_add_2d_japi_vtile_nomask(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    b, n = y.size(0), y.size(1)
    out = torch.empty_like(y)
    y_flat = y.view(-1)
    out_flat = out.view(-1)

    for tile_b in hl.tile(b):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts

        # Prefix region [0:nnz): out = y + jagged
        for tile_j in hl.vtile(nnz):
            x_idx = starts[:, None] + tile_j.index[None, :]
            x_slice = hl.load(x_data, [x_idx])
            out[tile_b, tile_j] = y[tile_b, tile_j] + x_slice

        # Tail region [nnz:n): out = y
        tail = n - nnz
        for tile_j in hl.vtile(tail):
            col = nnz[:, None] + tile_j.index[None, :]
            flat_idx = tile_b.index[:, None] * n + col
            y_slice = hl.load(y_flat, [flat_idx])
            hl.store(out_flat, [flat_idx], y_slice)

    return out


def reference_jagged_dense_add_2d(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    out = y.clone()
    b = x_offsets.numel() - 1
    for i in range(b):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        out[i, : end - start] += x_data[start:end]
    return out


def main() -> None:
    device = "cuda"
    dtype = torch.float32

    b, max_seq_len, n = 8, 16, 32
    offsets = _make_offsets(b, max_seq_len, device=device)
    total_len = int(offsets[-1].item())

    x_data = torch.randn(total_len, dtype=dtype, device=device)
    y = torch.randn(b, n, dtype=dtype, device=device)

    out = jagged_dense_add_2d_japi_vtile_nomask(x_data, offsets, y)
    ref = reference_jagged_dense_add_2d(x_data, offsets, y)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
