from __future__ import annotations

import itertools
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
def jagged_softmax_kernel_japi_vtile_nomask(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    b = x_offsets.size(0) - 1
    m = x_data.size(1)
    x_flat = x_data.view(-1)
    out_data = torch.zeros_like(x_data)
    out_flat = out_data.view(-1)

    for tile_b in hl.tile(b):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        lengths = ends - starts

        for tile_m in hl.tile(m):
            block_max = hl.full([tile_b, tile_m], float("-inf"), dtype=x_data.dtype)
            block_l = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)

            for tile_k in hl.vtile(lengths):
                base = starts[:, None] + tile_k.index
                flat_idx = base[:, :, None] * m + tile_m.index[None, None, :]
                x_slice = hl.load(x_flat, [flat_idx])
                slice_max = x_slice.amax(dim=1)
                new_max = torch.maximum(block_max, slice_max)
                block_l = block_l * torch.exp(block_max - new_max)
                block_l = block_l + torch.exp(x_slice - new_max[:, None, :]).sum(dim=1)
                block_max = new_max

            for tile_k in hl.vtile(lengths):
                base = starts[:, None] + tile_k.index
                flat_idx = base[:, :, None] * m + tile_m.index[None, None, :]
                x_slice = hl.load(x_flat, [flat_idx])
                result = (
                    torch.exp(x_slice - block_max[:, None, :]) / block_l[:, None, :]
                )
                hl.store(out_flat, [flat_idx], result)

    return out_data


def reference_jagged_softmax(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    for start, end in itertools.pairwise(x_offsets):
        chunks.append(torch.softmax(x_data[start:end], dim=0))
    return torch.cat(chunks, dim=0)


def main() -> None:
    device = "cuda"
    dtype = torch.float32

    b, max_seq_len, m = 8, 16, 32
    offsets = _make_offsets(b, max_seq_len, device=device)
    total_len = int(offsets[-1].item())
    x_data = torch.randn(total_len, m, dtype=dtype, device=device)

    out = jagged_softmax_kernel_japi_vtile_nomask(x_data, offsets)
    ref = reference_jagged_softmax(x_data, offsets)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
