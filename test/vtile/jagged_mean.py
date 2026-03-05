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

# 0.0072 RTX3090
@helion.kernel(config=helion.Config(block_sizes=[1, 32, 16], indexing=['block_ptr', 'pointer', 'block_ptr', 'pointer', 'pointer'], load_eviction_policies=['last', '', '', 'last'], num_stages=7, num_warps=1, pid_type='flat', range_flattens=[None, None, True], range_multi_buffers=[None, False, None], range_num_stages=[0, 4, 0], range_unroll_factors=[0, 0, 2], range_warp_specializes=[]), static_shapes=True)
def jagged_mean_kernel_japi_vtile_nomask(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,
    max_m: int,
) -> torch.Tensor:
    b = x_offsets.size(0) - 1
    x_flat = x_data.view(-1)
    out = torch.zeros([b, max_m], dtype=x_data.dtype, device=x_data.device)

    for tile_b in hl.tile(b):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        feature_counts = x_feature_counts[tile_b]

        for tile_m in hl.vtile(feature_counts):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)

            for tile_k in hl.vtile(nnz):
                flat_indices = (starts[:, None] + tile_k.index)[:, :, None] * max_m + (
                    tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_indices])
                row_sums = row_sums + x_slice.sum(dim=1)

            lengths_f = nnz.to(x_data.dtype)[:, None]
            result = torch.where(lengths_f > 0, row_sums / lengths_f, 0.0)
            out[tile_b, tile_m] = result

    return out


def reference_jagged_mean(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,
    max_m: int,
) -> torch.Tensor:
    b = x_offsets.numel() - 1
    out = torch.zeros((b, max_m), dtype=x_data.dtype, device=x_data.device)
    for i in range(b):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        f = int(x_feature_counts[i])
        if end > start and f > 0:
            out[i, :f] = x_data[start:end, :f].mean(dim=0)
    return out


def main() -> None:
    device = "cuda"
    dtype = torch.float32

    b, max_seq_len, m = 8, 16, 32
    offsets = _make_offsets(b, max_seq_len, device=device)
    total_len = int(offsets[-1].item())
    x_data = torch.randn(total_len, m, dtype=dtype, device=device)
    feature_counts = torch.randint(1, m + 1, (b,), dtype=torch.int32, device=device)

    out = jagged_mean_kernel_japi_vtile_nomask(x_data, offsets, feature_counts, m)
    ref = reference_jagged_mean(x_data, offsets, feature_counts, m)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
