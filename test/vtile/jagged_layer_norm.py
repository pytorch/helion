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

# 0.0051 RTX3090
@helion.kernel
def jagged_layer_norm_kernel_japi_vtile_nomask(
    x_values: torch.Tensor,
    x_offsets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    b = x_offsets.size(0) - 1
    m = x_values.size(1)
    x_flat = x_values.view(-1)
    out_values = torch.empty_like(x_values)
    out_flat = out_values.view(-1)

    for tile_b in hl.tile(b):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        lengths = ends - starts

        mean_acc = hl.zeros([tile_b], dtype=x_values.dtype)
        for tile_m in hl.tile(m):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.vtile(lengths):
                flat_idx = (starts[:, None] + tile_k.index)[:, :, None] * m + (
                    tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_idx])
                row_sums = row_sums + x_slice.sum(dim=1)
            mean_acc = mean_acc + row_sums.sum(dim=1)
        mean_acc = mean_acc / (lengths.to(x_values.dtype) * m)

        var_acc = hl.zeros([tile_b], dtype=x_values.dtype)
        for tile_m in hl.tile(m):
            var_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.vtile(lengths):
                flat_idx = (starts[:, None] + tile_k.index)[:, :, None] * m + (
                    tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_idx])
                centered = x_slice.to(torch.float32) - mean_acc[:, None, None]
                var_sums = var_sums + (centered * centered).sum(dim=1)
            var_acc = var_acc + var_sums.sum(dim=1)
        rstd = torch.rsqrt(var_acc / (lengths.to(x_values.dtype) * m) + eps)

        for tile_m in hl.tile(m):
            for tile_k in hl.vtile(lengths):
                flat_idx = (starts[:, None] + tile_k.index)[:, :, None] * m + (
                    tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_idx])
                normalized = (
                    x_slice.to(torch.float32) - mean_acc[:, None, None]
                ) * rstd[:, None, None]
                hl.store(out_flat, [flat_idx], normalized.to(x_values.dtype))

    return out_values


def reference_jagged_layer_norm(
    x_values: torch.Tensor,
    x_offsets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    out = torch.empty_like(x_values)
    b = x_offsets.numel() - 1
    for i in range(b):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        x = x_values[start:end]
        mean = x.mean()
        var = x.var(unbiased=False)
        out[start:end] = (x - mean) * torch.rsqrt(var + eps)
    return out


def main() -> None:
    device = "cuda"
    dtype = torch.float32

    b, max_seq_len, m = 8, 16, 32
    offsets = _make_offsets(b, max_seq_len, device=device)
    total_len = int(offsets[-1].item())
    x_values = torch.randn(total_len, m, dtype=dtype, device=device)

    out = jagged_layer_norm_kernel_japi_vtile_nomask(x_values, offsets, eps=1e-6)
    ref = reference_jagged_layer_norm(x_values, offsets, eps=1e-6)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
