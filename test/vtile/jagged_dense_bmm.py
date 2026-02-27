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
def jagged_dense_bmm_japi_vtile_nomask(
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    total_len, d = jagged.shape
    b = seq_offsets.size(0) - 1
    k = dense.size(2)
    dtype = torch.promote_types(jagged.dtype, dense.dtype)
    jagged_flat = jagged.view(-1)

    output = torch.empty((total_len, k), dtype=dtype, device=jagged.device)
    out_flat = output.view(-1)

    for tile_b in hl.tile(b):
        starts = seq_offsets[tile_b]
        ends = seq_offsets[tile_b.index + 1]
        lane_lengths = ends - starts

        for tile_len in hl.vtile(lane_lengths):
            for tile_k in hl.tile(k):
                acc = hl.zeros([tile_b, tile_len, tile_k], dtype=dtype)
                for tile_d in hl.tile(d):
                    base = starts[:, None] + tile_len.index
                    jagged_idx = base[:, :, None] * d + tile_d.index[None, None, :]
                    j_data = hl.load(jagged_flat, [jagged_idx])
                    d_data = dense[tile_b, tile_d, tile_k]
                    acc = acc + torch.matmul(j_data, d_data)
                if bias is not None:
                    acc = acc + bias[tile_b, tile_k].unsqueeze(1)

                base = starts[:, None] + tile_len.index[None, :]
                flat_idx = base[:, :, None] * k + tile_k.index[None, None, :]
                hl.store(out_flat, [flat_idx], acc)

    return output


def reference_jagged_dense_bmm(
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    total_len, _ = jagged.shape
    b, _, k = dense.shape
    out = torch.empty((total_len, k), dtype=jagged.dtype, device=jagged.device)
    for i in range(b):
        start = int(seq_offsets[i])
        end = int(seq_offsets[i + 1])
        if end > start:
            y = torch.matmul(jagged[start:end], dense[i])
            if bias is not None:
                y = y + bias[i].unsqueeze(0)
            out[start:end] = y
    return out


def main() -> None:
    device = "cuda"
    dtype = torch.float32

    b, max_seq_len, d, k = 8, 16, 12, 10
    offsets = _make_offsets(b, max_seq_len, device=device)
    total_len = int(offsets[-1].item())

    jagged = torch.randn(total_len, d, dtype=dtype, device=device)
    dense = torch.randn(b, d, k, dtype=dtype, device=device)
    bias = torch.randn(b, k, dtype=dtype, device=device)

    out = jagged_dense_bmm_japi_vtile_nomask(offsets, jagged, dense, bias)
    ref = reference_jagged_dense_bmm(offsets, jagged, dense, bias)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
