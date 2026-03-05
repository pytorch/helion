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


@helion.kernel(config=helion.Config(block_sizes=[1, 16, 16, 16, 16], indexing=['pointer', 'pointer', 'pointer', 'block_ptr', 'block_ptr', 'block_ptr'], load_eviction_policies=['', 'first', 'first', '', 'first'], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, None, True, None, True], range_multi_buffers=[None, None, None, False, None], range_num_stages=[0, 3, 0, 0, 4], range_unroll_factors=[0, 3, 0, 1, 0], range_warp_specializes=[]), static_shapes=True)
def jagged_hstu_attention_japi_vtile_nomask(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0 / max_seq_len
    b = seq_offsets.size(0) - 1
    d = hl.specialize(v.size(1))
    q_flat = q.view(-1)
    k_flat = k.view(-1)
    v_flat = v.view(-1)
    out = torch.zeros_like(v)
    out_flat = out.view(-1)

    for tile_b in hl.tile(b):
        starts = seq_offsets[tile_b]
        ends = seq_offsets[tile_b.index + 1]
        lengths = ends - starts

        for tile_q in hl.vtile(lengths):
            q_base = starts[:, None] + tile_q.index
            for tile_dv in hl.tile(d):
                acc = hl.zeros([tile_b, tile_q, tile_dv], dtype=torch.float32)

                for tile_kv in hl.vtile(lengths):
                    scores = hl.zeros([tile_b, tile_q, tile_kv], dtype=torch.float32)
                    kv_base = starts[:, None] + tile_kv.index

                    for tile_d in hl.tile(d):
                        q_idx = q_base[:, :, None] * d + tile_d.index[None, None, :]
                        q_blk = hl.load(q_flat, [q_idx])

                        k_idx = kv_base[:, :, None] * d + tile_d.index[None, None, :]
                        k_blk = hl.load(k_flat, [k_idx])

                        scores = scores + torch.matmul(
                            q_blk.to(torch.float32),
                            k_blk.transpose(1, 2).to(torch.float32),
                        )

                    scores = torch.nn.functional.silu(scores * alpha) * scale
                    causal = tile_q.index[:, None] > tile_kv.index[None, :]
                    scores = torch.where(causal[None, :, :], scores, 0.0)

                    v_idx = kv_base[:, :, None] * d + tile_dv.index[None, None, :]
                    v_blk = hl.load(v_flat, [v_idx])
                    acc = acc + torch.matmul(scores.to(v.dtype), v_blk)

                out_idx = q_base[:, :, None] * d + tile_dv.index[None, None, :]
                hl.store(out_flat, [out_idx], acc.to(out.dtype))

    return out


def reference_jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(v)
    scale = 1.0 / max_seq_len
    b = seq_offsets.numel() - 1

    for i in range(b):
        start = int(seq_offsets[i])
        end = int(seq_offsets[i + 1])
        if end <= start:
            continue
        q_blk = q[start:end]
        k_blk = k[start:end]
        v_blk = v[start:end]

        scores = torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha) * scale
        causal = torch.tril(torch.ones_like(scores, dtype=torch.bool), diagonal=-1)
        scores = torch.where(causal, scores, torch.zeros_like(scores))
        out[start:end] = torch.matmul(scores.to(v.dtype), v_blk)

    return out


def main() -> None:
    device = "cuda"
    dtype = torch.float32

    b, max_seq_len, d = 8, 16, 16
    offsets = _make_offsets(b, max_seq_len, device=device)
    total_len = int(offsets[-1].item())
    alpha = 1.0 / (d * d)

    q = torch.randn(total_len, d, dtype=dtype, device=device)
    k = torch.randn(total_len, d, dtype=dtype, device=device)
    v = torch.randn(total_len, d, dtype=dtype, device=device)

    out = jagged_hstu_attention_japi_vtile_nomask(max_seq_len, alpha, q, k, v, offsets)
    ref = reference_jagged_hstu_attention(max_seq_len, alpha, q, k, v, offsets)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
