# === HELION KERNEL REPRO ===
import helion
import helion.language as hl
import torch
from torch._dynamo.testing import rand_strided

@helion.kernel(config=helion.Config(block_sizes=[2, 4, 2], indexing='pointer', load_eviction_policies=[], loop_orders=[[0, 1]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, None, None, None], range_multi_buffers=[None, None, None, None], range_num_stages=[0, 0, 0, 0], range_unroll_factors=[0, 0, 0, 0], range_warp_specializes=[], static_ranges=[False, False, False]), static_shapes=True)
def helion_gdpa_kernel(
    q: torch.Tensor,
    q_offsets: torch.Tensor,
    k: torch.Tensor,
    k_offsets: torch.Tensor,
    v: torch.Tensor,
    max_seq_len_q: hl.constexpr,
    max_seq_len_kv: hl.constexpr,
    activation_enum_int: hl.constexpr,
    qk_scale: hl.constexpr,
    broadcast_q: hl.constexpr,
    # USE_I64_IDX: hl.constexpr,
) -> torch.Tensor:
    if broadcast_q:
        BATCH = k_offsets.size(0) - 1
    else:
        BATCH = q_offsets.size(0) - 1

    d = q.size(2)
    h = q.size(1)

    output = torch.zeros_like(q)
    head_idx = torch.arange(h, dtype=q_offsets.dtype, device=q.device)

    # Tile over batch, head, sequence
    for tile_b in hl.tile(BATCH, block_size=1):
        q_start = q_offsets[tile_b.begin]
        q_end = q_offsets[tile_b.begin + 1]
        q_batch = q_end - q_start

        q_start, q_end = q_offsets[tile_b.begin], q_offsets[tile_b.begin + 1]
        k_start, k_end = k_offsets[tile_b.begin], k_offsets[tile_b.begin + 1]

        q_batch = q[tile_b.index + q_start, :, :]  # [seq_len_q, H, D]
        k_batch = k[tile_b.index + k_start, :, :]  # [seq_len_k, H, D]
        v_batch = v[tile_b.index + k_start, :, :]  # [seq_len_k, H, D]

        # Permute to [H, seq_len, D] to process each head independently
        q_batch = q_batch.permute(1, 0, 2)  # [H, seq_len_q, D]
        k_batch = k_batch.permute(1, 0, 2)  # [H, seq_len_k, D]
        v_batch = v_batch.permute(1, 0, 2)  # [H, seq_len_k, D]

        for tile_m, tile_d in hl.tile([max_seq_len_q, d]):
            q_blk = q_batch[:, tile_m, tile_d]
            acc = hl.zeros([h, tile_m, tile_d], dtype=v.dtype)

            for tile_n in hl.tile(max_seq_len_kv):
                k_blk = k_batch[:, tile_n, tile_d]
                v_blk = v_batch[:, tile_n, tile_d]

                qk = torch.bmm(q_blk, k_blk.transpose(1, 2))  #

                if activation_enum_int == 0:
                    p = qk
                elif activation_enum_int == 3:
                    p = (
                        0.5
                        * qk
                        * (
                            1.0
                            + torch.tanh(0.7978845608 * qk * (1.0 + 0.044715 * qk * qk))
                        )
                    )
                else:
                    p = qk

                p *= qk_scale
                p = p.to(v.dtype)

                acc = torch.baddbmm(acc, p, v_blk)

            seq_idx = q_start + tile_m.index
            dim_idx = tile_d.index
            seq_mask = seq_idx < q_end

            output_block = acc.permute(1, 0, 2).contiguous().to(output.dtype)
            hl.store(
                output,
                [seq_idx, head_idx, dim_idx],
                output_block,
                extra_mask=seq_mask[:, None, None],
            )

    return output

def helion_repro_caller():
    torch.manual_seed(0)
    q = rand_strided((6, 3, 4), (12, 4, 1), dtype=torch.bfloat16, device='cuda:0')
    q.requires_grad_(True)
    q_offsets = rand_strided((4,), (1,), dtype=torch.int64, device='cuda:0')
    k = rand_strided((6, 3, 4), (12, 4, 1), dtype=torch.bfloat16, device='cuda:0')
    k.requires_grad_(True)
    k_offsets = rand_strided((4,), (1,), dtype=torch.int64, device='cuda:0')
    v = rand_strided((6, 3, 4), (12, 4, 1), dtype=torch.bfloat16, device='cuda:0')
    v.requires_grad_(True)
    max_seq_len_q = 2
    max_seq_len_kv = 2
    activation_enum_int = 3
    qk_scale = 1.0
    broadcast_q = False
    return helion_gdpa_kernel(q, q_offsets, k, k_offsets, v, max_seq_len_q, max_seq_len_kv, activation_enum_int, qk_scale, broadcast_q)

helion_repro_caller()
# === END HELION KERNEL REPRO ===
