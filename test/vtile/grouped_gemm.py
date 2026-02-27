from __future__ import annotations

import os

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False, autotune_effort="none")
def grouped_gemm_jagged_vtile_nomask(
    a_packed: torch.Tensor,
    b: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    total_m, k = a_packed.shape
    k2, n = b.shape
    assert k == k2

    out = torch.empty(
        total_m,
        n,
        dtype=torch.promote_types(a_packed.dtype, b.dtype),
        device=a_packed.device,
    )
    a_flat = a_packed.view(-1)
    out_flat = out.view(-1)
    g = group_offsets.size(0) - 1

    for tile_g in hl.tile(g):
        starts = group_offsets[tile_g]
        ends = group_offsets[tile_g.index + 1]
        group_sizes = ends - starts

        for tile_m in hl.vtile(group_sizes):
            row_idx = starts[:, None] + tile_m.index
            for tile_n in hl.tile(n):
                acc = hl.zeros([tile_g, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    a_idx = row_idx[:, :, None] * k + tile_k.index[None, None, :]
                    a_blk = hl.load(a_flat, [a_idx])
                    b_blk = b[tile_k, tile_n]
                    acc = acc + torch.matmul(a_blk, b_blk)
                out_idx = row_idx[:, :, None] * n + tile_n.index[None, None, :]
                hl.store(out_flat, [out_idx], acc.to(out.dtype))

    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def grouped_gemm_jagged_persistent_vtile_nomask(
    a_packed: torch.Tensor,
    b: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    device = a_packed.device
    if device.type == "xpu":
        num_workers = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "cuda":
        num_workers = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    else:
        num_workers = 1

    block_m = hl.register_block_size(32, 128)
    block_n = hl.register_block_size(32, 128)

    total_m, k = a_packed.shape
    k2, n = b.shape
    assert k == k2

    out = torch.zeros(
        total_m,
        n,
        dtype=torch.promote_types(a_packed.dtype, b.dtype),
        device=a_packed.device,
    )

    g = group_offsets.size(0) - 1
    for worker_id in hl.grid(num_workers):
        for group_id in hl.grid(g):
            start = group_offsets[group_id]
            end = group_offsets[group_id + 1]
            m_size = end - start
            if m_size > 0:
                num_m_tiles = (m_size + block_m - 1) // block_m
                num_n_tiles = (n + block_n - 1) // block_n
                num_group_tiles = num_m_tiles * num_n_tiles

                for local_tile in hl.grid(num_group_tiles):
                    tile_in_group = local_tile * num_workers + worker_id
                    if tile_in_group < num_group_tiles:
                        m_tile_idx = tile_in_group % num_m_tiles
                        n_tile_idx = tile_in_group // num_m_tiles
                        m_start = m_tile_idx * block_m
                        n_start = n_tile_idx * block_n

                        m_remaining = m_size - m_start
                        if m_remaining > block_m:
                            m_block = block_m
                        else:
                            m_block = m_remaining

                        if n_start + block_n <= n:
                            n_end = n_start + block_n
                        else:
                            n_end = n

                        for tile_m in hl.vtile(m_block):
                            row_idx = start + m_start + tile_m.index
                            for tile_n in hl.tile(n_start, n_end):
                                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                                for tile_k in hl.tile(k):
                                    a_blk = a_packed[row_idx, tile_k]
                                    b_blk = b[tile_k, tile_n]
                                    acc = torch.addmm(acc, a_blk, b_blk)
                                out[row_idx, tile_n] = acc.to(out.dtype)

    return out


def _pack_group_inputs(
    group_a: list[torch.Tensor], group_b: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(group_a) > 0
    device = group_a[0].device
    dtype = group_a[0].dtype
    b_shared = group_b[0]

    m_sizes = [int(a.size(0)) for a in group_a]
    starts = [0]
    for m in m_sizes:
        starts.append(starts[-1] + m)
    group_offsets = torch.tensor(starts, device=device, dtype=torch.int32)
    a_packed = torch.cat(group_a, dim=0).to(device=device, dtype=dtype).contiguous()
    return a_packed, b_shared, group_offsets


def reference_grouped_gemm(
    group_a: list[torch.Tensor], group_b: list[torch.Tensor]
) -> torch.Tensor:
    b_shared = group_b[0]
    outs = [a @ b_shared for a in group_a]
    return torch.cat(outs, dim=0)


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float32
    g = 4
    k, n = 64, 48

    group_a = [
        torch.randn(16 * (i + 1), k, device=device, dtype=dtype).contiguous()
        for i in range(g)
    ]
    group_b = [torch.randn(k, n, device=device, dtype=dtype).contiguous()] * g
    a_packed, b_shared, group_offsets = _pack_group_inputs(group_a, group_b)

    out = grouped_gemm_jagged_vtile_nomask(a_packed, b_shared, group_offsets)
    out_persistent = grouped_gemm_jagged_persistent_vtile_nomask(
        a_packed, b_shared, group_offsets
    )
    ref = reference_grouped_gemm(group_a, group_b)

    max_diff = (out - ref).abs().max().item()
    max_diff_persistent = (out_persistent - ref).abs().max().item()
    print(f"max_diff={max_diff}")
    print(f"max_diff_persistent={max_diff_persistent}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
