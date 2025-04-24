from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def jagged_dense_bmm(A_values: torch.Tensor, A_offsets: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # m, k = x.size()
    # k2, n = y.size()
    # assert k == k2, f"size mismatch {k} != {k2}"
    # out = torch.empty(
    #     [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    # )
    # for tile_m, tile_n in hl.tile([m, n]):
    #     acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    #     for tile_k in hl.tile(k):
    #         acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
    #     out[tile_m, tile_n] = acc
    # return out

    # TODO: implement


def check_jagged_dense_bmm(b: int, k: int, n: int, max_length: int) -> None:
    # A: [sum_B(m), K], B: [B, K, N], C: [sum_B(m), N]
    from triton.testing import do_bench
    from .refs import jagged_refs
    
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.float16
    
    A_lengths = torch.randint(1, max_length + 1, (b,), device=device, dtype=torch.long)
    A_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0, dtype=torch.long)], dim=0)
    
    lengths_sum = int(A_lengths.sum().item())
    values = torch.rand(lengths_sum, k, device=device, dtype=dtype)
    dense_b = torch.rand((b, k, n), device=device, dtype=dtype)
    
    padded_values = jagged_refs.jagged_to_padded_dense(
        values,
        [offsets],
        max_lengths=[max_length],
        padding_value=0.0,
    )
    padded_ref = torch.bmm(padded_values, dense_b)
    ref_via_padded_op = jagged_refs.dense_to_jagged(padded_ref, [offsets])[0]
    ref_via_jagged = jagged_refs.jagged_dense_bmm(
        values, dense_b, offsets, max_length, allow_tf32=False, use_fbgemm_kernel=False
    )
    
    res_helion = jagged_dense_bmm(A_values, A_offsets, B)
    torch.testing.assert_close(res_helion, ref_via_jagged, 1e-5)
    torch.testing.assert_close(res_helion, ref_via_padded_op, 1e-5)
    sec = do_bench(lambda: jagged_dense_bmm(A_values, A_offsets, B))
    baseline_sec = do_bench(lambda: jagged_dense_bmm(A_values, A_offsets, B))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )

if __name__ == "__main__":
    check_jagged_dense_bmm(1024, 1024, 1024)
