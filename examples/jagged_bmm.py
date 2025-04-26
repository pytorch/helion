from __future__ import annotations

import torch

import helion
import helion.language as hl

# from torchrec.sparse.jagged_tensor import JaggedTensor
# from torch.nested._internal.nested_tensor import NestedTensor


# NOTE: We want to use `unified_bmm` to represent all dense bmm and jagged bmm use cases.
@helion.kernel()
def unified_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [sum_B(Mi), K], B: [B, K, N], Out: [sum_B(Mi), N]   # jagged-dense bmm
    # A: [sum_B(Mi), K], B: [K, sum_B(Ni)], Out: [sum_B(Mi * Ni)]   # jagged-jagged bmm jagged out
    # A: [M, sum_B(Ki)], B: [sum_B(Ki), N], Out: [B, M, N]   # jagged-jagged bmm dense out
    # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm

    # Infer the size of each dim (some are potentially jagged dims)
    b, m, n, k, out_shape = infer_dims_for_bmm(A, B)

    # Create the output tensor (potentially jagged)
    out = torch.empty(out_shape, device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))

    for bi in range(b):
        for tile_n, tile_m in hl.tile([n, m]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            # A[bi] / B[bi] / out[bi] gives the ith batch-slice of the tensor, regardless of which dim is the batch dim.
            # For regular dense tensor, the first dim is assumed to be the batch dim.
            # For dense tensor with batch dim != 0, use tensor subclass and override `def batch_dim()` to specify batch dim.
            # For jagged tensor, override `def batch_dim()` to specify batch dim.
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, A[bi][tile_m, tile_k], B[bi][tile_k, tile_n])
            out[bi][tile_m, tile_n] = acc
    return out



# def check_jagged_dense_bmm(b: int, k: int, n: int, max_length: int, jagged_tensor_type: Any) -> None:
#     # A: [sum_B(m), K], B: [B, K, N], Out: [sum_B(m), N]
#     from triton.testing import do_bench
#     from .refs import jagged_refs

#     assert jagged_tensor_type in [JaggedTensor, NestedTensor]
    
#     torch.manual_seed(123)
#     device = torch.device("cuda")
#     dtype = torch.float16
    
#     A_lengths = torch.randint(1, max_length + 1, (b,), device=device, dtype=torch.long)
#     A_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(A_lengths, dim=0, dtype=torch.long)], dim=0)
    
#     A_lengths_sum = int(A_lengths.sum().item())
#     A_values = torch.rand(A_lengths_sum, k, device=device, dtype=dtype)
#     B = torch.rand((b, k, n), device=device, dtype=dtype)
    
#     padded_values = jagged_refs.jagged_to_padded_dense(
#         A_values,
#         [A_offsets],
#         max_lengths=[max_length],
#         padding_value=0.0,
#     )
#     padded_ref = torch.bmm(padded_values, B)
#     ref_via_padded_op = jagged_refs.dense_to_jagged(padded_ref, [A_offsets])[0]
#     ref_via_jagged = jagged_refs.jagged_dense_bmm(
#         A_values, B, A_offsets, max_length, allow_tf32=False, use_fbgemm_kernel=False
#     )
    
#     A_tensor = jagged_tensor_type(values=A_values, offsets=A_offsets)

#     res_helion = bmm(A_tensor, B)
#     torch.testing.assert_close(res_helion, ref_via_jagged, 1e-5)
#     torch.testing.assert_close(res_helion, ref_via_padded_op, 1e-5)
#     sec = do_bench(lambda: bmm(A_tensor, B))
#     baseline_sec = do_bench(lambda: jagged_refs.jagged_dense_bmm(
#         A_values, B, A_offsets, max_length, allow_tf32=False, use_fbgemm_kernel=False
#     ))
#     print(
#         f"Helion time: {sec:.4f}s, custom triton time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
#     )

# if __name__ == "__main__":
#     for jagged_tensor_type in [JaggedTensor, NestedTensor]:
#         check_jagged_dense_bmm(1024, 1024, 1024, 256, jagged_tensor_type)
