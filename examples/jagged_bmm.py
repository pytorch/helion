from __future__ import annotations

import torch

import helion
import helion.language as hl

# from torchrec.sparse.jagged_tensor import JaggedTensor
# from torch.nested._internal.nested_tensor import NestedTensor

"""
Jagged BMM variants we want to support:
A: [sum_B(Mi), K], B: [B, K, N], Out: [sum_B(Mi), N]   # jagged-dense bmm
A: [sum_B(Mi), K], B: [K, sum_B(Ni)], Out: [sum_B(Mi * Ni)]   # jagged-jagged bmm jagged out
A: [M, sum_B(Ki)], B: [sum_B(Ki), N], Out: [B, M, N]   # jagged-jagged bmm dense out

High-level design ideas:
- Since the input/output shapes for each jagged BMM variant are different, we choose to
  write a different kernel for each variant instead of having a single "unified" kernel
  to represent all the them. This makes the kernel easier to read and understand, and also avoid
  needing magic functions to extract the right shape information from the input tensors or
  compute the right output shape.
"""


# NOTE: `dense_bmm` doesn't use jagged tensors. Having this here just for comparison.
@helion.kernel()
def dense_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty([b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel()
def jagged_dense_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [sum_B(Mi), K], B: [B, K, N], Out: [sum_B(Mi), N]   # jagged-dense bmm
    b, m = A.size(0)  # unpacking a jagged dim
    k = A.size(1)
    n = B.size(2)
    assert k == B.size(1), f"size mismatch {k} != {B.size(1)}"
    # This creates 2D jagged tensor of shape [sum_B(Mi), N]
    # NOTE: `(a, b)` represents a composite dim (jagged dim in this case)
    out = type(A).empty([(b, m), n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    for bi in range(b):
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, A[(bi, tile_m), tile_k], B[bi, tile_k, tile_n])
            out[(bi, tile_m), tile_n] = acc
    return out


@helion.kernel()
def jagged_jagged_bmm_jagged_out(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [sum_B(Mi), K], B: [K, sum_B(Ni)], Out: [sum_B(Mi * Ni)]   # jagged-jagged bmm jagged out
    b, m = A.size(0)
    b_from_B, n = B.size(1)
    k = A.size(1)
    k_from_B = B.size(0)
    assert b == b_from_B, f"size mismatch {b} != {b_from_B}"
    assert k == k_from_B, f"size mismatch {k} != {k_from_B}"
    # This creates jagged tensor: [M1*N1, M2*N2, ..., MB*NB] i.e. a 1D jagged tensor
    out = type(A).empty([(b, m * n)], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    for bi in range(b):
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, A[(bi, tile_m), tile_k], B[tile_k, (bi, tile_n)])
            # Each slice of the `out` jagged tensor is a 1D tensor of length Mi * Ni,
            # so we need to unflatten it first and then do the `acc` copying.
            out[bi].view(m, n)[tile_m, tile_n] = acc
    return out


@helion.kernel()
def jagged_jagged_bmm_dense_out(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, sum_B(Ki)], B: [sum_B(Ki), N], Out: [B, M, N]   # jagged-jagged bmm dense out
    m = A.size(0)
    n = B.size(1)
    b, k = A.size(1)
    b_from_B, k_from_B = B.size(0)
    assert b == b_from_B, f"size mismatch {b} != {b_from_B}"
    assert k == k_from_B, f"size mismatch {k} != {k_from_B}"
    out = torch.empty([b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    for bi in range(b):
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, A[tile_m, (bi, tile_k)], B[(bi, tile_k), tile_n])
            out[bi, tile_m, tile_n] = acc
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
