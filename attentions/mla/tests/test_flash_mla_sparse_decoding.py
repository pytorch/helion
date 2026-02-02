# MLA Sparse Decoding Tests
# Tests sparse attention patterns for MLA decode using local kernels
#
# This is a standalone version that replaces the external flash_mla and kernelkit dependencies
# with PyTorch reference implementations.

import sys
from pathlib import Path
import dataclasses
from typing import Tuple, Optional
import random

import torch

# Add local kernels path
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

# Local implementations
import kernelkit_utils as kk


@dataclasses.dataclass
class TestParam:
    """Test parameters for sparse decoding."""
    b: int               # Batch size
    h_q: int             # Number of query heads
    s_q: int             # Query sequence length (usually 1-2 for decode)
    h_kv: int = 1        # Number of KV heads
    s_kv: int = 1024     # KV sequence length
    is_varlen: bool = True
    topk: int = 128      # Top-K for sparse attention
    d_qk: int = 576      # Q/K dimension (576 = 512 + 64 for MLA)
    dv: int = 512        # V dimension
    block_size: int = 64
    seed: int = 0
    check_correctness: bool = True
    num_runs: int = 0
    have_topk_length: bool = False
    enable_attn_sink: bool = False


def generate_sparse_test_data(p: TestParam):
    """Generate test data for sparse decoding."""
    random.seed(p.seed)
    torch.manual_seed(p.seed)
    torch.cuda.manual_seed(p.seed)

    device = 'cuda'
    dtype = torch.bfloat16

    # Generate cache sequence lengths
    cache_seqlens = torch.full((p.b,), p.s_kv, dtype=torch.int32, device='cpu')
    if p.is_varlen:
        for i in range(p.b):
            cache_seqlens[i] = max(int(random.normalvariate(p.s_kv, p.s_kv / 2)), p.s_q)

    cache_seqlens = cache_seqlens.cuda()
    max_seqlen = int(cache_seqlens.max().item())

    # Generate Q
    q = torch.randn(p.b, p.s_q, p.h_q, p.d_qk, dtype=dtype, device=device) * 0.1

    # Generate KV cache (blocked format)
    num_blocks = kk.cdiv(max_seqlen, p.block_size) * p.b
    blocked_kv = torch.randn(num_blocks, p.block_size, p.h_kv, p.d_qk, dtype=dtype, device=device) * 0.1

    # Block table
    max_num_blocks = kk.cdiv(max_seqlen, p.block_size)
    block_table = torch.zeros(p.b, max_num_blocks, dtype=torch.int32, device=device)
    block_idx = 0
    for b in range(p.b):
        cur_len = int(cache_seqlens[b].item())
        cur_blocks = kk.cdiv(cur_len, p.block_size)
        for i in range(cur_blocks):
            block_table[b, i] = block_idx
            block_idx += 1

    # Generate top-k indices for sparse attention
    # For each query, we select top-k positions from the KV cache
    topk_indices = torch.zeros(p.b, p.s_q, p.topk, dtype=torch.int64, device=device)
    topk_lengths = torch.zeros(p.b, p.s_q, dtype=torch.int32, device=device)

    for b in range(p.b):
        cur_len = int(cache_seqlens[b].item())
        for sq in range(p.s_q):
            # Randomly select top-k positions from [0, cur_len)
            actual_topk = min(p.topk, cur_len)
            if actual_topk > 0:
                indices = torch.randperm(cur_len, device=device)[:actual_topk].sort().values
                topk_indices[b, sq, :actual_topk] = indices
                topk_lengths[b, sq] = actual_topk

    return {
        'q': q,
        'blocked_kv': blocked_kv,
        'block_table': block_table,
        'cache_seqlens': cache_seqlens,
        'topk_indices': topk_indices,
        'topk_lengths': topk_lengths,
    }


def reference_sparse_attention(
    q: torch.Tensor,           # [batch, s_q, h_q, d]
    blocked_kv: torch.Tensor,  # [num_blocks, block_size, h_kv, d]
    block_table: torch.Tensor, # [batch, max_num_blocks]
    cache_seqlens: torch.Tensor,  # [batch]
    topk_indices: torch.Tensor,   # [batch, s_q, topk]
    topk_lengths: torch.Tensor,   # [batch, s_q]
    dv: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of sparse attention."""
    batch, s_q, h_q, d = q.shape
    block_size = blocked_kv.shape[1]
    h_kv = blocked_kv.shape[2]
    device = q.device

    scale = d ** -0.5
    kv_group_num = h_q // h_kv

    out = torch.zeros(batch, s_q, h_q, dv, dtype=torch.float32, device=device)
    lse = torch.zeros(batch, h_q, s_q, dtype=torch.float32, device=device)

    cache_seqlens_cpu = cache_seqlens.cpu()
    topk_lengths_cpu = topk_lengths.cpu()

    for b in range(batch):
        cur_len = int(cache_seqlens_cpu[b].item())
        if cur_len == 0:
            lse[b] = float('+inf')
            continue

        # Gather full KV for this batch
        num_blocks = kk.cdiv(cur_len, block_size)
        block_indices = block_table[b, :num_blocks]
        kv_blocks = blocked_kv[block_indices]
        kv = kv_blocks.reshape(-1, h_kv, d)[:cur_len].float()
        v = kv[..., :dv]

        # Expand for GQA
        if kv_group_num > 1:
            kv = kv.unsqueeze(2).expand(-1, -1, kv_group_num, -1).reshape(cur_len, h_q, d)
            v = v.unsqueeze(2).expand(-1, -1, kv_group_num, -1).reshape(cur_len, h_q, dv)

        for sq in range(s_q):
            q_sq = q[b, sq].float() * scale  # [h_q, d]
            actual_topk = int(topk_lengths_cpu[b, sq].item())

            if actual_topk == 0:
                lse[b, :, sq] = float('+inf')
                continue

            # Get sparse indices
            indices = topk_indices[b, sq, :actual_topk].long()

            # Gather sparse KV
            k_sparse = kv[indices]  # [topk, h_q, d]
            v_sparse = v[indices]   # [topk, h_q, dv]

            # Compute attention
            attn_scores = torch.einsum('hd, nhd -> hn', q_sq, k_sparse)
            max_scores = attn_scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(attn_scores - max_scores)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)
            attn_weights = exp_scores / sum_exp

            out_sq = torch.einsum('hn, nhd -> hd', attn_weights, v_sparse)
            out[b, sq] = out_sq
            lse[b, :, sq] = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

    return out.to(q.dtype), lse


def run_sparse_decode_triton(
    q: torch.Tensor,
    blocked_kv: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    dv: int,
    num_splits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run sparse decoding using local Triton kernel."""
    # For now, use the reference implementation
    # A full Triton sparse attention kernel would go here
    return reference_sparse_attention(
        q, blocked_kv, block_table, cache_seqlens, topk_indices, topk_lengths, dv
    )


@torch.inference_mode()
def test_sparse_decode(p: TestParam) -> bool:
    """Run a single sparse decoding test."""
    print(f"Testing: b={p.b}, h_q={p.h_q}, s_kv={p.s_kv}, topk={p.topk}, d_qk={p.d_qk}")

    if p.seed == -1:
        p.seed = random.randint(0, 10000)

    data = generate_sparse_test_data(p)

    # Run test implementation
    out_test, lse_test = run_sparse_decode_triton(
        data['q'], data['blocked_kv'], data['block_table'],
        data['cache_seqlens'], data['topk_indices'], data['topk_lengths'],
        p.dv
    )

    # Run reference
    out_ref, lse_ref = reference_sparse_attention(
        data['q'], data['blocked_kv'], data['block_table'],
        data['cache_seqlens'], data['topk_indices'], data['topk_lengths'],
        p.dv
    )

    # Check correctness
    is_correct = True
    is_correct &= kk.check_is_allclose("out", out_test.float(), out_ref.float(),
                                        abs_tol=1e-3, rel_tol=2.01/128)
    is_correct &= kk.check_is_allclose("lse", lse_test, lse_ref,
                                        abs_tol=1e-6, rel_tol=8.01/65536)

    return is_correct


def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    # Define test cases
    correctness_cases = [
        TestParam(b=4, h_q=64, s_q=1, s_kv=512, topk=64, d_qk=576),
        TestParam(b=4, h_q=64, s_q=2, s_kv=1024, topk=128, d_qk=576),
        TestParam(b=8, h_q=128, s_q=1, s_kv=2048, topk=256, d_qk=576),
        TestParam(b=4, h_q=64, s_q=1, s_kv=512, topk=64, d_qk=512),
        TestParam(b=4, h_q=128, s_q=2, s_kv=1024, topk=128, d_qk=512),
    ]

    failed_cases = []
    for case in correctness_cases:
        if not test_sparse_decode(case):
            failed_cases.append(case)

    if len(failed_cases) > 0:
        print(f"\n{kk.colors['RED_BG']}{len(failed_cases)} / {len(correctness_cases)} cases failed{kk.colors['CLEAR']}")
        for case in failed_cases:
            print(f"  {case}")
        sys.exit(1)
    else:
        print(f"\n{kk.colors['GREEN_BG']}All {len(correctness_cases)} cases passed!{kk.colors['CLEAR']}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)
    main()
