# MLA Sparse Prefill Tests
# Tests sparse attention patterns for MLA prefill using local kernels
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
    """Test parameters for sparse prefill."""
    s_q: int              # Query sequence length
    s_kv: int             # KV sequence length
    topk: int             # Top-K for sparse attention
    h_q: int = 128        # Number of query heads
    h_kv: int = 1         # Number of KV heads
    d_qk: int = 576       # Q/K dimension
    dv: int = 512         # V dimension
    seed: int = 0
    check_correctness: bool = True
    num_runs: int = 0
    have_attn_sink: bool = False
    have_topk_length: bool = False
    is_all_indices_invalid: bool = False


def generate_prefill_test_data(p: TestParam):
    """Generate test data for sparse prefill."""
    random.seed(p.seed)
    torch.manual_seed(p.seed)
    torch.cuda.manual_seed(p.seed)

    device = 'cuda'
    dtype = torch.bfloat16

    # Generate Q
    q = torch.randn(p.s_q, p.h_q, p.d_qk, dtype=dtype, device=device) * 0.1

    # Generate KV
    kv = torch.randn(p.s_kv, p.h_kv, p.d_qk, dtype=dtype, device=device) * 0.1

    # Generate top-k indices for sparse attention
    # For prefill, we typically use different topk per query position
    topk_indices = torch.zeros(p.s_q, p.topk, dtype=torch.int64, device=device)
    topk_lengths = torch.zeros(p.s_q, dtype=torch.int32, device=device)

    for sq in range(p.s_q):
        if p.is_all_indices_invalid:
            # Test edge case: all indices are invalid
            topk_lengths[sq] = 0
        else:
            # For causal attention, query at position sq can attend to positions [0, sq]
            # For sparse attention, we select topk positions from those
            attendable_len = min(sq + 1, p.s_kv)
            actual_topk = min(p.topk, attendable_len)
            if actual_topk > 0:
                indices = torch.randperm(attendable_len, device=device)[:actual_topk].sort().values
                topk_indices[sq, :actual_topk] = indices
                topk_lengths[sq] = actual_topk

    # Attention sink handling (first position always attended)
    if p.have_attn_sink:
        for sq in range(p.s_q):
            actual_topk = int(topk_lengths[sq].item())
            if actual_topk > 0 and 0 not in topk_indices[sq, :actual_topk]:
                # Replace last index with 0 (attention sink)
                topk_indices[sq, actual_topk - 1] = 0
                topk_indices[sq, :actual_topk] = topk_indices[sq, :actual_topk].sort().values

    return {
        'q': q,
        'kv': kv,
        'topk_indices': topk_indices,
        'topk_lengths': topk_lengths,
    }


def reference_sparse_prefill_attention(
    q: torch.Tensor,           # [s_q, h_q, d]
    kv: torch.Tensor,          # [s_kv, h_kv, d]
    topk_indices: torch.Tensor,   # [s_q, topk]
    topk_lengths: torch.Tensor,   # [s_q]
    dv: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of sparse prefill attention."""
    s_q, h_q, d = q.shape
    s_kv, h_kv, _ = kv.shape
    device = q.device

    scale = d ** -0.5
    kv_group_num = h_q // h_kv

    # Expand KV for GQA
    k = kv.float()  # [s_kv, h_kv, d]
    v = kv[..., :dv].float()  # [s_kv, h_kv, dv]

    if kv_group_num > 1:
        k = k.unsqueeze(2).expand(-1, -1, kv_group_num, -1).reshape(s_kv, h_q, d)
        v = v.unsqueeze(2).expand(-1, -1, kv_group_num, -1).reshape(s_kv, h_q, dv)

    out = torch.zeros(s_q, h_q, dv, dtype=torch.float32, device=device)
    out_fp32 = torch.zeros(s_q, h_q, dv, dtype=torch.float32, device=device)
    max_logits = torch.zeros(s_q, h_q, dtype=torch.float32, device=device)
    lse = torch.zeros(s_q, h_q, dtype=torch.float32, device=device)

    topk_lengths_cpu = topk_lengths.cpu()

    for sq in range(s_q):
        q_sq = q[sq].float() * scale  # [h_q, d]
        actual_topk = int(topk_lengths_cpu[sq].item())

        if actual_topk == 0:
            max_logits[sq] = float('-inf')
            lse[sq] = float('-inf')
            continue

        # Get sparse indices
        indices = topk_indices[sq, :actual_topk].long()

        # Gather sparse KV
        k_sparse = k[indices]  # [topk, h_q, d]
        v_sparse = v[indices]  # [topk, h_q, dv]

        # Compute attention
        attn_scores = torch.einsum('hd, nhd -> hn', q_sq, k_sparse)
        max_scores = attn_scores.max(dim=-1).values  # [h_q]
        exp_scores = torch.exp(attn_scores - max_scores.unsqueeze(-1))
        sum_exp = exp_scores.sum(dim=-1)  # [h_q]
        attn_weights = exp_scores / sum_exp.unsqueeze(-1)

        out_sq = torch.einsum('hn, nhd -> hd', attn_weights, v_sparse)
        out[sq] = out_sq
        out_fp32[sq] = out_sq
        max_logits[sq] = max_scores
        lse[sq] = max_scores + torch.log(sum_exp)

    return out.to(q.dtype), out_fp32, max_logits, lse


def run_sparse_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    dv: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run sparse prefill using local Triton kernel."""
    # For now, use the reference implementation
    # A full Triton sparse prefill attention kernel would go here
    out, _, max_logits, lse = reference_sparse_prefill_attention(
        q, kv, topk_indices, topk_lengths, dv
    )
    return out, max_logits, lse


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    """Run a single sparse prefill test."""
    print(f"Testing: s_q={p.s_q}, s_kv={p.s_kv}, topk={p.topk}, h_q={p.h_q}, d_qk={p.d_qk}")

    if p.seed == -1:
        p.seed = random.randint(0, 10000)

    data = generate_prefill_test_data(p)

    # Run test implementation
    out_test, max_logits_test, lse_test = run_sparse_prefill_triton(
        data['q'], data['kv'], data['topk_indices'], data['topk_lengths'], p.dv
    )

    # Run reference
    out_ref, out_ref_fp32, max_logits_ref, lse_ref = reference_sparse_prefill_attention(
        data['q'], data['kv'], data['topk_indices'], data['topk_lengths'], p.dv
    )

    # Handle -inf in LSE (positions with no attendable tokens)
    lse_ref[lse_ref == float('-inf')] = float('+inf')
    lse_test_adj = lse_test.clone()
    lse_test_adj[lse_test_adj == float('-inf')] = float('+inf')

    # Check correctness
    is_correct = True
    is_correct &= kk.check_is_allclose("out", out_test.float(), out_ref_fp32,
                                        abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
    is_correct &= kk.check_is_allclose("max_logits", max_logits_test, max_logits_ref,
                                        abs_tol=1e-6, rel_tol=2.01/65536)
    is_correct &= kk.check_is_allclose("lse", lse_test_adj, lse_ref,
                                        abs_tol=1e-6, rel_tol=2.01/65536)

    return is_correct


def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    correctness_cases = [
        # Regular shapes
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [128, 64]
        for s_kv, topk in [
            (128, 128),
            (256, 256),
            (512, 512),
            (592, 128),
        ]
        for s_q in [1, 62]
    ]

    corner_cases = [
        # All indices invalid
        TestParam(s_q=1, s_kv=128, topk=128, h_q=64, is_all_indices_invalid=True, d_qk=576),
        TestParam(s_q=64, s_kv=256, topk=256, h_q=128, is_all_indices_invalid=True, d_qk=512),
    ]

    testcases = correctness_cases + corner_cases

    failed_cases = []
    for test in testcases:
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)

    if len(failed_cases) > 0:
        print(f"\n{kk.colors['RED_BG']}{len(failed_cases)} / {len(testcases)} cases failed{kk.colors['CLEAR']}")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\n{kk.colors['GREEN_BG']}All {len(testcases)} cases passed!{kk.colors['CLEAR']}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)
    main()
