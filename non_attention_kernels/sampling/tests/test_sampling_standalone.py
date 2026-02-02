# SPDX-License-Identifier: Apache-2.0
"""
Standalone tests for sampling kernels with PyTorch reference implementations.
"""
import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch
import torch.nn.functional as F

from temperature_vllm import apply_temperature
try:
    from min_p_vllm import apply_min_p
    HAS_MIN_P = True
except ImportError:
    HAS_MIN_P = False
try:
    from penalties_vllm import apply_penalties
    HAS_PENALTIES = True
except ImportError:
    HAS_PENALTIES = False
try:
    from gumbel_vllm import gumbel_sample
    HAS_GUMBEL = True
except ImportError:
    HAS_GUMBEL = False
try:
    from logprob_vllm import compute_token_logprobs
    HAS_LOGPROB = True
except ImportError:
    HAS_LOGPROB = False
try:
    from logit_bias_vllm import apply_logit_bias
    HAS_LOGIT_BIAS = True
except ImportError:
    HAS_LOGIT_BIAS = False


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def temperature_scale_ref(logits, temperature):
    """Reference implementation of temperature scaling."""
    return logits / temperature.unsqueeze(-1)


def min_p_filter_ref(logits, min_p):
    """Reference implementation of min-p filtering."""
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = max_probs * min_p.unsqueeze(-1)
    mask = probs < threshold
    logits_filtered = logits.clone()
    logits_filtered[mask] = float("-inf")
    return logits_filtered


def top_k_filter_ref(logits, k):
    """Reference implementation of top-k filtering."""
    batch_size = logits.shape[0]
    result = logits.clone()
    for i in range(batch_size):
        if k[i] > 0:
            topk_vals = logits[i].topk(k[i].item()).values
            threshold = topk_vals[-1]
            result[i][logits[i] < threshold] = float("-inf")
    return result


def top_p_filter_ref(logits, top_p):
    """Reference implementation of top-p (nucleus) filtering."""
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_mask = cumsum_probs - probs > top_p.unsqueeze(-1)

    # Scatter mask back
    result = logits.clone()
    batch_indices = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)
    result[batch_indices.expand_as(sorted_indices), sorted_indices] = torch.where(
        sorted_mask, torch.tensor(float("-inf"), device=logits.device), sorted_logits
    )
    return result


def repetition_penalty_ref(logits, input_ids, penalty):
    """Reference implementation of repetition penalty."""
    result = logits.clone()
    for i, (seq_logits, seq_ids) in enumerate(zip(result, input_ids)):
        for token_id in seq_ids.unique():
            if token_id >= 0:  # Skip padding
                if seq_logits[token_id] > 0:
                    seq_logits[token_id] = seq_logits[token_id] / penalty[i]
                else:
                    seq_logits[token_id] = seq_logits[token_id] * penalty[i]
    return result


def gumbel_sample_ref(logits, temperature=1.0):
    """Reference implementation of Gumbel sampling."""
    u = torch.rand_like(logits).clamp(min=1e-10, max=1.0)
    gumbel_noise = -torch.log(-torch.log(u))
    return torch.argmax(logits / temperature + gumbel_noise, dim=-1)


class TestTemperature:
    """Tests for temperature scaling kernel."""

    def test_temperature_basic(self):
        """Test basic temperature scaling."""
        batch_size = 4
        vocab_size = 1000

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
        temperature = torch.rand(batch_size, device="cuda", dtype=torch.float32) + 0.5

        # Triton kernel
        out_triton = apply_temperature(logits.clone(), temperature)

        # Reference
        out_ref = temperature_scale_ref(logits, temperature)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-4, atol=1e-4)

    def test_temperature_various_values(self):
        """Test temperature scaling with various temperature values."""
        batch_size = 8
        vocab_size = 512

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

        for temp_val in [0.1, 0.5, 1.0, 1.5, 2.0]:
            temperature = torch.full((batch_size,), temp_val, device="cuda")
            out_triton = apply_temperature(logits.clone(), temperature)
            out_ref = temperature_scale_ref(logits, temperature)
            torch.testing.assert_close(
                out_triton, out_ref, rtol=1e-4, atol=1e-4,
                msg=f"Failed for temperature {temp_val}"
            )


class TestMinP:
    """Tests for min-p filtering kernel."""

    @pytest.mark.skipif(not HAS_MIN_P, reason="min_p_vllm not available")
    def test_min_p_basic(self):
        """Test basic min-p filtering."""
        batch_size = 4
        vocab_size = 1000

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
        min_p = torch.full((batch_size,), 0.1, device="cuda", dtype=torch.float32)
        # idx_mapping maps request index to parameter index (identity for simple test)
        idx_mapping = torch.arange(batch_size, device="cuda", dtype=torch.int32)

        # Triton kernel (modifies in-place)
        out_triton = logits.clone()
        apply_min_p(out_triton, idx_mapping, min_p)

        # Reference
        out_ref = min_p_filter_ref(logits, min_p)

        # Check that filtered tokens are -inf in both
        triton_inf_mask = torch.isinf(out_triton) & (out_triton < 0)
        ref_inf_mask = torch.isinf(out_ref) & (out_ref < 0)
        torch.testing.assert_close(triton_inf_mask, ref_inf_mask)


class TestPenalties:
    """Tests for penalty kernels."""

    @pytest.mark.skipif(not HAS_PENALTIES, reason="penalties_vllm not available")
    def test_penalties_basic(self):
        """Test basic penalty application."""
        batch_size = 4
        vocab_size = 1000
        seq_len = 50

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        repetition_penalty = torch.full((batch_size,), 1.2, device="cuda")

        # Triton kernel
        out_triton = apply_penalties(logits.clone(), input_ids, repetition_penalty)

        # Reference
        out_ref = repetition_penalty_ref(logits, input_ids, repetition_penalty)

        # Check tokens that appeared in input are penalized
        for i in range(batch_size):
            unique_tokens = input_ids[i].unique()
            for token in unique_tokens:
                if token >= 0:
                    # Original and output should differ for repeated tokens
                    assert not torch.allclose(out_triton[i, token], logits[i, token])


class TestGumbel:
    """Tests for Gumbel sampling kernel."""

    @pytest.mark.skipif(not HAS_GUMBEL, reason="gumbel_vllm not available")
    def test_gumbel_basic(self):
        """Test basic Gumbel sampling."""
        batch_size = 4
        vocab_size = 1000

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
        idx_mapping = torch.arange(batch_size, device="cuda", dtype=torch.int32)
        temperature = torch.ones(batch_size, device="cuda", dtype=torch.float32)
        seed = torch.randint(0, 2**31, (batch_size,), device="cuda", dtype=torch.int64)
        pos = torch.zeros(batch_size, device="cuda", dtype=torch.int64)

        # Triton kernel
        sampled = gumbel_sample(logits, idx_mapping, temperature, seed, pos, apply_temperature=True)

        # Check output shape and validity
        assert sampled.shape == (batch_size,)
        assert sampled.dtype == torch.int64 or sampled.dtype == torch.int32
        assert (sampled >= 0).all() and (sampled < vocab_size).all()

    @pytest.mark.skipif(not HAS_GUMBEL, reason="gumbel_vllm not available")
    def test_gumbel_distribution(self):
        """Test that Gumbel sampling produces expected distribution."""
        vocab_size = 10
        num_samples = 1000  # Reduced for faster testing

        # Create a distribution where token 0 should be sampled most often
        logits = torch.zeros(1, vocab_size, device="cuda", dtype=torch.float32)
        logits[0, 0] = 5.0  # Much higher probability
        idx_mapping = torch.zeros(1, device="cuda", dtype=torch.int32)
        temperature = torch.ones(1, device="cuda", dtype=torch.float32)
        pos = torch.zeros(1, device="cuda", dtype=torch.int64)

        # Sample many times
        samples = []
        for i in range(num_samples):
            seed = torch.tensor([i], device="cuda", dtype=torch.int64)
            sample = gumbel_sample(logits.clone(), idx_mapping, temperature, seed, pos, apply_temperature=True)
            samples.append(sample.item())

        # Token 0 should be sampled most frequently
        token_0_count = sum(1 for s in samples if s == 0)
        assert token_0_count > num_samples * 0.8  # Should be > 80% of samples


class TestLogProb:
    """Tests for log probability kernels."""

    @pytest.mark.skipif(not HAS_LOGPROB, reason="logprob_vllm not available")
    def test_logprob_basic(self):
        """Test basic log probability computation."""
        batch_size = 4
        vocab_size = 1000
        num_logprobs = 5

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
        # token_ids should be 2D: [batch_size, num_logprobs]
        token_ids = torch.randint(0, vocab_size, (batch_size, num_logprobs), device="cuda")

        # Triton kernel
        logprobs = compute_token_logprobs(logits, token_ids)

        # Reference - compute log softmax and gather
        ref_log_softmax = F.log_softmax(logits, dim=-1)
        ref_gathered = ref_log_softmax.gather(1, token_ids)

        torch.testing.assert_close(logprobs, ref_gathered, rtol=1e-4, atol=1e-4)


class TestLogitBias:
    """Tests for logit bias kernel."""

    @pytest.mark.skipif(not HAS_LOGIT_BIAS, reason="logit_bias_vllm not available")
    def test_logit_bias_basic(self):
        """Test basic logit bias application."""
        batch_size = 4
        vocab_size = 1000
        num_bias_tokens = 10

        logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
        idx_mapping = torch.arange(batch_size, device="cuda", dtype=torch.int32)
        pos = torch.zeros(batch_size, device="cuda", dtype=torch.int64)

        # Create sparse bias - per request
        logit_bias_token_ids = torch.randint(0, vocab_size, (batch_size, num_bias_tokens), device="cuda", dtype=torch.int64)
        logit_bias = torch.randn(batch_size, num_bias_tokens, device="cuda", dtype=torch.float32)
        num_logit_bias = torch.full((batch_size,), num_bias_tokens, device="cuda", dtype=torch.int32)

        # No allowed token filtering or stop token filtering for this test
        num_allowed_token_ids = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
        allowed_token_ids = torch.zeros(batch_size, 1, device="cuda", dtype=torch.int64)
        min_lens = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
        num_stop_token_ids = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
        stop_token_ids = torch.zeros(batch_size, 1, device="cuda", dtype=torch.int64)

        # Apply bias
        out_triton = logits.clone()
        apply_logit_bias(
            out_triton, idx_mapping, pos,
            num_allowed_token_ids, allowed_token_ids,
            num_logit_bias, logit_bias_token_ids, logit_bias,
            min_lens, num_stop_token_ids, stop_token_ids
        )

        # Reference - manually add bias per request
        out_ref = logits.clone()
        for b in range(batch_size):
            for i in range(num_bias_tokens):
                token_id = logit_bias_token_ids[b, i].item()
                out_ref[b, token_id] += logit_bias[b, i]

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
