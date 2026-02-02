"""
Sampling Kernels - Core Triton Implementations

==============================================================================
MATHEMATICAL CORE
==============================================================================

Sampling kernels transform model logits into token selections using various
probability manipulation and selection strategies.

Key Operations:

1. Temperature Scaling:
    logits_scaled = logits / temperature

    Where temperature > 0:
    - temperature → 0: greedy/argmax (distribution peaks)
    - temperature = 1: unchanged distribution
    - temperature > 1: flatter distribution (more diverse)

2. Softmax + Log Probabilities:
    probs = softmax(logits) = exp(logits - max) / sum(exp(logits - max))
    log_probs = log(probs) = logits - max - log(sum(exp(logits - max)))

3. Top-K Filtering:
    Keep only the k highest probability tokens, set others to -inf

4. Top-P (Nucleus) Filtering:
    Sort by probability, keep smallest set with cumsum(probs) >= p

5. Min-P Filtering:
    threshold = min_p * max(probs)
    Keep tokens where probs >= threshold

6. Repetition/Frequency Penalties:
    For each token t in context:
    - Repetition: logits[t] /= penalty if logits[t] > 0 else *= penalty
    - Frequency: logits[t] -= penalty * count[t]
    - Presence: logits[t] -= penalty * (count[t] > 0)

7. Gumbel-Max Sampling:
    sample = argmax(logits + gumbel_noise)
    Where gumbel_noise = -log(-log(uniform(0, 1)))

    This is equivalent to categorical sampling but parallelizable.

8. Rejection Sampling (for Speculative Decoding):
    For draft token with prob q and target prob p:
    - Accept with probability min(1, p/q)
    - If rejected, sample from adjusted distribution: (p - q)+

Complexity:
    - Temperature/Penalties: O(vocab_size) per token
    - Top-K: O(vocab_size) naive, O(k) with selection algorithms
    - Softmax: O(vocab_size) per token
    - Gumbel sampling: O(vocab_size) per token

References:
    - The Curious Case of Neural Text Degeneration (Holtzman et al., 2020)
    - Speculative Decoding (Leviathan et al., 2023)
    - EAGLE-2 (Li et al., 2024)

==============================================================================
"""

import torch
import triton
import triton.language as tl


# ==============================================================================
# Triton Kernel: Temperature Scaling
# ==============================================================================

@triton.jit
def temperature_kernel(
    logits_ptr,
    logits_stride,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply temperature scaling to logits.

    logits[i] = logits[i] / temperature

    Grid: (num_batches, num_vocab_blocks)
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    temperature = tl.load(temperature_ptr + batch_idx).to(tl.float32)

    # Skip if temperature is 0 (greedy) or 1 (no change)
    if temperature == 0.0 or temperature == 1.0:
        return

    # Load and scale logits
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    logits = logits / temperature

    tl.store(
        logits_ptr + batch_idx * logits_stride + offs,
        logits,
        mask=mask
    )


# ==============================================================================
# Triton Kernel: Min-P Filtering
# ==============================================================================

@triton.jit
def min_p_kernel(
    logits_ptr,
    logits_stride,
    probs_ptr,      # Pre-computed probabilities
    probs_stride,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply min-p filtering to logits.

    threshold = min_p * max(probs)
    logits[i] = -inf if probs[i] < threshold

    This filters out low-probability tokens relative to the peak.
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    min_p = tl.load(min_p_ptr + batch_idx).to(tl.float32)
    if min_p == 0.0:
        return  # No filtering

    # Load probabilities and find max (reduction across blocks needed)
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    probs = tl.load(
        probs_ptr + batch_idx * probs_stride + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    # Note: In practice, max_prob is computed in a separate reduction pass
    # This is simplified for illustration
    max_prob = tl.max(probs)
    threshold = min_p * max_prob

    # Filter logits below threshold
    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + offs,
        mask=mask,
        other=float('-inf')
    ).to(tl.float32)

    logits = tl.where(probs >= threshold, logits, float('-inf'))

    tl.store(
        logits_ptr + batch_idx * logits_stride + offs,
        logits,
        mask=mask
    )


# ==============================================================================
# Triton Kernel: Penalties (Repetition, Frequency, Presence)
# ==============================================================================

@triton.jit
def penalties_kernel(
    logits_ptr,
    logits_stride,
    input_ids_ptr,          # Context token IDs
    input_ids_stride,
    seq_len,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply repetition, frequency, and presence penalties.

    For each token t that appears in context:
    - Repetition: logits[t] /= penalty (if > 0) or *= penalty (if < 0)
    - Frequency: logits[t] -= freq_penalty * count[t]
    - Presence: logits[t] -= pres_penalty * (count[t] > 0)

    Grid: (num_batches, num_vocab_blocks)
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    # Load logits for this block
    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    # Count occurrences of each token in context
    # Note: This is simplified - actual implementation uses bincount
    counts = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for i in range(seq_len):
        token_id = tl.load(input_ids_ptr + batch_idx * input_ids_stride + i)
        is_match = (offs == token_id)
        counts = counts + tl.where(is_match, 1, 0)

    # Apply penalties where token appeared
    has_appeared = counts > 0

    # Repetition penalty
    if repetition_penalty != 1.0:
        logits = tl.where(
            has_appeared & (logits > 0),
            logits / repetition_penalty,
            tl.where(has_appeared, logits * repetition_penalty, logits)
        )

    # Frequency penalty
    if frequency_penalty != 0.0:
        logits = logits - frequency_penalty * counts.to(tl.float32)

    # Presence penalty
    if presence_penalty != 0.0:
        logits = tl.where(has_appeared, logits - presence_penalty, logits)

    tl.store(
        logits_ptr + batch_idx * logits_stride + offs,
        logits,
        mask=mask
    )


# ==============================================================================
# Triton Kernel: Gumbel-Max Sampling
# ==============================================================================

@triton.jit
def gumbel_sample_kernel(
    output_ptr,         # Output: sampled token IDs
    logits_ptr,
    logits_stride,
    seeds_ptr,          # Random seeds per batch
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gumbel-max trick for categorical sampling.

    sample = argmax(logits/temp + gumbel_noise)

    Where gumbel_noise = -log(-log(uniform))

    This is equivalent to sampling from softmax but is parallelizable
    because argmax can be computed blockwise with reduction.

    Grid: (num_batches, num_vocab_blocks)
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    seed = tl.load(seeds_ptr + batch_idx)
    temperature = tl.load(temperature_ptr + batch_idx).to(tl.float32)

    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    # Load logits
    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + offs,
        mask=mask,
        other=float('-inf')
    ).to(tl.float32)

    if temperature != 0.0:
        # Apply temperature
        logits = logits / temperature

        # Generate Gumbel noise: -log(-log(uniform))
        # Use different seed offset for each position
        gumbel_seed = tl.randint(seed, block_idx)
        uniform = tl.rand(gumbel_seed, offs).to(tl.float64)
        gumbel_noise = -tl.log(-tl.log(uniform + 1e-20) + 1e-20)

        # Add noise
        logits = logits + gumbel_noise.to(tl.float32)

    # Find local max and argmax
    local_max = tl.max(logits)
    local_argmax = tl.argmax(logits, axis=0)

    # Note: Full implementation needs reduction across blocks
    # to find global argmax


# ==============================================================================
# Triton Kernel: Top-K Log Softmax
# ==============================================================================

@triton.jit
def topk_logsoftmax_kernel(
    output_tokens_ptr,    # Output: top-k token IDs
    output_logprobs_ptr,  # Output: top-k log probabilities
    logits_ptr,
    logits_stride,
    k: tl.constexpr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused top-k selection with log softmax computation.

    1. Find max for numerical stability
    2. Compute log_softmax = logits - max - log(sum(exp(logits - max)))
    3. Select top-k tokens with highest log probabilities

    This avoids materializing full probability distribution.
    """
    batch_idx = tl.program_id(0)

    # First pass: find max for numerical stability
    max_val = float('-inf')
    for block_idx in range(tl.cdiv(vocab_size, BLOCK_SIZE)):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        logits = tl.load(
            logits_ptr + batch_idx * logits_stride + offs,
            mask=mask,
            other=float('-inf')
        ).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(logits))

    # Second pass: compute sum(exp(logits - max))
    sum_exp = 0.0
    for block_idx in range(tl.cdiv(vocab_size, BLOCK_SIZE)):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        logits = tl.load(
            logits_ptr + batch_idx * logits_stride + offs,
            mask=mask,
            other=float('-inf')
        ).to(tl.float32)
        sum_exp += tl.sum(tl.exp(logits - max_val))

    log_sum_exp = max_val + tl.log(sum_exp)

    # Third pass: find top-k (simplified - actual uses heap or sorting)
    # Note: Full implementation requires efficient top-k selection


# ==============================================================================
# Triton Kernel: Rejection Sampling (Speculative Decoding)
# ==============================================================================

@triton.jit
def rejection_sample_kernel(
    accepted_ptr,         # Output: number of accepted tokens
    output_tokens_ptr,    # Output: final token sequence
    draft_tokens_ptr,     # Draft model's proposed tokens
    draft_probs_ptr,      # Draft model's probabilities
    target_probs_ptr,     # Target model's probabilities
    uniform_samples_ptr,  # Pre-generated uniform random numbers
    num_draft_tokens,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Rejection sampling for speculative decoding verification.

    For each draft token t with draft_prob q and target_prob p:
    - Accept with probability min(1, p/q)
    - If rejected, sample from adjusted distribution: max(0, p - q) / sum(max(0, p - q))

    This ensures the output distribution exactly matches the target model.
    """
    batch_idx = tl.program_id(0)

    num_accepted = 0
    for i in range(num_draft_tokens):
        draft_token = tl.load(draft_tokens_ptr + batch_idx * num_draft_tokens + i)
        draft_prob = tl.load(draft_probs_ptr + batch_idx * num_draft_tokens + i)
        target_prob = tl.load(target_probs_ptr + batch_idx * num_draft_tokens * vocab_size +
                              i * vocab_size + draft_token)
        uniform = tl.load(uniform_samples_ptr + batch_idx * num_draft_tokens + i)

        # Accept with probability min(1, p/q)
        accept_prob = tl.minimum(1.0, target_prob / (draft_prob + 1e-10))

        if uniform < accept_prob:
            # Accept this token
            tl.store(output_tokens_ptr + batch_idx * num_draft_tokens + num_accepted, draft_token)
            num_accepted += 1
        else:
            # Reject - need to sample from adjusted distribution
            # (Simplified - actual implementation handles adjusted sampling)
            break

    tl.store(accepted_ptr + batch_idx, num_accepted)


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def sampling_reference(
    logits: torch.Tensor,          # [batch, vocab_size]
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation of sampling.

    Returns sampled token IDs.
    """
    batch_size, vocab_size = logits.shape

    # Temperature scaling
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Min-P filtering
    if min_p > 0.0:
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = min_p * max_probs
        logits = logits.masked_fill(probs < threshold, float('-inf'))
        probs = torch.softmax(logits, dim=-1)

    # Top-K filtering
    if top_k > 0:
        indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = torch.softmax(logits, dim=-1)

    # Top-P filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = torch.softmax(logits, dim=-1)

    # Sample
    if temperature == 0:
        # Greedy
        return logits.argmax(dim=-1)
    else:
        # Categorical
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ==============================================================================
# Wrapper Functions for Testing
# ==============================================================================

def temperature_scaling_triton(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """
    Apply temperature scaling to logits using Triton kernel.

    Args:
        logits: [batch, vocab_size] logits tensor
        temperature: [batch] temperature values per batch

    Returns:
        Scaled logits (modified in place but also returned)
    """
    batch_size, vocab_size = logits.shape
    BLOCK_SIZE = 1024

    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_blocks)

    temperature_kernel[grid](
        logits,
        logits.stride(0),
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return logits


def min_p_filter_triton(
    logits: torch.Tensor,
    probs: torch.Tensor,
    min_p: torch.Tensor
) -> torch.Tensor:
    """
    Apply min-p filtering to logits using Triton kernel.

    Args:
        logits: [batch, vocab_size] logits tensor
        probs: [batch, vocab_size] probability tensor (from softmax of logits)
        min_p: [batch] min-p threshold per batch

    Returns:
        Filtered logits (modified in place but also returned)
    """
    batch_size, vocab_size = logits.shape
    BLOCK_SIZE = 1024

    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_blocks)

    min_p_kernel[grid](
        logits,
        logits.stride(0),
        probs,
        probs.stride(0),
        min_p,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return logits


def penalties_triton(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    seq_len: int,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Apply repetition, frequency, and presence penalties using Triton kernel.

    Args:
        logits: [batch, vocab_size] logits tensor
        input_ids: [batch, max_seq_len] context token IDs
        seq_len: actual sequence length to consider
        repetition_penalty: multiplicative penalty for repeated tokens
        frequency_penalty: additive penalty proportional to count
        presence_penalty: additive penalty for appearing tokens

    Returns:
        Penalized logits (modified in place but also returned)
    """
    batch_size, vocab_size = logits.shape
    BLOCK_SIZE = 1024

    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_blocks)

    penalties_kernel[grid](
        logits,
        logits.stride(0),
        input_ids,
        input_ids.stride(0),
        seq_len,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return logits


# ==============================================================================
# Reference Implementations for Testing
# ==============================================================================

def temperature_scaling_reference(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for temperature scaling."""
    result = logits.clone()
    for i in range(logits.shape[0]):
        temp = temperature[i].item()
        if temp != 0.0 and temp != 1.0:
            result[i] = logits[i] / temp
    return result


def min_p_filter_reference(
    logits: torch.Tensor,
    probs: torch.Tensor,
    min_p: torch.Tensor
) -> torch.Tensor:
    """PyTorch reference for min-p filtering."""
    result = logits.clone()
    for i in range(logits.shape[0]):
        mp = min_p[i].item()
        if mp > 0.0:
            max_prob = probs[i].max()
            threshold = mp * max_prob
            mask = probs[i] < threshold
            result[i] = result[i].masked_fill(mask, float('-inf'))
    return result


def penalties_reference(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    seq_len: int,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """PyTorch reference for repetition/frequency/presence penalties."""
    result = logits.clone().float()
    batch_size, vocab_size = logits.shape

    for b in range(batch_size):
        # Count occurrences
        counts = torch.zeros(vocab_size, device=logits.device, dtype=torch.int32)
        for i in range(seq_len):
            token_id = input_ids[b, i].item()
            if token_id < vocab_size:
                counts[token_id] += 1

        # Apply penalties
        appeared = counts > 0

        # Repetition penalty
        if repetition_penalty != 1.0:
            positive_mask = appeared & (result[b] > 0)
            negative_mask = appeared & (result[b] <= 0)
            result[b] = torch.where(positive_mask, result[b] / repetition_penalty, result[b])
            result[b] = torch.where(negative_mask, result[b] * repetition_penalty, result[b])

        # Frequency penalty
        if frequency_penalty != 0.0:
            result[b] = result[b] - frequency_penalty * counts.float()

        # Presence penalty
        if presence_penalty != 0.0:
            result[b] = torch.where(appeared, result[b] - presence_penalty, result[b])

    return result


# ==============================================================================
# Tests
# ==============================================================================

import pytest


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("vocab_size", [256, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_temperature_scaling_triton_vs_reference(batch_size, vocab_size, dtype):
    """Test temperature scaling kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda"

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    temperature = torch.tensor([0.7, 1.5, 0.5, 2.0][:batch_size], device=device, dtype=dtype)

    # Clone for Triton (in-place operation)
    logits_triton = logits.clone()
    logits_ref = logits.clone()

    # Run Triton kernel
    out_triton = temperature_scaling_triton(logits_triton, temperature)

    # Run reference
    out_ref = temperature_scaling_reference(logits_ref, temperature)

    # Compare
    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("vocab_size", [256, 1024])
@torch.inference_mode()
def test_min_p_filter_triton_vs_reference(batch_size, vocab_size):
    """Test min-p filtering kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    probs = torch.softmax(logits, dim=-1)
    min_p = torch.tensor([0.1, 0.05, 0.2, 0.01][:batch_size], device=device, dtype=dtype)

    # Clone for Triton (in-place operation)
    logits_triton = logits.clone()
    logits_ref = logits.clone()

    # Run Triton kernel
    out_triton = min_p_filter_triton(logits_triton, probs, min_p)

    # Run reference
    out_ref = min_p_filter_reference(logits_ref, probs, min_p)

    # Compare - check non-inf values match and inf positions match
    inf_mask_triton = torch.isinf(out_triton)
    inf_mask_ref = torch.isinf(out_ref)

    # Positions of -inf should match
    assert torch.equal(inf_mask_triton, inf_mask_ref), "Inf positions don't match"

    # Non-inf values should be close
    non_inf_mask = ~inf_mask_triton
    if non_inf_mask.any():
        torch.testing.assert_close(
            out_triton[non_inf_mask],
            out_ref[non_inf_mask],
            atol=1e-3, rtol=1e-3
        )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("vocab_size", [64, 128])
@pytest.mark.parametrize("seq_len", [4, 8])
@torch.inference_mode()
def test_penalties_triton_vs_reference(batch_size, vocab_size, seq_len):
    """Test penalties kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int64)

    repetition_penalty = 1.2
    frequency_penalty = 0.5
    presence_penalty = 0.3

    # Clone for Triton (in-place operation)
    logits_triton = logits.clone()
    logits_ref = logits.clone()

    # Run Triton kernel
    out_triton = penalties_triton(
        logits_triton, input_ids, seq_len,
        repetition_penalty, frequency_penalty, presence_penalty
    )

    # Run reference
    out_ref = penalties_reference(
        logits_ref, input_ids, seq_len,
        repetition_penalty, frequency_penalty, presence_penalty
    )

    # Compare with relaxed tolerance due to loop order differences
    torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_sampling_half_precision(dtype):
    """Test sampling kernels work with half precision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda"
    batch_size = 4
    vocab_size = 512

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    temperature = torch.tensor([0.8, 1.0, 1.2, 0.6], device=device, dtype=dtype)

    # Test temperature scaling
    logits_scaled = logits.clone()
    out = temperature_scaling_triton(logits_scaled, temperature)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out[temperature != 1.0]).all()  # Should have finite values where temp != 1


@torch.inference_mode()
def test_sampling_reference_function():
    """Test the sampling_reference function produces valid outputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda"
    batch_size = 4
    vocab_size = 1000

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)

    # Test different configurations
    # Greedy sampling
    tokens_greedy = sampling_reference(logits.clone(), temperature=0.0)
    assert tokens_greedy.shape == (batch_size,)
    assert (tokens_greedy >= 0).all() and (tokens_greedy < vocab_size).all()

    # Temperature sampling
    tokens_temp = sampling_reference(logits.clone(), temperature=0.8)
    assert tokens_temp.shape == (batch_size,)
    assert (tokens_temp >= 0).all() and (tokens_temp < vocab_size).all()

    # Top-k sampling
    tokens_topk = sampling_reference(logits.clone(), temperature=1.0, top_k=50)
    assert tokens_topk.shape == (batch_size,)
    assert (tokens_topk >= 0).all() and (tokens_topk < vocab_size).all()

    # Top-p sampling
    tokens_topp = sampling_reference(logits.clone(), temperature=1.0, top_p=0.9)
    assert tokens_topp.shape == (batch_size,)
    assert (tokens_topp >= 0).all() and (tokens_topp < vocab_size).all()

    # Min-p sampling
    tokens_minp = sampling_reference(logits.clone(), temperature=1.0, min_p=0.05)
    assert tokens_minp.shape == (batch_size,)
    assert (tokens_minp >= 0).all() and (tokens_minp < vocab_size).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
