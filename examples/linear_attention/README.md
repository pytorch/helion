# Generic Chunkwise Linear Attention Primitive for Helion

## Motivation

Modern efficient sequence models -- Gated DeltaNet (Qwen3.5), GLA, DeltaNet, RetNet,
Mamba2, RWKV -- all share a common computational pattern: chunkwise linear recurrence.

Today, each model requires a hand-written Triton kernel (typically 200-500 lines).
This contribution provides a generic Helion primitive that captures the shared
chunkwise recurrence pattern, parameterized by model-specific update rules.

## The Common Pattern

All chunkwise linear attention models follow this structure:

    For each chunk c = 0, 1, ..., NT-1:
        1. STORE:  h[c] = S
        2. INTRA:  compute within-chunk interactions (model-specific)
        3. DECAY:  S = S * decay_factor (model-specific)
        4. UPDATE: S = S + update(k, v, g, c) (model-specific)

## Files

- chunk_recurrence.py: The generic chunkwise recurrence primitive
- gated_delta_net.py: Gated DeltaNet using the primitive
- retnet.py: RetNet using the primitive
- delta_net.py: DeltaNet (ungated) using the primitive
- benchmark_comparison.py: Side-by-side performance comparison

## References

- Yang et al. Gated Delta Networks (ICLR 2025)
- Yang et al. Parallelizing Linear Transformers with the Delta Rule (NeurIPS 2024)
- Sun et al. Retentive Network (2023)
