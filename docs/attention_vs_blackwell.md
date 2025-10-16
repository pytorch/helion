# Attention Kernel Comparison

## Scope & Outputs

- `examples/attention.py` keeps batch/head dimensions explicit via `q_view.reshape([-1, m_dim, head_dim])`, returning only the attention tensor in the input dtype.
- `examples/blackwell_attention.py` flattens `(B·H, M)`, specializes head/value dims with `hl.specialize`, and emits both the output tensor and per-row log-sum-exp cache.

## Tiling & Scheduling

- `attention` relies on Helion’s default tiler across `(batch_tile, m_tile)` and iterates N with `torch.bmm`/`torch.baddbmm` subproblems.
- `blackwell_attention_kernel` registers explicit block sizes, uses a persistent-interleaved PID, and streams matching K/V tiles via `start_N` to align with Blackwell warp ownership.

## Math & Vectorization

- The baseline kernel computes logits with `torch.bmm`, keeps accumulators as `[tile_b, tile_m, head_dim]`, and performs alpha scaling in plain PyTorch ops.
- The Blackwell version switches to `hl.dot`, applies the scale/subtract with optional `_fma_f32x2`, and subtiles the accumulator (`hl.split` → `_mul_f32x2` → `hl.join`) to issue PTX `mul.f32x2` while respecting TMEM layout.

## Autotuning & Config

- `attention` uses the default Helion launch parameters with `static_shapes=True`.
- The Blackwell kernel enumerates tuned configs over block sizes, register budgets, tensor-descriptor indexing, and warp specialization, constraining choices through `EnumFragment` for Blackwell-friendly search space.

## Testing & Benchmark Hooks

- `attention` exposes static and dynamic variants and validates against PyTorch SDP, FlexAttention, and a hand-written reference implementation.
- The Blackwell sample keeps only the static kernel, adds a TritonBench wrapper, and times throughput with `do_bench` to showcase architecture-specific gains.
