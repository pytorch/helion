# LLM-Guided Autotuner for Helion — MVP

## Motivation

Helion's default autotuner (`"full"` effort) evaluates **hundreds of kernel configurations** using random search, pattern search, and differential evolution. For a simple `add` kernel, this means ~676 configs over ~3.5 minutes. For complex kernels like matmul, it can be significantly longer.

This experiment tests whether an **LLM can pre-rank candidate configurations** so the autotuner only benchmarks the top-K most promising ones, dramatically reducing search time.

## Approach

```
Helion Kernel
      |
Generate N random candidate configs (Helion's ConfigGeneration API)
      |
Extract structured kernel features (operation type, shape, memory pattern, etc.)
      |
LLM ranks configs from best to worst
      |
Benchmark only the top K configs
      |
Return the best
```

The LLM receives:
- **Kernel features**: structured metadata (kernel type, tensor shapes, memory access pattern, whether it's compute/memory bound, etc.)
- **Candidate configs**: the key tuning parameters (`block_sizes`, `num_warps`, `num_stages`)

It returns a ranked ordering. We benchmark only the top-K.

## Files

| File | Description |
|------|-------------|
| `autotuner_mvp.py` | Main benchmark harness. Imports kernels directly from `examples/`, runs three strategies (default autotuner, random baseline, LLM-guided), logs results. |
| `llm_ranker.py` | LLM ranking module. Supports OpenAI-compatible APIs (including custom endpoints) and Anthropic. Includes expert system prompt with GPU optimization heuristics. |
| `analyze_results.py` | Post-hoc analysis of saved results. Per-run breakdowns, aggregate statistics, config pattern analysis, LLM ranking effectiveness. |

## Supported Kernels

All imported directly from `examples/` — no duplicate kernel code:

| Kernel | Type | Config Complexity |
|--------|------|-------------------|
| `add` | Elementwise | Low (2D tile sizes) |
| `softmax` | Row-wise reduction | Medium (batch tile + reduction loop) |
| `matmul` | Tiled GEMM | High (3D tiles, loop order, L2 grouping) |
| `layer_norm` | Fused normalization | High (batch tile + nested reductions) |

## Results

Tested with `gpt-5-mini-2025-08-07` via OpenAI API. 15 candidate configs, top-5 benchmarked. All runs skip the default autotuner for speed.

### Per-Kernel Results

#### `add` (1024x1024, float32) — Elementwise
| Strategy | Best Latency | Configs Evaluated |
|----------|-------------|-------------------|
| Random baseline | 0.008 ms | 5 |
| LLM-guided | 0.008 ms | 5 |

**Verdict**: Tied. The `add` kernel is too simple — the default config is nearly optimal, so there's little for the LLM to differentiate.

#### `matmul` (1024x1024x1024, float16) — Tiled GEMM
| Strategy | Best Latency | Configs Evaluated |
|----------|-------------|-------------------|
| Random baseline | 0.233 ms | 3 (2 failed) |
| LLM-guided | **0.039 ms** | 5 |

**Verdict**: LLM is **6x better**. It correctly ranked `block_sizes=[128, 32, 32]` as #1, which was indeed the fastest config. The LLM's understanding of tiled matmul heuristics (balanced tile dimensions, appropriate warp count) pays off here.

#### `softmax` (4096x2560, float16) — Row-wise Reduction
| Strategy | Best Latency | Configs Evaluated |
|----------|-------------|-------------------|
| Random baseline | **0.023 ms** | 5 |
| LLM-guided | 0.268 ms | 4 (1 failed) |

**Verdict**: LLM is **12x worse**. It preferred large `block_sizes` (128, 256, 512), but for this kernel `block_sizes` controls batch rows — not the reduction tile. The "bigger blocks = better" heuristic from elementwise/matmul doesn't transfer to reduction kernels where `reduction_loops` is the critical parameter.

### Key Findings

1. **LLM ranking works well for matmul** where standard GPU heuristics (large balanced tiles, moderate warps) directly map to performance.

2. **LLM ranking fails for softmax** where the relationship between config parameters and performance is non-obvious (e.g., `block_sizes` means batch rows, not tile size).

3. **The LLM needs kernel-specific context** — generic GPU optimization heuristics are not enough. The prompt should explain what each config parameter controls for the specific kernel type.

4. **Config generation matters** — many random configs fail to compile or produce terrible performance. The LLM could potentially also filter out obviously bad configs.

### Aggregate Statistics (4 runs on `add` kernel)
| Strategy | Avg Latency | Std Dev |
|----------|------------|---------|
| Random baseline | 0.0092 ms | 0.0010 |
| LLM-guided | 0.0082 ms | 0.0000 |

The LLM consistently picks the same optimal config for `add`, showing zero variance.

## Usage

```bash
# Run a specific kernel:
python autotuner_mvp.py --kernel matmul --candidates 15 --top-k 5

# Run all kernels:
python autotuner_mvp.py --kernel all --skip-default

# Dry-run (no LLM API call, random shuffle instead):
python autotuner_mvp.py --kernel softmax --dry-run

# Analyze saved results:
python analyze_results.py --log-dir logs/
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | Custom API endpoint (default: `https://api.openai.com/v1`) |
| `AUTOTUNER_LLM_MODEL` | Model name (default: `gpt-5-mini-2025-08-07`) |
| `AUTOTUNER_LLM_BACKEND` | `openai` (default) or `anthropic` |

## Next Steps

1. **Kernel-aware prompting**: Include what each config parameter means for the specific kernel type (e.g., "block_sizes controls batch rows for softmax, not tile size").

2. **Iterative refinement**: Let the LLM see benchmark results from round 1 and suggest refined configs for round 2 (agentic loop).

3. **Helion IR features**: Extract features from the compiled IR rather than hand-written metadata, enabling automatic feature extraction for any kernel.

4. **Integration with search algorithms**: Use LLM ranking as the initial population for pattern search or differential evolution, rather than random initialization.

5. **Few-shot examples**: Include examples of good configs for similar kernels to improve ranking quality.
