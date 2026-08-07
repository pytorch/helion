# Four-arm autotuning experiment: does retrieval help the LLM autotuner?

Helion can autotune with LFBO tree search, with an LLM proposing configurations,
or with both. This experiment adds a fourth option — giving the LLM retrieved
configurations from past CI autotuning runs — and measures all four head to head
under one controlled, paired campaign.

The question it answers: **is retrieval-augmented LLM autotuning worth pursuing?**
The short version of the published result is *yes, but modestly* — see
[Results](#results).

## The four arms

| Arm id | `HELION_AUTOTUNER` | Retrieval | LLM requests |
|---|---|---|---:|
| `lfbo` | `LFBOTreeSearch` | off | 0 |
| `llm` | `LLMGuidedSearch` | off | 1 |
| `hybrid_lfbo_llm` | `LLMSeededLFBOTreeSearch` | off | 1 |
| `contextual_rag_llm` | `LLMGuidedSearch` | **on** | 1 |

`llm` and `contextual_rag_llm` share identical provider, model, and round
settings, so retrieval is the *only* difference between them. That pairing is
what isolates the value of retrieval; everything else in the design exists to
keep it clean.

The arm table lives in
[`scripts/helion_rag/helion_rag/experiment/head_to_head.py`](../scripts/helion_rag/helion_rag/experiment/head_to_head.py)
(`ARM_POLICIES`), which is also the single source of truth for the frozen
controls below.

## Design

**Paired and balanced.** Every (workload, repetition) block runs all four arms
with the same seed. Execution order is a balanced Latin square
(`experiment/scheduler.py`) so drift in machine state cannot favour one arm.

**Frozen controls.** `build_study_manifest()` writes an immutable
`study_manifest.json` with a SHA-256 `config_hash` over every control: seeds,
candidate-attempt limit, effort profile, provider/model, LLM round settings,
cache policy, and the workload list. A resume refuses to mix results produced
under a different hash.

**Cold every time.** Each run unit gets its own Helion, TorchInductor, Triton,
event-log, and temp directory. Exact-cache reads, best-available reads, and
cache writes are all disabled, so no arm can inherit another's work.

**One shared budget for the hybrid.** The hybrid's LLM and LFBO stages share a
single attempt ledger and one continuous trajectory clock
(`helion/autotuner/candidate_budget.py`), so stage two cannot quietly claim a
second full allowance. `llm` and `contextual_rag_llm` typically stop early at ~26
attempts; that realized effort is reported as an outcome rather than padded out
to the ceiling.

**Two outcomes, jointly.** Selected latency (stabilized `run_example` timing)
*and* readiness wall time. Neither alone tells you anything useful — an arm can
always buy quality with time.

## Running it

### Prerequisites

- A GPU host (the published campaign used a single H100).
- The retrieval corpus and index, built by
  [`scripts/helion_rag/setup-helion-rag.sh`](../scripts/helion_rag/setup-helion-rag.sh)
  — see [`scripts/helion_rag/README.md`](../scripts/helion_rag/README.md).
- Credentials for your LLM provider.

### Environment

```bash
source .helion-rag/env.sh
export HELION_RAG_GENERATION_ID=000000   # REQUIRED — see below
export HELION_LLM_API_KEY=...            # from your own credential source
export HF_HUB_OFFLINE=1                  # keep embedding model resolution local
```

> **Pin the generation.** `env.sh` does not set `HELION_RAG_GENERATION_ID`.
> Without it every RAG arm raises `GenerationPinError`, falls back to
> `BaselineSearch`, and silently degenerates into a second copy of the `llm`
> arm — the campaign completes and the contrast you are measuring is gone. Pin
> it to the generation your index was built for.

### Launch

```bash
PYTHONPATH=scripts/helion_rag \
  nohup .venv/bin/python scripts/helion_rag/run_head_to_head_campaign.py \
  --output-dir .helion-rag/head_to_head_4arm \
  --repetitions 5 --candidate-attempt-limit 80 --timeout-seconds 1800 \
  --resume > .helion-rag/head_to_head_4arm/campaign.log 2>&1 &
```

`--workloads` selects a subset of the registry (46 shapes are registered; the
published campaign used 33). Expect ~14 h for the full 660 run units on one
H100, dominated by provider and compilation latency.

The driver is built to be interrupted. Each run's terminal outcome is written
atomically to a file keyed by (workload, arm, repetition), and `runs.jsonl` is
rebuilt from those files rather than from append order. Re-running the same
command with `--resume` skips completed units and refuses to mix in results from
a different configuration hash. A PID lock stops two campaigns from targeting
one directory.

```bash
# progress
tail -f .helion-rag/head_to_head_4arm/campaign.log
python scripts/helion_rag/run_head_to_head_campaign.py \
  --output-dir .helion-rag/head_to_head_4arm --status
```

Two failure modes worth recognizing before you debug the harness:

- `CandidatePopulationUnderfilled: initial candidate population underfilled: N/30`
  means `--candidate-attempt-limit` is below LFBO's initial population of 30. Use
  at least 40; the published campaign used 80.
- `Invalid handle. Cannot load symbol cudnnGetVersion` while the RAG arm loads the
  embedding model is an intermittent symbol-resolution race between the embedding
  model and the live Triton CUDA context, not a harness bug. It shows up as a
  failed run unit; `--resume` re-runs it and it normally succeeds.

### Analyze

```bash
PYTHONPATH=scripts/helion_rag .venv/bin/python \
  scripts/helion_rag/analyze_head_to_head.py \
  --campaign .helion-rag/head_to_head_4arm

# optional matplotlib figures (pip install -e 'scripts/helion_rag[figures]')
PYTHONPATH=scripts/helion_rag .venv/bin/python \
  scripts/helion_rag/plot_narrative_figures.py \
  --campaign .helion-rag/head_to_head_4arm
```

Under `<campaign>/analysis/`:

| File | Contents |
|---|---|
| `per_run.csv` | one row per run unit — the raw data |
| `per_kernel_arm.csv` | per (workload, arm) medians, regret, tokens |
| `per_arm_summary.csv` | suite-level geometric means per arm |
| `all_arm_table.{csv,md}` | the full per-kernel table |
| `aggregate_statistics.{json,csv}` | pairwise contrasts, CIs, Holm-adjusted p |
| `reliability.csv`, `cost.csv` | correctness and provider-token accounting |
| `trajectory_long.csv` | one row per candidate evaluation |

Figures land in `<campaign>/figures/` as SVG, PDF, and 300-DPI PNG. The gnuplot
pack needs only `gnuplot` on `PATH`; both figure scripts skip with a note rather
than failing when their renderer is missing.

### Statistics

Per kernel: the median matched-repetition ratio between two arms. Across
kernels: geometric mean, with a 200,000-resample percentile kernel bootstrap for
95% CIs and a two-sided Wilcoxon signed-rank test on log ratios. All 12
performance/readiness contrasts are Holm-corrected as one family
(`helion_rag/stats/{paired,gates}.py`). Only `status="completed"` and
`correct=true` records contribute.

Regret is reported against two baselines: the **bounded campaign oracle** (the
fastest correct latency any arm reached for that kernel — an observed reference,
not an exhaustive hardware oracle) and **LFBO**, the default autotuner.

## Results

From the published campaign — 33 workloads across 11 kernel families, 5
repetitions, 660 runs on one H100 with `claude-opus-4-8` via Vertex. Full
tables, figures, and the write-up are in
[`scripts/helion_rag/results/2026-07-26-h100/`](../scripts/helion_rag/results/2026-07-26-h100/).

| Arm | latency GM (ms) | readiness GM (s) | regret vs LFBO | tokens |
|---|---:|---:|---:|---:|
| `lfbo` | 0.0909 | 108.9 | 0% | 0 |
| `llm` | 0.0825 | **62.8** | −9.3% | 746k |
| `hybrid_lfbo_llm` | **0.0743** | 175.1 | **−18.3%** | 747k |
| `contextual_rag_llm` | 0.0808 | 71.8 | −11.1% | 875k |

There is no single winner — there is a frontier. Hybrid buys the best kernels
with 2.8x LFBO's search time. LLM is by far the fastest to be ready and still
beats LFBO on quality. RAG-LLM sits between them.

On the contrast this experiment was built to measure, **retrieval beats
context-free LLM search by 1.9% on selected latency** (ratio 0.981, 95% CI
0.964–0.994, Holm-adjusted *p*=0.034) — real, but small, and it costs 14.3% more
readiness time and 17.3% more provider tokens. All four arms produced 164/165
correct results.

Whether 1.9% justifies a retrieval corpus in production is the open question
this PR exists to let the team probe.

## Limitations

- One host, one GPU (H100), one model (`claude-opus-4-8` via Vertex), 5 seeds.
- 33 workloads is enough for the paired statistics used here, but kernel-family
  coverage is uneven and family-level effects are noisy.
- The oracle is bounded by what the campaign observed, not exhaustive search, so
  absolute regret is a lower bound.
- Retrieval quality is capped by the corpus generation the index was built from;
  a richer corpus could move the RAG arm either way.

## Where the code lives

| Path | Role |
|---|---|
| `helion/autotuner/rag/` | the retrieval adapter, policy, seeding, and instrumentation |
| `helion/autotuner/candidate_budget.py` | the shared attempt ledger the hybrid needs |
| `scripts/helion_rag/helion_rag/experiment/head_to_head.py` | arm table, frozen controls, manifest |
| `scripts/helion_rag/helion_rag/experiment/workloads/` | the benchmarkable kernel registry |
| `scripts/helion_rag/run_head_to_head_campaign.py` | resumable campaign driver |
| `scripts/helion_rag/analyze_head_to_head.py` | tables, statistics, gnuplot figures |
| `scripts/helion_rag/plot_narrative_figures.py` | optional matplotlib figures |
