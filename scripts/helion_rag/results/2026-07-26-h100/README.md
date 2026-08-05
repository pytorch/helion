# Published campaign: 2026-07-26, single H100

Reference results for the four-arm head-to-head experiment, checked in so a
replication can be diffed against ours rather than judged in isolation. See
[`docs/rag_autotuning_experiment.md`](../../../../docs/rag_autotuning_experiment.md)
for the design and how to re-run it.

**Setup:** 33 workloads across 11 kernel families x 4 arms x 5 repetitions = 660
runs, one H100, `claude-opus-4-8` via Vertex, `quick` effort, 80-attempt ceiling,
retrieval generation `000000`. Every control is frozen in `study_manifest.json`
under `config_hash = 1acdab90...`; the driver refuses to resume a campaign whose
hash differs.

## Headline

| Arm | latency GM (ms) | readiness GM (s) | regret vs LFBO | correct | tokens |
|---|---:|---:|---:|---:|---:|
| `lfbo` | 0.0909 | 108.9 | 0% | 164/165 | 0 |
| `llm` | 0.0825 | **62.8** | −9.3% | 164/165 | 746k |
| `hybrid_lfbo_llm` | **0.0743** | 175.1 | **−18.3%** | 164/165 | 747k |
| `contextual_rag_llm` | 0.0808 | 71.8 | −11.1% | 164/165 | 875k |

Retrieval vs context-free LLM — the contrast the experiment exists to measure —
is **1.9% lower selected latency** (ratio 0.981, 95% CI 0.964–0.994,
Holm-adjusted *p*=0.034) for 14.3% more readiness time and 17.3% more tokens.

`REPORT.md` is the full write-up: hypotheses, per-family breakdown, robustness,
seed repeatability, and limitations.

## Files

| Path | Contents |
|---|---|
| `study_manifest.json` | every frozen control plus the config hash |
| `REPORT.md` | the full write-up with all tables |
| `analysis/per_run.csv` | one row per run unit — the rawest data kept here |
| `analysis/per_kernel_arm.csv` | per (workload, arm) medians, regret, tokens |
| `analysis/per_arm_summary.csv` | suite-level geometric means |
| `analysis/all_arm_table.md` | the full per-kernel table |
| `analysis/aggregate_statistics.{json,csv}` | pairwise contrasts, CIs, adjusted p |
| `analysis/reliability.csv`, `analysis/cost.csv` | correctness and token accounting |
| `figures/*.svg` | five representative figures |

`trajectory_long.csv` (29,052 rows, 4.2 MB) and the PDF/PNG figure renderings are
omitted to keep the repository small. Both regenerate from a campaign directory:

```bash
PYTHONPATH=scripts/helion_rag .venv/bin/python \
  scripts/helion_rag/analyze_head_to_head.py --campaign <campaign-dir>
```

The raw 660 event logs and per-run result files are not checked in either; ask if
you need them for a deeper replication.

## Comparing a replication

Different hardware, provider, or corpus generation will move the absolute
numbers. The claims worth checking are the *relative* ones:

1. Does `hybrid_lfbo_llm` still win on latency and lose badly on readiness time?
2. Is `llm` still the fastest to be ready while beating `lfbo` on latency?
3. Does `contextual_rag_llm` still beat `llm` on latency at all — and by how
   much? A result near or below zero here is the finding that matters most.

`analysis/aggregate_statistics.json` holds all 12 Holm-corrected contrasts, so a
replication can be compared contrast by contrast rather than headline to
headline.
