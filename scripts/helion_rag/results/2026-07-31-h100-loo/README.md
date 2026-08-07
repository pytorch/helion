# Published run: 2026-07-31, single H100

Reference results for the leave-one-workload-out study of RAG-warm-started LFBO
autotuning, checked in so a replication can be diffed against ours rather than
judged in isolation. See
[`docs/rag_lfbo_warm_start_experiment.md`](../../../../docs/rag_lfbo_warm_start_experiment.md)
for the design and how to re-run it.

**Setup:** 15 kernels x 3 held-out shapes x 3 repetitions x 2 arms = 270 cells
(135 matched pairs), one H100, `full` effort, `Qwen/Qwen3-Embedding-8B` with
`embed_text=minimalist`, `sim_threshold=0.75`, `k=3` neighbours, `top_n=3`
configs each, shape reranking on. Run `lfbo-confirm-full-20260724T155452Z`, code
revision `AB-lfbo-full-r3`, ~2 days wall clock.

Both arms run the identical `LFBOTreeSearch`. The only difference is whether the
search starts from retrieved neighbour configs, so the measured effect is
attributable to the seeds alone.

## Headline

| Metric | Ratio (rag_lfbo / lfbo) | 95% CI | Reading |
|---|---:|---|---|
| Kernel latency | **0.9923** | [0.9635, 1.0131] | non-inferior at the 1.02 margin; not superior |
| Autotune time | **0.8948** | [0.8094, 1.0009] | ~10.5% faster; CI upper misses < 1.0 by 0.0009 |
| End-to-end time | 0.9066 | [0.8208, 1.0105] | ~9% faster |
| Completion | 97.8% | McNemar *p* = 0.750 | gate passes |

**Verdict `non_inferior_only`.** Warm starting matches plain LFBO on kernel
quality and converges with a median **564 configs explored vs 780** (~28%
fewer), but the preregistered cluster bootstrap leaves the search-time CI upper
bound at 1.0009 — a hair above the < 1.0 needed to claim superiority. Leakage
was zero and Tier-1 coverage 100%.

`REPORT.md` is the full write-up: design, controls, per-kernel breakdown,
confirmatory tests, and limitations.

## Files

| Path | Contents |
|---|---|
| `REPORT.md` | the full write-up with all tables |
| `lfbo_confirm_report.{md,json}` | machine-generated verdict |
| `margins.json` | preregistered non-inferiority margins |
| `perf_configs_runtime_raw.csv` | one row per eval cell — the rawest data kept here |
| `perf_configs_runtime_by_workload.md` | per-workload configs and runtime |
| `perf_autotune_geomean_by_kernel.md` | per-kernel geometric means |
| `figures/*.svg` | per-kernel latency and search-time charts |

The campaign directory's JSONL files (269 result rows, oracles, preflight, the
planned matrix) are not checked in. `lfbo_confirm_report.json` regenerates from
`workload_results.jsonl` alone:

```bash
PYTHONPATH=scripts/helion_rag .rag-venv/bin/python -m helion_rag.loo_report \
  --results <run-dir>/workload_results.jsonl \
  --oracles <run-dir>/workload_oracles.jsonl \
  --margins-json scripts/helion_rag/results/2026-07-31-h100-loo/margins.json \
  --out /tmp/report.md
```

## Comparing a replication

Different hardware, corpus generation, or embedding model will move the absolute
numbers. The claims worth checking are the relative ones:

1. Does warm starting still explore materially fewer configs than a cold search?
2. Does kernel latency still land inside the non-inferiority margin — that is, do
   the seeds cost nothing in quality?
3. Does the search-time CI clear 1.0 on a larger sample? That is the single
   result this run could not establish.
