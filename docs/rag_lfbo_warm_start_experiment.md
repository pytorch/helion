# Warm-starting the LFBO autotuner from retrieved configs

Helion's `LFBOTreeSearch` explores the configuration space from a cold start for
every new workload, even when a very similar shape was tuned before. This
experiment retrieves the measured-best configs of *similar* previously-tuned
workloads and hands them to the search as seeds, then measures whether that
makes autotuning cheaper without making the resulting kernels slower.

The question it answers: **does warm-starting the surrogate search buy a cheaper
search at no cost in kernel quality?** The published result is *quality is
unharmed and the search is ~10% cheaper, but the speedup narrowly missed
statistical superiority* — see [Results](#results).

This is a different question from the four-arm campaign in
[`rag_autotuning_experiment.md`](rag_autotuning_experiment.md), which asks
whether retrieval helps the **LLM** autotuner. Nothing here involves an LLM;
both arms make zero provider calls.

## The two arms

| Arm id | `HELION_AUTOTUNER` | Retrieval seeds | LLM requests |
|---|---|---|---:|
| `lfbo` | `LFBOTreeSearch` | off | 0 |
| `rag_lfbo` | `LFBOTreeSearch` | **on** | 0 |

The arms are byte-identical except for `HELION_RAG_LOO_SEEDING`, so the seeds are
the only treatment. `arm_environment()` in
[`scripts/helion_rag/helion_rag/loo_experiment.py`](../scripts/helion_rag/helion_rag/loo_experiment.py)
is the single source of truth, and a test pins that the two environments differ
in exactly that one variable.

> **Two retrieval paths exist.** This harness drives its own seeding path,
> `helion_rag.patch`, under `HELION_RAG_LOO_SEEDING=1`. It predates the in-tree
> adapter at `helion/autotuner/rag/`, which is gated on `HELION_RAG_ENABLED=1`.
> They must not both be enabled — each would retrieve and seed independently —
> and `patch.install()` asserts that. This experiment keeps its own path because
> that is the treatment the published numbers came from.

## Design

**Leave-one-workload-out.** The threat to validity is leakage: if the exact
target workload is retrievable, retrieval "wins" by replaying the stored answer,
which never happens in deployment. So for each held-out workload the pipeline
builds a dedicated fold whose FAISS index and exact map exclude that workload —
*semantically*, dropping every record with the same (kernel, shapes, dtypes),
because CI snapshots can store the same workload under several keys. Other
shapes of the same kernel stay, which is the realistic condition: a new shape of
a kernel the corpus already knows.

**Three phases, each resumable.**

1. `preflight` — run the **baseline** on a pool of candidate shapes per kernel to
   establish which workloads the autotuner can tune at all, plus a top-5
   rebenchmark per workload as a drift-resistant oracle. Treatment comparison is
   never distorted by workloads that fail for reasons unrelated to retrieval.
2. `select` — freeze up to 3 size-spanning shapes per kernel from the eligible
   set and build one fold each. A kernel enters the analysis only with ≥2
   eligible selected workloads.
3. `eval` — run both arms over the selected workloads at the configured
   repetitions.

**Counterbalanced.** The two arms of each (kernel, workload, rep) run adjacently,
with a seeded coin flip choosing which goes first, so thermal state and machine
drift cannot systematically favour one arm. Each cell records its global
`run_index`.

**Cold and isolated.** Every cell is a fresh subprocess under a hard timeout;
`HELION_SKIP_CACHE=1` stops any cross-cell config reuse. A timeout writes an
explicit `status: "timeout"` row, so the results manifest never has a silent gap.

**Resumable.** Each cell's resume key is a SHA-256 over its full spec, including
the phase, corpus fingerprint, code revision, and retrieval settings. Only cells
without a successful result re-run. Pin `--code-revision` when the working tree
is churned by unrelated activity, or the dirty-tree fingerprint drifts and forces
a full redo.

**Asymmetric criterion.** Warm starting is a *cost* lever, so the claim it must
earn is a faster search; on kernel quality it only has to prove it does no harm.
The verdict is `effective` only when the completion gate passes, kernel latency
is non-inferior at the preregistered margin, **and** the autotune-time CI upper
bound is below 1.0; `non_inferior_only` when the first two hold but not the
third.

**Statistics** (`loo_report.analyze_workload_results`): per-rep log ratios are
averaged within each (kernel, workload) *before* resampling, so a workload with
more surviving reps does not gain weight; a cluster bootstrap resamples kernels,
then workloads within each kernel. The completion gate is a one-sided exact
McNemar test on the discordance table. Perf is re-estimated with candidate-only
failures imputed at 1.25 / 1.5 / 2.0 to show robustness to the missing cells.

## Running it

### Prerequisites

A built retrieval corpus and index (see
[`scripts/helion_rag/README.md`](../scripts/helion_rag/README.md)), a GPU, and
the `helion_rag` package importable. Fold construction re-embeds the corpus once
per held-out workload, which dominates setup time.

```bash
export PYTHONPATH=scripts/helion_rag
RUN=.helion-rag/loo_evaluation/my-run

for phase in preflight select eval; do
  .rag-venv/bin/python -m helion_rag.loo_experiment \
    --phase $phase --env-file .helion-rag/env.sh --family h100 \
    --output-dir $RUN --repetitions 3 --timeout-s 5400 \
    --code-revision my-run-r1
done
```

Re-running any phase resumes it. `--dry-run` on `preflight` or `eval` writes the
planned matrix and stops, which is the cheap way to check the cell count before
committing GPU time. `--kernels` restricts the *targets* only — retrieval always
draws on the full corpus.

### Reading the output

```bash
.rag-venv/bin/python -m helion_rag.loo_report \
  --results $RUN/workload_results.jsonl \
  --oracles $RUN/workload_oracles.jsonl \
  --margins-json scripts/helion_rag/results/2026-07-31-h100-loo/margins.json \
  --out $RUN/report.md
```

Check the diagnostics block first. `heldout_shape_leakage_count` and any Tier-0
RAG cell must be **0** — a nonzero value means the fold did not actually hold the
workload out and the treatment estimate is contaminated. `tier1_coverage` tells
you what fraction of candidate cells were actually seeded; a low value means the
study mostly measured the baseline against itself.

## Results

Published run: 15 kernels, 45 held-out workloads, 3 repetitions, one H100.

| Metric | Ratio (rag_lfbo / lfbo) | 95% CI |
|---|---:|---|
| Kernel latency | **0.9923** | [0.9635, 1.0131] |
| Autotune time | **0.8948** | [0.8094, 1.0009] |
| End-to-end time | 0.9066 | [0.8208, 1.0105] |

Verdict `non_inferior_only`: latency is non-inferior at the 1.02 margin, the
completion gate passes at 97.8% (McNemar *p* = 0.750), and the search converges
with a median 564 configs explored versus 780 — but the autotune-time CI upper
bound landed at 1.0009, missing superiority by 0.0009.

Full write-up, per-kernel tables, and the reference numbers to diff a replication
against: [`scripts/helion_rag/results/2026-07-31-h100-loo/`](../scripts/helion_rag/results/2026-07-31-h100-loo/README.md).

## Limitations

One host, one GPU family (H100), one embedding model, one retrieval setting. The
oracle is the corpus's recorded top-5 rebenchmarked, so it bounds "best known",
not "best possible". The corpus is a snapshot: retrieval quality is capped by
what CI happened to have tuned. Fifteen kernels is enough for the paired
statistics used here but leaves per-kernel effects noisy — the `rope_fwd`
regression, for instance, is driven by a single microsecond-scale shape where
launch overhead dominates.
