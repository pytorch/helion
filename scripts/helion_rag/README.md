# Helion RAG

Helion RAG builds a searchable corpus of Helion CI autotuning artifacts.

You can use the `setup-helion-rag.sh` script to configure the tools in an existing Helion checkout. It validates the Python interpreter and Manifold access, installs the package in editable mode, fetches the corpus for the selected hardware, and builds a local FAISS index.

You don't need any LLM API keys to build or query the corpus. The "RAG" here is purely retrieval-based and searches over CI benchmark artifacts. The `helion_rag` CLI is standalone: `lookup` prints results and never applies a retrieved configuration for you.

Helion itself can consume this corpus during autotuning, but only when you opt in with `HELION_RAG_ENABLED=1`; that path lives in `helion/autotuner/rag/`, not here. The `experiment/` and `stats/` subpackages drive the four-arm study that measures whether it is worth it — see [`docs/rag_autotuning_experiment.md`](../../docs/rag_autotuning_experiment.md).

The `loo_*` modules drive a second, separate study: whether retrieved configs are worth using as **seeds for the LFBO search** (no LLM involved) — see [`docs/rag_lfbo_warm_start_experiment.md`](../../docs/rag_lfbo_warm_start_experiment.md). It has its own seeding path in `patch.py` under `HELION_RAG_LOO_SEEDING=1`, which is mutually exclusive with `HELION_RAG_ENABLED`.

## Package layout

Here's a quick tour of how the codebase is organized:

```text
helion_rag/
  __init__.py       # Minimal package marker
  __main__.py       # Entry point for `python -m helion_rag`
  cli.py            # Argparse setup and console script entry point
  config.py         # Parses HELION_RAG_* environment variables into a Config dataclass
  corpus.py         # Handles benchmark zips: parsing meta.jsonl, workload hashing, tier-0 checks, and deduplication
  index.py          # FAISS index generation, exact mapping, and similarity search
  safe_index.py     # Signature-checked index loading that fails closed on tampering
  signing.py        # Ed25519 signing and verification of published index generations
  lookup.py         # The core lookup logic (exact match → similar match → miss)
  rerank.py         # Shape-aware reranking of the semantic candidate pool
  ingest.py         # Writes autotune logs back to the corpus idempotently
  upload.py         # Uploads zip runs to Manifold atomically
  manifest.py       # Validates and loads manifest.json
  hardware.py       # Figures out the hardware family based on the device string
  models.py         # Dataclasses for PerfStats, Ref, and ExactEntry
  setup_helpers.py  # CLI helpers for the bash setup script (validating, resolving families, checking artifacts, etc.)
  _util.py          # Shared constants and logging helpers
  experiment/       # Four-arm head-to-head harness (see below)
    head_to_head.py         # Arm table, frozen controls, and the study manifest
    head_to_head_worker.py  # Runs one (workload, arm, repetition) unit in isolation
    scheduler.py            # Balanced Latin-square execution order
    events.py               # The canonical instrumentation event every arm emits
    workloads/              # Registry of benchmarkable kernels; add a file to add a shape
  stats/            # Paired statistics for the study
    paired.py               # Wilcoxon, rank-biserial, wins/ties/losses, bootstrap ratio CIs
    gates.py                # Holm multiplicity control across the contrast family
  patch.py          # Seeds LFBO from retrieved configs (HELION_RAG_LOO_SEEDING)
  shape_distance.py # Log-scaled shape proximity used to rank neighbours
  embedding_text.py # Index/query text composition, shared so both stay in sync
  loo.py            # Leave-one-workload-out fold construction and corpus fingerprints
  loo_inputs.py     # Builds real tensors and loads kernels for a workload
  loo_experiment.py # The three-phase driver: preflight, select, eval
  loo_run.py        # Runs one cell in a fresh subprocess and records its outcome
  loo_report.py     # Cluster bootstrap, completion gate, and the verdict
```

Top-level scripts, all run from the repository root with `PYTHONPATH=scripts/helion_rag`:

```text
run_head_to_head_campaign.py  # Resumable campaign driver
analyze_head_to_head.py       # Tables, statistics, and the gnuplot figure pack
plot_narrative_figures.py     # Optional matplotlib figures (needs the `figures` extra)
results/                      # Checked-in results from the published campaign
```

## Setup

To get started, run the setup script from the repository root or the `scripts/helion_rag/` directory:

```bash
./setup-helion-rag.sh \
  --helion-repo /path/to/helion \
  --manifold-base manifold://helion_ci_artifacts/tree/... \
  --python /path/to/.venv/bin/python \
  --hardware-family h100   # This is also auto-detected if you don't specify it
```

You can pass `--dry-run` to see what the script plans to do without actually mutating anything, or `--non-interactive` to skip prompts. Run `./setup-helion-rag.sh --help` for the full list of options.

If a prerequisite check fails, the script exits before fetching or indexing the corpus.

## Environment

When the setup finishes successfully, it creates a `<helion_repo>/.helion-rag/` directory and adds a source block to your shell's rc file.

You can load it into your current shell like this:

```bash
source /path/to/.helion-rag/env.sh   # (or source ~/.bashrc if you restarted your shell)
```

The `env.sh` script exports several configuration variables:

| Variable | What it does |
|---|---|
| `HELION_RAG_HARDWARE_FAMILY` | The hardware family we resolved (e.g., `h100`). |
| `HELION_RAG_MANIFOLD_BASE` | The base Manifold URI where we push manifest artifacts. |
| `HELION_RAG_MANIFEST` | Path to your local `manifest.json`. |
| `HELION_RAG_DATA_DIR` | Directory for `ci_artifacts/` (this gets wiped and re-fetched on each run). |
| `HELION_RAG_INDEX_DIR` | Where the per-family FAISS indexes live (`rag_index/`). |
| `HELION_RAG_WRITEBACK_DIR` | Local storage for ingested autotune runs (`rag_writeback/`). |
| `HELION_RAG_AUTOTUNE_LOG_DIR` | Where collected autotune logs go (`autotune_logs/`). |
| `HELION_RAG_UPLOADS_DIR` | Holds archives and markers for Manifold uploads (`uploads/`). |
| `HELION_EMBED_MODEL` | The embedding model ID (defaults to `Qwen/Qwen3-Embedding-8B`). |
| `HELION_RAG_EMBED_DEVICE` | Forces CPU or CUDA for embeddings (auto-detected by default). |

Because the package is installed in editable mode, you'll have access to the `helion-rag` CLI command right away. If you haven't sourced the environment variables, you can pass `--env-file` or set `HELION_RAG_ENV_FILE`.

## CLI subcommands

The `helion-rag` command gives you a few different tools:

```bash
helion-rag extract          # Unzips benchmark archives, strips generated code, and deduplicates configs
helion-rag index [--force]  # Builds the FAISS generation index for your hardware family
helion-rag lookup --kernel-source-file f.py --shapes '...' --dtypes '...' --hardware h100 [--settings-json '{}']
helion-rag ingest           # Idempotently merges meta.jsonl and csv logs, updates the ledger, and rebuilds the index
helion-rag upload [--dry-run] [--reupload]   # Zips unuploaded runs to Manifold, marking them only if the upload succeeds
helion-rag setup-helper --help
helion-rag setup-helper publish-manifest --manifold-base <uri> --family <fam> [--artifact-path p] [--alias a...]
python -m helion_rag <cmd>  # Alternative module entry point (does the same thing as the console script)
```


## Lookup tiers

The standalone `lookup` command searches in three stages:

- **Tier 0 (Exact Match):** Calculates a workload key by hashing the normalized AST source, codegen settings, canonical shapes, dtypes, and hardware family. A matching `exact.json` entry returns the measured-best config.
- **Tier 1 (Similar Match):** Runs FAISS similarity search over corpus source embeddings and returns the top neighbors with their measured configurations. `HELION_RAG_SIM_THRESHOLD` accepts a finite value in `[0, 1]` and controls the minimum score; invalid values use the default `0.85`.
- **Tier 2 (Miss):** Reports that no exact or sufficiently similar result was found.

The command prints these results as JSON for inspection or use by external tooling. Helion does not consume them automatically.

To keep workload keys in sync with upstream Helion, `_CODEGEN_SETTINGS` and `_codegen_signature` are imported directly from `helion.autotuner.metrics` — a single source of truth, so there is nothing to keep in sync.

## Index publication

Index builds are single-writer per hardware family. A second builder fails immediately while the first holds the advisory filesystem lock. Publication writes a temporary generation, atomically renames it to its numeric final directory, and then atomically replaces the `current` pointer. Before the next build, stale temporary directories are removed and a completed numeric generation newer than `current` is promoted, recovering a crash between the final rename and pointer update. Generation IDs encode publication order, not relative quality, and manual rollback by editing `current` is unsupported because recovery always promotes the newest completed generation.

## How we resolve hardware

When figuring out what hardware you're running on, `hardware.resolve_family()` checks a few things in this specific order:
1. An explicit override
2. The `env_family` environment variable
3. Device token substring matches
4. `torch.cuda.get_device_name`
5. Aliases in the manifest
6. Compute capability

If it still doesn't recognize the hardware, it returns `None` and drops down to a Tier-2 miss. The pure resolution helpers (`_family_from_device`, `_family_from_manifest_alias`, `_family_from_compute_capability`) take explicit inputs, so they're unit-tested directly without needing to stub the `torch` device probe.

## Manifest handling

`manifest.validate_manifest()` makes sure we're using a version 1 schema where the families dictionary contains an `artifact_path` and a list of aliases.

If you need to add a new family, `setup_helpers.publish_manifest()` will safely merge it into the existing `manifest.json` on Manifold without clobbering anything else. If the family is already there, it just does nothing.

## Ingest and upload

- **Ingest:** This command reads `*.meta.jsonl` and `perf.csv` files from your autotune log directory. It joins them by `run_id` and `config_id`, filters out anything that didn't pass, and aggregates the stats (median, mean, n_samples). It writes the result to the local writeback directory as `local-autotune.meta.jsonl` and updates `ledger.json` so we don't ingest the same `run_id` twice. Finally, it can optionally trigger a reindex.
- **Upload:** This command grabs any unuploaded runs from your logs, builds a `batch-manifest.json`, and zips everything into the uploads directory. It uses `manifold_put` to push to Manifold. To make sure everything is atomic, it only writes the per-run success markers if the upload actually succeeds. If it fails, it throws an error without marking anything as uploaded.

## Tests

The test suite covers the standalone corpus, index, lookup, ingest, upload, manifest, and hardware-resolution paths, plus the four-arm harness and its analysis:

```text
scripts/helion_rag/tests/
  __init__.py                 # Package marker for relative imports
  _fixtures.py                # Shared setup imported via inspect to avoid duplication
  test_corpus_ingest_upload.py   # Covers extraction (incl. the CLI path), idempotent ingest, atomic uploads
  test_corpus_extended_schema.py # Extended-vs-historical record schema and tier-0 eligibility
  test_index_generation.py       # Single-writer publication and interrupted-build recovery
  test_safe_index.py             # Fail-closed loading of a tampered index
  test_signing.py                # Ed25519 signing and verification
  test_rollback.py               # Republishing last-good content as a newer generation
  test_lookup.py                 # Tier-0 exact and Tier-2 miss over an on-disk bundle
  test_rerank.py                 # Shape-aware reranking of the candidate pool
  test_integration.py            # Tier-1 lookup over a real FAISS index
  test_manifest_hardware.py      # Manifest validation and hardware family resolution (pure helpers)
  test_setup_environment.py      # env.sh generation from the setup helpers
  test_util.py                   # Similarity-threshold input contract
  test_workload_key.py           # Workload key parity, deduplication, and AST checks
  test_experiment_events.py      # The canonical instrumentation event schema
  test_experiment_scheduler.py   # Latin-square balance
  test_head_to_head.py           # Arm table, frozen controls, manifest hashing
  test_head_to_head_campaign.py  # Driver resume, locking, and atomic result recording
  test_head_to_head_analysis.py  # Tables, estimands, and figures on a synthetic campaign
  test_plot_narrative_figures.py # Optional matplotlib figures
  test_stats_gates.py            # Holm multiplicity control
```

You can run the suite like this:
```bash
PYTHONPATH=scripts/helion_rag .venv/bin/python -m pytest scripts/helion_rag/tests -q
```

The Helion-side RAG adapter is tested separately under the repository's own suite
(`test/test_rag_*.py`).

## Leave-one-workload-out LFBO study

Measures whether warm-starting `LFBOTreeSearch` from retrieved neighbour configs
makes the search cheaper without costing kernel quality. Three resumable phases:

```bash
PYTHONPATH=scripts/helion_rag .rag-venv/bin/python -m helion_rag.loo_experiment \
  --phase preflight --env-file .helion-rag/env.sh --family h100 \
  --output-dir .helion-rag/loo_evaluation/my-run --code-revision my-run-r1
```

Repeat with `--phase select` and `--phase eval`, then analyse with
`python -m helion_rag.loo_report`. Add `--dry-run` to a phase to write the
planned matrix and stop. Design, controls, and the published H100 results are in
[`docs/rag_lfbo_warm_start_experiment.md`](../../docs/rag_lfbo_warm_start_experiment.md).

## Prerequisites

There are a few things you need to have in place before setting this up. If any of these are missing, the setup script will fail loudly and safely:

| Requirement | Why you need it | What to do |
|---|---|---|
| Network access for `pip install -e` | To install `helion-rag` in editable mode. | Make sure pip works, or pre-install offline using `pip install --no-build-isolation --no-deps -e scripts/helion_rag`. |
| `manifold` CLI with read access | To fetch the manifest and corpus. | Ensure `manifold ls <base>` works. |
| `manifold` write access (optional) | To upload runs and publish manifests. | Ensure `manifold put` works under your base URI. |
| `HF_TOKEN` (optional) | To download the embedding model for the first time. | Run `huggingface-cli login` or export `HF_TOKEN`. |
| Sourced `env.sh` | So lookup/ingest commands can read `HELION_RAG_*` paths. | Source `<helion_repo>/.helion-rag/env.sh` or pass `--env-file`. |
| A `~/.bashrc` block | To auto-load the environment in new shell sessions. | It appends safely; you can remove the block manually to opt out. |
| Embedding model / device | The default 8B model is quite heavy. | Override with `HELION_EMBED_MODEL` and set `HELION_RAG_EMBED_DEVICE=cpu` if needed. |
| Rebuild index | The indexer skips rebuilding if the index already exists. | Run `helion-rag index --force` or delete the index directory to rebuild. |

## Add data about new hardware

If you're running on a new hardware family that isn't in the manifest yet, you won't have a corpus to fetch. Here's how to onboard it:

1. The setup script writes `env.sh` and can optionally configure local autotune collection.
2. Run your autotuning jobs normally. Then, run `helion-rag ingest` to build your local index, and `helion-rag upload` to contribute your runs back to Manifold (`.../contrib/<family>/`).
3. Finally, register the new family:
   ```bash
   helion-rag setup-helper publish-manifest --manifold-base <uri> --family <fam> [--artifact-path p] [--alias ...]
   ```
   This safely merges your new family into the existing `manifest.json` on Manifold.
