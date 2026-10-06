# Helion joint pipeline autotuning

This experimental feature tunes a complete operation that calls several Helion
kernels. `helion.autotune_pipeline` selects a coherent configuration bundle using
the operation's GPU graph replay latency across representative inputs. It is
designed for cases where optimizing each stage independently can make the full
operation slower.

## Motivation: K4 variable-length top-k

The motivating kernel is the exact variable-length top-k implementation used by
K4 in the ROCm benchmark suite. It returns top-k values and their original indices
for each row's valid prefix, including padding for rows shorter than k. The
standalone reproduction is [pretuned_kernels/pipeline_topk/pipeline_topk.py](pretuned_kernels/pipeline_topk/pipeline_topk.py).

Each stage keeps up to k candidates from every tile. Reducing a stage's tile
width can make that stage faster while producing more candidates for later
stages. Tuning only the first stage cannot account for that extra work.

For FP32 inputs shaped `[24, 128000]` with `k=2048`, the tile width changes the
entire hierarchy. The widths below include the original input and final output:

| Tile width used at each stage | Successive widths | GPU stages |
| --- | --- | ---: |
| 16384 | 128000 → 16384 → 2048 | 2 |
| 4096 | 128000 → 65536 → 32768 → 16384 → 8192 → 4096 → 2048 | 6 |

The shape determines this topology; full and ragged row lengths use the same
stage counts. Their values and lengths can still change a stage's device work,
so both inputs participate in the tuning objective.

## Feature and search

The API accepts the public operation, representative argument tuples, an
independent reference, and a correctness check. The default objective is the
geomean of complete-operation latencies across all supplied inputs. An optional
maximum-latency objective is also available. Inputs sharing a stage dispatch key
share its configuration, and every candidate is checked on every input.

```python
import helion

result = helion.autotune_pipeline(
    pipeline,
    representative_inputs,
    reference=reference,
    check=check,
    algorithm="LLMSeededLFBOTreeSearch",
    effort="full",
    aggregation="geomean",
)

with result.config.activate():
    output = pipeline(*representative_inputs[0])

result.save("pipeline_tuning.json")
```

The implementation uses the existing LLM-seeded LFBO search for each stage
coordinate, with that stage's native configuration space. Every proposal and
recheck receives a complete-pipeline score. The LLM prompt includes the pipeline
source, current bundle, representative stage traces, and objective. Existing
`HELION_LLM_*` provider, model, and effort settings apply.

An upstream proposal may change intermediate shapes or introduce new stage
keys. The evaluator realizes that complete pipeline before measuring it. The
controller tries configurations for newly introduced stages, retains a bounded
beam of alternative topologies, and revisits coordinates across rounds. Each
branch can continue improving even while its intermediate result is slower than
the global incumbent. Final confirmation always includes the original bundle.

Dispatch is scoped: applying a bundle does not replace saved AOT configurations.
Discovery resolves saved configs or explicit defaults without launching a nested
stage autotuner. Once prepared, dispatch is frozen for graph capture. Input
cloning, compilation, references, and correctness checks are outside the timing
region. Capture failure rejects the candidate; it never substitutes eager timing.
A custom benchmark callback receives only the captured graph's replay callable,
allowing an application to retain its existing profiler and cache policy.

## Measured result

A bounded validation run on an AMD Instinct MI350X used the standalone top-k
example, FP32 `[24, 128000]`, `k=2048`, and two representative inputs: full and
ragged rows. The table reports complete HIP graph replay times from GPU events,
with the default median event timer.

| Configuration | Stages | Full rows (µs) | Ragged rows (µs) | Geomean (µs) |
| --- | ---: | ---: | ---: | ---: |
| Tile 4096 throughout | 6 | 149.482 | 147.642 | 148.559 |
| Retained tile 16384 incumbent, final confirmation | 2 | 87.761 | 87.421 | 87.591 |

The retained two-stage bundle is **1.696× faster**, with **41.0% lower latency**,
than the six-stage configuration. The joint run used 16 finite search evaluations
and two final confirmations, taking 2.176 seconds of recorded tuning time. It
explored two-, three-, and four-stage candidates and retained the original
two-stage incumbent.

This demonstrates avoiding the topology regression associated with stagewise
tuning. It does **not** establish a new kernel or an improvement over the original
pre-LLM two-stage configuration. The example's event timings also differ from
the paper's ROCm activity timing protocol and must not replace paper measurements.

Validation receipt: `/tmp/helion-pipeline-topk-joint-final.json`.
SHA256: `7c1c9254ddb6f4c597396df96227bfa147b6a0e9682b78cdcafc7dc8e49b8dea`.

Reproduce the bounded comparison without an LLM request:

```bash
PYTHONPATH=. python pretuned_kernels/pipeline_topk/pipeline_topk.py \
  --tune --output /tmp/pipeline-topk.json
```

### Actual library measurements with the paper protocol

The actual `helion_kernel_library.ops.top_k_varlen` operation was separately
remeasured on all ten K4 paper cases after restoring the retained pre-LLM AOT
configs. These measurements use the paper's HIP graph activity timer, five runs,
and each case's original physical GPU. They include the complete operation and
use the original inputs, correctness checks, warmup, and cache policy.

| Input shape, dtype, k | Previous paper: stagewise configs (µs) | Retained configs, fresh measurement (µs) | Latency reduction |
| --- | ---: | ---: | ---: |
| `[24, 128000]`, FP32, 2048 | 139.601 | 83.600 | 40.12% |
| `[24, 128000]`, BF16, 2048 | 91.400 | 55.800 | 38.95% |

Across all ten cases, the geomean speedup over the previous stagewise-tuned paper
timings is **1.341×**, or **25.42% lower latency**. Nine cases improved. The small
BF16 `[4, 4096]`, `k=512` case changed from 13.320 to 13.399 µs, a 0.59% regression.

No kernel body changed and no additional autotuning was performed for these
measurements. The restored configurations were already committed in `45845c1`.
Relative to the original pre-LLM timings for those same configs, the fresh
measurements have a geomean speedup of 0.99014×, approximately 1% slower. The
reported gain therefore measures recovery from the stagewise-tuning regression;
it does not show an improvement over the earlier retained incumbent.

The library receipt is
[`repro/breadth/results/k4_retained_graph_20261006/receipt.json`](../helion-kernel-library/repro/breadth/results/k4_retained_graph_20261006/receipt.json),
with SHA256
`ec2b53b253053ee2d30a4841ee9f33a1af04e4da9072d90f4e49996a706a2392`.
The compact paper overlay is
[`repro/breadth/final_result/rocm/k4_retained_graph_20261006.json`](../helion-kernel-library/repro/breadth/final_result/rocm/k4_retained_graph_20261006.json).

## Verification and current limits

Verification passed 125 CPU tests with 23 subtests and four bounded ROCm
integration tests. The GPU tests cover repeated stages sharing a key, ordinary
and direct-bound dispatch state preservation, saved AOT resolution with implicit
autotuning forbidden, ordinary `torch.compile` compatibility outside pipeline
scopes, and BF16 top-k ties, zero rows, and ragged lengths. The actual graph
comparison above also passed complete-output and poisoned-replay checks.

The LLM integration tests use fake responses and exercise the native search
handoff without network requests. The measured GPU search used explicit finite
candidates; a live LLM-seeded pipeline tuning campaign has not been run.

The initial API requires immutable external inputs on one GPU, eager host
dispatch, and a single stream. Pipelines may allocate and overwrite internal
scratch. General input-reset protocols, distributed execution, and compiled host
pipelines inside the configuration scope are unsupported. Search is bounded and
does not guarantee the global optimum. Aggregate improvement can include an
individual-input regression, which the result reports explicitly. Receipts
record the actual search phases, per-input timings, traces, evaluation counts,
and tuning time; a requested LLM search method alone does not prove that every
phase completed before its budget expired.
