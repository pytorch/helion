Joint pipeline autotuning
========================

``autotune_pipeline`` is an experimental API for tuning a Python operation that
calls several Helion kernels. It measures the complete operation under CUDA or
HIP graph replay for every representative input. A configuration that makes one
stage faster is accepted only according to the complete pipeline objective.

The return value is a ``PipelineTuningResult`` whose ``.config`` contains the
``PipelineConfig`` bundle. Applying a bundle is scoped to the calling context;
it does not replace the kernels' saved AOT configurations.

Usage
-----

Provide the ordinary pipeline function, an independent reference, a correctness
check, and representative argument tuples. The check receives the actual output
first and the reference output second, and must raise on a mismatch.

.. code-block:: python

   import torch

   from helion.autotuner.pipeline import autotune_pipeline

   # pipeline and pytorch_reference are application-defined callables.
   # inputs contains representative tuples of arguments on the current GPU.
   result = autotune_pipeline(
       pipeline,
       inputs,
       reference=pytorch_reference,
       check=lambda actual, expected: torch.testing.assert_close(
           actual, expected, rtol=1e-3, atol=1e-3
       ),
       algorithm="LLMSeededLFBOTreeSearch",
       effort="full",
       aggregation="geomean",
       max_evaluations=512,
       coordinate_evaluations=96,
   )

   with result.config.activate():
       output = pipeline(*inputs[0])

   result.save("pipeline_tuning.json")

Select tolerances appropriate to the operation. The independent reference and
correctness checks run outside the timing region. Inputs must remain immutable;
pipelines may allocate and overwrite their own scratch storage. Mutation or
reset of caller-owned inputs is not supported by this initial API.

The evaluator validates eager output and graph replay before accepting a
measurement. It freezes the discovered stage dispatch during graph capture.
Capture failures reject a candidate; they do not change the measurement to an
eager timer. All input tensors must use the same current CUDA or ROCm GPU.
Distributed operation and tuning inside an existing graph capture are unsupported.
The pipeline uses eager host dispatch in one host thread and the current GPU
stream. Host control flow may depend on tensor metadata or static arguments, but
must not read GPU tensor values or move work to another stream.
Do not wrap the pipeline or its stage calls in ``torch.compile``: compiled
callables can bypass the scoped Python dispatcher. Ordinary kernels remain
compatible with ``torch.compile`` outside a pipeline scope.

Search and objective
--------------------

The controller searches one stage coordinate at a time while maintaining
complete configuration bundles. Each coordinate uses that stage's native
``ConfigSpec`` and the existing ``LLMSeededLFBOTreeSearch`` implementation. The
LLM search hands its measured configurations to the real LFBO surrogate. The
incumbent, explicit candidate seeds, and ``Settings.autotune_seed_configs`` remain
available. Existing ``HELION_LLM_PROVIDER``, ``HELION_LLM_MODEL``,
``HELION_LLM_EFFORT_LEVEL``, and ``HELION_LLM_FAST_MODE`` controls apply normally.
The full effort profile retains the hybrid's existing LLM seed-stage defaults
and full LFBO settings.

Every candidate, including normal, suspicious, and final coordinate rechecks,
is scored by a complete pipeline evaluation over all supplied inputs. The default
scalar objective is the geomean of their latencies in milliseconds;
``aggregation="max"`` minimizes the maximum latency. Each representative input
has equal weight. No isolated-stage latency is used as a surrogate target.
The LLM prompt includes the pipeline source, coordinate key, current bundle,
per-input traces, and aggregation mode. It explicitly identifies scores as
complete-pipeline graph timings and explains that downstream shapes and stage
counts can change.

If an upstream configuration changes downstream arguments or stage topology,
the evaluator discovers and validates the resulting complete pipeline. The
controller gives new downstream stages bounded configuration trials, retains a
small beam of alternatives, and revisits coordinates across rounds. This is a
budgeted coordinate search, not a guarantee of the global optimum.

The ``benchmark`` callback can replace the default median GPU-event timer. It
receives only an already captured graph's replay callable and must return a
positive latency in milliseconds. This permits a caller to use an established
profiler-based protocol. The callback owns warmup, repetition, and any cache
policy. Coordinate rebenchmarking invokes the complete evaluator again; it never
times the Python evaluator callback itself.

Budgets and evidence
--------------------

``max_evaluations`` limits complete pipeline evaluations during discovery and
search, including downstream trials. ``coordinate_evaluations`` limits evaluator
calls in each coordinate search, including native rechecks. The defaults are
512 pipeline evaluations and 96 calls per coordinate. ``max_seconds`` is an
optional search wall-time limit checked between evaluations. A running GPU
measurement is not interrupted by a time limit.
Existing LLM transport timeouts still apply to requests already in flight.

Final confirmation is separately bounded by ``final_top_k + 1`` complete
measurements and always includes the original incumbent. The result keeps the
incumbent if no confirmed alternative improves the aggregate latency. A small
budget may end during initial seeds or before LFBO begins; the requested method
name does not imply that all its phases completed. Coordinate receipts record
phase measurement counts, successful fresh LLM proposal measurements, seed
handoff, budget exhaustion, and the hybrid breakdown when available.

The saved result contains the selected bundle, per-input timings and speedups,
indices of inputs that regressed,
stage traces, search and confirmation counts, elapsed tuning time, and candidate
history. Coordinate scores are pipeline milliseconds; coordinate elapsed time
is host wall time spent searching. These are different quantities.

For bounded integration tests without an LLM request, supply explicit configs:

.. code-block:: python

   result = autotune_pipeline(
       pipeline,
       inputs,
       reference=pytorch_reference,
       check=check_output,
       algorithm="finite",
       config_candidates=lambda stage: candidates_for(stage),
       max_evaluations=16,
       coordinate_evaluations=4,
   )

Finite mode is reported as ``finite``. It evaluates the incumbent and distinct
explicit/settings seeds and does not invoke an LLM or LFBO.

Dispatch and cache scope
-----------------------

Default stage keys include kernel source identity, backend, structural
configuration-space identity, and runtime argument metadata.
Representative tensors with different values but the same metadata share a
configuration. Different shapes, strides, dtypes, or static arguments can produce
different keys. A custom ``key_fn`` may define an application-specific grouping;
bindings must have compatible backends and configuration spaces, or discovery
raises ``ValueError``. The caller is responsible for choosing a meaningful
grouping. The complete-pipeline check still runs on every input.

The custom key function is preserved by in-memory bundles. Functions are not
serialized, so pass it explicitly when restoring a bundle:

.. code-block:: python

   from helion.runtime.pipeline import PipelineConfig

   bundle = PipelineConfig.from_dict(saved_config, key_fn=my_key_fn)
   with bundle.activate():
       output = pipeline(*inputs[0])

``initial_bundle`` or ``initial_config`` can supply the starting configurations.
Otherwise, discovery resolves existing saved configurations without running a
nested autotuner. A bundle rejects previously unseen stage keys when activated
for ordinary execution unless the caller explicitly enables discovery.
If an exact saved AOT selector has no entry for a newly discovered shape, the
scope uses an explicit reference configuration and records that source.

``cache_path`` stores a separate pipeline receipt and bundle. Cache reuse requires
both ``workload_tag`` and ``benchmark_tag``: the caller must change these when
input-value distributions or the timer protocol change. The cache fingerprint
also records code, argument metadata, and runtime details. Matching cached bundles
are seeds and are validated and measured again; stored latencies are never
reused as current measurements. Isolated-stage timing caches are not consulted
by the coordinate adapter.

Runnable example
----------------

``pretuned_kernels/pipeline_topk/pipeline_topk.py`` compares exact variable-length top-k hierarchies
over full and ragged rows. With 128,000 columns and k=2,048, a tile of 16,384
produces two stages while a tile of 4,096 produces six. The example checks values
and original indices with tie-aware correctness, then measures complete graphs.

From the repository root, run the fixed comparison and a bounded joint search:

.. code-block:: bash

   PYTHONPATH=. python pretuned_kernels/pipeline_topk/pipeline_topk.py --tune --output /tmp/pipeline-topk.json

This uses 16 finite search evaluations plus bounded final confirmation and makes
no LLM requests. Omit ``--tune`` for only the two fixed comparisons. The default
dtype is float32; ``--dtype bfloat16`` selects bfloat16.
