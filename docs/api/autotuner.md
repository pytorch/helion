# Autotuner Module

The `helion.autotuner` module provides automatic optimization of kernel configurations.

Autotuning effort can be adjusted via :attr:`helion.Settings.autotune_effort`, which configures how much each algorithm explores (``"none"`` disables autotuning, ``"quick"`` runs a smaller search, ``"full"`` uses the full search budget). Users may still override individual autotuning parameters if they need finer control.

```{note}
**Guard your entry script with `if __name__ == "__main__":`.** By default Helion
benchmarks (and may precompile) candidate configs in a *spawned* subprocess,
which re-imports your entry module. If the top-level kernel call is not under an
`if __name__ == "__main__":` guard, the worker re-runs it on import and
autotuning aborts with `NoConfigFound` (often with `failed to send job to
worker`). Either add the guard, or set `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0`
to benchmark in-process.
```

```{eval-rst}
.. currentmodule:: helion.autotuner

.. automodule:: helion.autotuner
   :members:
   :undoc-members:
   :show-inheritance:
```

## Choosing a source-optimization handoff point

Ordinary kernel calls can run the complete workflow without custom orchestration:

```bash
HELION_AUTOTUNE_HANDOFF=1 \
HELION_AUTOTUNE_LOG=/tmp/rms/run \
HELION_AUTOTUNE_LOG_DETAILS=1 \
HELION_AUTOTUNE_BUDGET_SECONDS=300 \
HELION_AUTOTUNE_HANDOFF_BUDGET_SECONDS=1500 \
HELION_HANDOFF_AGENT=codex HELION_HANDOFF_EFFORT=ultra \
HELION_BACKEND=cute CUDA_VISIBLE_DEVICES=0 \
python your_kernel.py
```

This allows five minutes for configuration search and 25 minutes for native-source
rounds, plus confirmation and export. Without a search budget, handoff waits for
search completion. The configured autotune method and cache behavior
remain in effect; use `HELION_SKIP_CACHE=1` for a fresh search.

`HELION_HANDOFF_AGENT` selects an authenticated CLI: `codex` (default) or `claude`.
Codex defaults to model `gpt-6-astra` and effort `ultra`; Claude defaults to `opus`
and `max`. Override them with `HELION_HANDOFF_MODEL` and `HELION_HANDOFF_EFFORT`,
using values supported by the selected CLI. For Claude, replace the agent line
above with `HELION_HANDOFF_AGENT=claude HELION_HANDOFF_EFFORT=max`. Each run starts
one continuous session with memory disabled and receives the native kernel
contract and optimization objective. The agent edits sources and runs
`python submit_candidate.py` whenever it has a candidate. The command returns
correctness, latency, and acceptance feedback in the same conversation; each
submission records a round. Native progress joins
the existing `.trace.jsonl` when detailed logging is enabled; native scores remain separate
from configuration timings. Bundles and agent records are saved in a unique
directory under `<log base>.handoff/`, or `/tmp/helion-handoff/` without a log path.
When the agent session ends, its full chat history, including tool calls and
responses, is saved as `<log base>.handoff.<session_id>.chat.jsonl` beside the
autotune logs. Each session has its own file; this requires `HELION_AUTOTUNE_LOG`
and does not require `HELION_AUTOTUNE_LOG_DETAILS`.
The agent receives the task prompt, native sources, and submission feedback.
The full manifest and raw autotune history stay in the bundle for evaluation
and reproducibility; they are not copied into the agent workspace.

Handoff is off by default. Logging alone never launches an agent. The hook runs
when a kernel actually autotunes; an already configured kernel or a single pinned
config keeps its normal fast path. The Helion call still uses its selected config;
the native winner is saved for direct use outside Helion. The explicit APIs below
remain available for custom stopping policies and agent adapters.

`find_handoff` runs an autotuner until a user-selected stopping point or search
completion, then confirms a starting kernel for source optimization. It is
opt-in; ordinary `autotune()` calls are unchanged.

```python
from helion.autotuner import HandoffPolicy, LFBOTreeSearch, find_handoff

bound = kernel.bind(args)
point = find_handoff(
    LFBOTreeSearch(bound, args),
    HandoffPolicy(after_trials=100),
)
# point.config and point.fn identify the selected starting kernel.
# point.measurements retains search trials and fresh confirmation measurements.
```

Use `HandoffPolicy(after_trials=100)`, `HandoffPolicy(after_seconds=60)`, or
`HandoffPolicy(callback=lambda progress: ...)` for explicit control. The default
`HandoffPolicy()` waits for search completion. Enabled triggers are alternatives;
limits are checked after each completed benchmark batch and can overrun by that
batch. Trials include rejected candidates. Hybrid stages share one counter and
one stop. Callbacks receive a frozen progress snapshot; distributed runs invoke
them on each rank and synchronize the decision.

If `autotune_baseline_fn` triggers autotuning of another Helion kernel, those
trials share the handoff session and can interfere with selection.

Multi-shape searches count the combined source identity across all shapes and
preserve their aggregate objective.

Selection rechecks correctness and benchmarks up to `finalists=5` distinct
sources, plus the returned config, using `repetitions=3` fresh passes with rotating
order. It chooses the lowest median among candidates that pass every repetition.
Final confirmation after stopping can extend the time limit. Confirmation does
not count as exploration. A cache-wrapped search is also accepted: cache hits are
confirmed and report `"completed"` with zero search trials. Completed searches
retain normal cache behavior; early handoff unwinds before the cache write.

The returned `HandoffPoint` includes the reason, callable, config, finalist
samples, and trial history. `autotune_log_details` records handoff and confirmation
events in the existing trace. This API selects the starting point; agent execution
and source editing are separate steps.

## Handing native source to an agent

`build_handoff` exports a selected kernel into an editable directory and checks
the standalone baseline before handing it to an agent. Keep the search object
used by `find_handoff`:

```python
from helion.autotuner import build_handoff

bundle = build_handoff(search, point, "handoff_workspace")

def agent(workspace):
    # Your agent reads workspace.prompt and workspace.sources, then returns
    # {"case_0/kernel.py": "<complete replacement native source>"}.
    return my_source_agent(workspace.prompt, workspace.sources)

result = bundle.run_agent(agent)

# Or optimize for a time budget, keeping only validated source improvements.
session = bundle.run_agent_rounds(agent, budget_seconds=600)
result = session.evaluation
```

The callback receives a workspace, with no `BoundKernel` or live autotuner.
It proposes native CuTe, Triton, or other backend source changes. The prompt
contains the native kernel contract and optimization objective. Measurements,
autotune history, and configs stay in the bundle records.

Each `case_N/` contains `kernel.py`, an `original.py` backup, saved inputs, and
frozen reference outputs and post-call inputs. `manifest.json` records workload,
hardware, dependencies, specialization metadata, tolerances, and autotune
evidence; `prompt.md` is the agent prompt. Multi-shape searches export a separate
module for each bound shape and retain their aggregate objective.

Output references use an explicit `reference_fn`, then `autotune_baseline_fn`, or
the confirmed original kernel. The original kernel supplies the mutation and
alias contract. Evaluation runs in a fresh worker with a per-case timeout and
checks outputs, aliases, and input mutations. Every timing sample uses restored
input values, excluding restoration from timing; graph replay keeps stable
storage addresses. Source snapshots, raw samples,
medians, median absolute deviations, and failures are saved under `evaluations/`.
Completed timing samples survive worker timeouts, but failed cases have no score.
Run `python handoff_workspace/evaluate.py` to re-evaluate edits, or reopen the
workspace with `HandoffBundle(Path("handoff_workspace"))`.

Each case writes a phase journal under `evaluations/`, so failures and timeouts
retain the last recorded phase (load, compile/launch, capture, warmup,
measurement, or validation) alongside their source snapshot and diagnostics.

New NVIDIA CUDA bundles default to `timing="cuda_graph"`, which places CUDA
events inside graph replay and excludes host dispatch, input restoration, and
L2 clearing. An explicit `timing` argument to `build_handoff` selects
`"cuda_graph"`, `"cuda_event"`, or `"wall_clock"`; saved bundles retain their
declared policy. Capture failures are reported instead of switching timers.
Direct benchmarks of GPU callables can use the same primitive:

```python
from statistics import median
from helion.autotuner.benchmarking import do_bench_cuda_graph

bundle = build_handoff(search, point, "graph_handoff", timing="cuda_graph")
# For a callable that does not mutate its inputs:
reference_ms = median(
    do_bench_cuda_graph(lambda: reference(*args), return_mode="median")
    for _ in range(5)
)
```

Graph timing requires capturable CUDA work. Arbitrary Python, container, CPU
tensor, or tensor-metadata mutation is unsupported because graph replay does
not repeat host operations. Select a legacy timing policy for such workloads;
the evaluator never silently falls back. Mutating direct benchmarks must
supply a capturable `reset` callback; bundle evaluation restores the saved CUDA
input storages automatically. This timer does
not change ordinary autotuner measurements. Remeasure searched sources under
the same policy before comparing them with native proposals. The existing
`autotune_benchmark_fn` setting is a batched rebenchmark callback, not this
single-call timer.

`run_agent` applies and evaluates one proposal, retaining its existing behavior.
`run_agent_rounds` adds incumbent selection and rollback. Set `budget_seconds`;
rounds track progress without limiting submissions. It first validates the
starting source, then compares each proposal against a nearby incumbent
measurement, alternating measurement order across rounds. A proposal must pass
every case and improve the aggregate
objective by more than both 0.1% and three times the larger noise estimate.
The estimate is the objective score times the largest per-case MAD/median; this
preserves the objective's units for multiple shapes and relative-latency ratios.
This is a selection heuristic, not a confidence interval.
Failed, noisy, and slower proposals restore the incumbent. Callback exceptions
are recorded and optimization continues while time remains.

Omitting the callback uses `CLISourceAgent`, selected by `HELION_HANDOFF_AGENT`:

```python
session = bundle.run_agent_rounds(budget_seconds=1500)
```

The time budget includes initial validation, agent work, and evaluation. The
default agent keeps one process and workspace until it exits or reaches the budget.
After each submission, its workspace holds the retained source, ready for further
edits. Unsubmitted edits are never evaluated.

The prompt includes an absolute `time.monotonic()` deadline. Custom synchronous
callbacks must enforce their own timeout; the controller rejects late results
and passes the deadline to evaluation. Provider selection
and context management remain the callback's responsibility. The original prompt
is restored when the controller returns.

Each invocation records an immutable starting-source snapshot, per-round source
and evaluation records, `feedback.json`, and `result.json` in a new `agent_runs/`
directory. `HandoffAgentResult` exposes the baseline, retained evaluation, and
`HandoffAgentRound` decisions. The original source backup remains unchanged;
only validated improvements replace the editable source. Reopening a bundle
does not resume an earlier agent conversation.

The saved workload covers the supplied inputs, not every possible
input value or shape. Custom accuracy/benchmark callbacks and distributed
workloads are currently unsupported. Backend standalone-export restrictions
also apply. The evaluation harness requires Helion; the exported native kernels
can be imported and called directly without it.

## Configuration Classes

### Config

```{eval-rst}
.. autoclass:: helion.runtime.config.Config
   :members:
   :undoc-members:
```

## Search Algorithms

The autotuner supports multiple search strategies:

### Pattern Search

```{eval-rst}
.. automodule:: helion.autotuner.pattern_search
   :members:
```

### LFBO Pattern Search

```{eval-rst}
.. automodule:: helion.autotuner.surrogate_pattern_search
   :members:
   :exclude-members: LFBOTreeSearch
```

### LFBO Tree Search (Default)

{py:class}`~helion.autotuner.surrogate_pattern_search.LFBOTreeSearch` is the default autotuner.
It extends LFBO Pattern Search with tree-guided neighbor generation, using greedy decision tree
traversal to focus search on parameters the surrogate model has identified as important.

```{eval-rst}
.. autoclass:: helion.autotuner.surrogate_pattern_search.LFBOTreeSearch
   :members:
   :show-inheritance:
```

### LLM-Guided Search

{py:class}`~helion.autotuner.llm_search.LLMGuidedSearch` uses a large language model to
iteratively propose kernel configurations. It sends the kernel source, config space, GPU
hardware info, and benchmark results to the LLM, which suggests promising configurations
across multiple refinement rounds.

```{eval-rst}
.. automodule:: helion.autotuner.llm_search
   :members:
```

#### LLM Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HELION_LLM_PROVIDER` | | LLM provider: `anthropic`, `openai`, `bedrock`, or `vertex` |
| `HELION_LLM_MODEL` | | Model name (e.g. `claude-opus-4-7`, `gpt-4o`) |
| `HELION_LLM_API_KEY` | | API key (falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). Optional for `vertex`/`bedrock`, which authenticate by client identity. |
| `HELION_LLM_API_BASE` | | Custom API base URL |
| `HELION_LLM_VERTEX_PROJECT` | | Cloud project id for the `vertex` provider (falls back to `ANTHROPIC_VERTEX_PROJECT_ID`) |
| `HELION_LLM_VERTEX_LOCATION` | `global` | Vertex AI location for the `vertex` provider (falls back to `CLOUD_ML_REGION`) |
| `HELION_LLM_EXTRA_HEADERS` | | Extra HTTP headers as a JSON object or newline-separated `Header: value` lines (for gateways that require custom routing/identity headers) |
| `HELION_LLM_COMPILE_TIMEOUT_S` | `15` | Compile timeout (seconds) for LLM-proposed configs |
| `HELION_LLM_CA_BUNDLE` | | Custom CA bundle path (for corporate proxies that do TLS inspection) |
| `HELION_LLM_CLIENT_CERT` | | Client certificate path (for proxies requiring mutual TLS) |
| `HELION_LLM_CLIENT_KEY` | | Client key path (for proxies requiring mutual TLS) |

The proxy/TLS variables are only needed in corporate environments where a proxy intercepts
HTTPS traffic. Most users connecting directly to the LLM API can ignore them.

The `vertex` provider talks to Anthropic models hosted on Google
[Vertex AI](https://docs.anthropic.com/en/api/claude-on-vertex-ai) (the publisher
`rawPredict` endpoint). It is the analog of `bedrock`: the model is named in the
URL and authentication is by client identity (a bearer token via
`HELION_LLM_API_KEY`, and/or mTLS via the client-cert variables), so it also
works behind an API-compatible gateway. The endpoint, project, and location each
fall back to the standard Anthropic-on-Vertex SDK variables
(`ANTHROPIC_VERTEX_BASE_URL`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`)
when the helion-specific knob is unset, so an environment already configured for
Anthropic-on-Vertex needs only `HELION_LLM_PROVIDER=vertex` and a model. Example:

```bash
export HELION_AUTOTUNER=LLMGuidedSearch
export HELION_LLM_PROVIDER=vertex
export HELION_LLM_MODEL=claude-sonnet-4-5
export HELION_LLM_API_BASE=https://<your-vertex-endpoint>/v1
export HELION_LLM_VERTEX_PROJECT=<your-project>
export HELION_LLM_VERTEX_LOCATION=us-east5  # optional, default: global
```

### LLM-Seeded Search (Hybrid)

{py:class}`~helion.autotuner.llm_seeded_lfbo.LLMSeededSearch` is a two-stage hybrid approach:

1. **Stage 1 (LLM)**: Run LLM-guided search for a configurable number of rounds to find good
   initial configs.
2. **Stage 2 (Surrogate)**: Run a non-LLM search algorithm (default: `LFBOTreeSearch`),
   seeded with the best LLM config and trained on all LLM benchmark results.

This combines the LLM's ability to make informed initial guesses with the surrogate model's
efficient local search.

{py:class}`~helion.autotuner.llm_seeded_lfbo.LLMSeededLFBOTreeSearch` is a convenience subclass
that locks stage 2 to `LFBOTreeSearch`.

```{eval-rst}
.. automodule:: helion.autotuner.llm_seeded_lfbo
   :members:
```

#### Hybrid Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HELION_HYBRID_SECOND_STAGE_ALGORITHM` | `LFBOTreeSearch` | Override the second-stage search algorithm |
| `HELION_HYBRID_LLM_MAX_ROUNDS` | *(effort-dependent)* | Override the number of LLM rounds in stage 1 |

To use the LLM-guided autotuner, set the `HELION_AUTOTUNER` environment variable:

```bash
# Pure LLM-guided search
export HELION_AUTOTUNER=LLMGuidedSearch

# LLM-seeded hybrid (recommended)
export HELION_AUTOTUNER=LLMSeededLFBOTreeSearch
```

#### Example: Using Claude as the LLM provider

```bash
export HELION_AUTOTUNER=LLMSeededLFBOTreeSearch
export HELION_LLM_PROVIDER=anthropic
export HELION_LLM_MODEL=claude-opus-4-7
export HELION_LLM_API_KEY=your-key-here
```

Then run your kernel as usual — the autotuner will use Claude to propose initial configs
before handing off to the surrogate-based search:

```python
out = matmul(torch.randn([2048, 2048], device="cuda"),
             torch.randn([2048, 2048], device="cuda"))
```

### DE Surrogate Hybrid

```{eval-rst}
.. automodule:: helion.autotuner.de_surrogate_hybrid
   :members:
```

### Differential Evolution

```{eval-rst}
.. automodule:: helion.autotuner.differential_evolution
   :members:
```

### Random Search

```{eval-rst}
.. automodule:: helion.autotuner.random_search
   :members:
```

### Finite Search

```{eval-rst}
.. automodule:: helion.autotuner.finite_search
   :members:
```

### Local Cache

```{eval-rst}
.. automodule:: helion.autotuner.local_cache
    :members:
```

### Search Space Logger

The `search_space_logger` module provides tools to analyze and log the valid search space during autotuning, including which features are enabled/disabled, total search space size, coverage metrics, and per-feature exploration tracking.

```{eval-rst}
.. automodule:: helion.autotuner.search_space_logger
    :members:
    :undoc-members:
    :show-inheritance:
```

#### Search Space Analysis Example

When search space logging is enabled (opt-in), you'll see output like:

```
============================================================
Autotune Search Space Analysis
============================================================
Search space for flash_attention_fwd:
  Backend: cute, Hardware: NVIDIA H100
  Total search space size: 2,457,600
  Search dimensions: 8
  Disabled features (12):
    - num_warps: CuTe backend (uses num_threads)
    - pallas_loop_type: CuTe backend (no Pallas loops)
    ...
  Search algorithm: DifferentialEvolutionSearch
  Time: 245.3s, Configs tested: 128
  Configs attempted: 512 (500 valid, 12 invalid, 97.7% valid)
  Overall search space coverage: 0.005208%
  Per-feature exploration:
    Average feature coverage: 31.2%
    Minimum feature coverage: 8.3%
    - loop_orders: 12/120 options tested (10.0%)
    - pid_type: 2/8 options tested (25.0%)
    - num_threads: 128/256 options tested (50.0%)

  Features with <50% exploration:
    - loop_orders: only 12 of 120 values tested
    - pid_type: only 2 of 8 values tested

Search Coverage:
  Configs tested: 128
  Total space: 2,457,600
  Coverage: 0.005208%
  Search algorithm: DifferentialEvolutionSearch
  Time elapsed: 245.3s
```

This helps identify if the search algorithm is properly exploring the space or getting stuck in local optima.
