# Autotuner Module

The `helion.autotuner` module provides automatic optimization of kernel configurations.

Autotuning effort can be adjusted via :attr:`helion.Settings.autotune_effort`, which configures how much each algorithm explores (``"none"`` disables autotuning, ``"quick"`` runs a smaller search, ``"full"`` uses the full search budget). Users may still override individual autotuning parameters if they need finer control.

```{eval-rst}
.. currentmodule:: helion.autotuner

.. automodule:: helion.autotuner
   :members:
   :undoc-members:
   :show-inheritance:
```

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
| `HELION_LLM_PROVIDER` | | LLM provider: `anthropic` or `openai` |
| `HELION_LLM_MODEL` | | Model name (e.g. `claude-opus-4-7`, `gpt-4o`) |
| `HELION_LLM_API_KEY` | | API key (falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `HELION_LLM_API_BASE` | | Custom API base URL |
| `HELION_LLM_COMPILE_TIMEOUT_S` | `15` | Compile timeout (seconds) for LLM-proposed configs |
| `HELION_LLM_CA_BUNDLE` | | Custom CA bundle path (for corporate proxies that do TLS inspection) |
| `HELION_LLM_CLIENT_CERT` | | Client certificate path (for proxies requiring mutual TLS) |
| `HELION_LLM_CLIENT_KEY` | | Client key path (for proxies requiring mutual TLS) |

The proxy/TLS variables are only needed in corporate environments where a proxy intercepts
HTTPS traffic. Most users connecting directly to the LLM API can ignore them.

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
