# Settings

The `Settings` class controls compilation behavior and debugging options for Helion kernels.

```{eval-rst}
.. currentmodule:: helion

.. autoclass:: Settings
   :members:
   :show-inheritance:
```

## Overview

**Settings** control the **compilation process** and **development environment** for Helion kernels.

### Key Characteristics

- **Not autotuned**: Settings remain constant across all kernel configurations
- **Meta-compilation**: Control the compilation process itself, debugging output, and development features
- **Environment-driven**: Often configured via environment variables
- **Development-focused**: Primarily used for debugging, logging, and development workflow optimization

### Settings vs Config

| Aspect | Settings | Config |
|--------|----------|--------|
| **Purpose** | Control compilation behavior | Control execution performance |
| **Autotuning** | ❌ Never autotuned | ✅ Automatically optimized |
| **Examples** | `print_output_code`, `autotune_effort` | `block_sizes`, `num_warps` |
| **When to use** | Development, debugging, environment setup | Performance optimization |

Settings can be configured via:

1. **Environment variables**
2. **Keyword arguments to `@helion.kernel`**

If both are provided, decorator arguments take precedence.

```{note}
Helion reads the environment variables for `Settings` when the
`@helion.kernel` decorator defines the function (typically at import
time). One can modify Kernel.settings to change settings
for an already defined kernel.
```

## Configuration Examples

### Using Environment Variables

```bash
env HELION_PRINT_OUTPUT_CODE=1  HELION_AUTOTUNE_EFFORT=none my_kernel.py
```

### Using Decorator Arguments

```python
import logging
import helion
import helion.language as hl

@helion.kernel(
    autotune_effort="none",           # Skip autotuning
    print_output_code=True,            # Debug: show generated Triton code
    print_repro=True,                  # Debug: show Helion kernel code, config, and caller code as a standalone repro script
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    for i in hl.grid(x.size(0)):
        result[i] = x[i] * 2
    return result
```


## Settings Reference

### Core Compilation Settings

```{eval-rst}
.. currentmodule:: helion

.. autoattribute:: Settings.index_dtype

   The data type used for index variables in generated code. By default Helion auto-selects
   between ``torch.int32`` and ``torch.int64`` based on whether any input tensor exceeds
   ``torch.iinfo(torch.int32).max`` elements. Override via ``HELION_INDEX_DTYPE=<dtype>``
   (or set it to ``auto`` to keep the automatic behavior).

.. autoattribute:: Settings.dot_precision

   Precision mode for dot product operations.
   - For Triton backend, this is initialized from the ``TRITON_F32_DEFAULT`` environment variable (defaulting to ``"tf32"``).
   - For Pallas backend, this is initialized from the ``JAX_DEFAULT_MATMUL_PRECISION`` environment variable (defaulting to ``"default"``).

   Supported values depend on the backend:
   - Triton (GPU): ``"tf32"``, ``"tf32x3"``, ``"ieee"``
   - Pallas (TPU): ``"default"`` (forces JAX default, typically bfloat16), ``"high"``, ``"highest"`` (float32 emulation).

   However, unified mappings exist so you can use any value on any backend. They map to the closest equivalent of equal or higher precision:
   - On Triton: ``"default"`` maps to ``"tf32"``, ``"high"`` maps to ``"tf32x3"``, and ``"highest"`` maps to ``"ieee"``.
   - On Pallas: ``"tf32"``, ``"tf32x3"`` and ``"ieee"`` all map to ``"highest"``.

.. autoattribute:: Settings.static_shapes

   When enabled, tensor shapes are treated as compile-time constants for optimization. Default is ``True``.
   Set ``HELION_STATIC_SHAPES=0`` the default if you need a compiled kernel instance to serve many shape variants.

.. autoattribute:: Settings.fast_math

   If ``True``, enable fast math approximations. This activates both Helion-level optimizations
   (e.g. fast sigmoid) and Inductor-level fast math (flush-to-zero exp, fast online softmax,
   etc.). May reduce numerical precision. Default is ``False``. Controlled by ``HELION_FAST_MATH=1``.

.. autoattribute:: Settings.persistent_reserved_sms

   Reserve this many streaming multiprocessors when launching persistent kernels. Default is ``0`` (use all SMs).
   Configure globally with ``HELION_PERSISTENT_RESERVED_SMS`` or per-kernel via ``@helion.kernel(..., persistent_reserved_sms=N)``.

.. autoattribute:: Settings.triton_direct_launch

   If ``True`` (default), repeat launches of an already-compiled Triton specialization skip
   ``JITFunction.run`` and invoke the cached compiled kernel directly, substantially reducing
   per-call launch overhead. Triton-only; the setting is accepted but ignored on other backends.
   Disable per-kernel with ``@helion.kernel(triton_direct_launch=False)`` or globally with
   ``HELION_TRITON_DIRECT_LAUNCH=0``.

   The direct path automatically falls back to the full ``JITFunction.run`` path whenever it
   cannot guarantee an identical launch: unaligned (non-16-byte) tensor pointer arguments,
   mutated ``used_global_vals`` (globals captured by the generated kernel), extra launch kwargs
   such as ``ptx_options``, Triton launch/pre-run hooks or debug/instrumentation knobs active
   when the kernel is first launched, and ``torch.compile`` tracing all take the slow path.

   One case is *not* auto-detected and requires opting out: Triton launch/pre-run hooks or
   knobs registered *after* a kernel's first launch are not re-checked on each call, so a
   profiler that attaches launch hooks mid-run will not observe direct launches. Set
   ``triton_direct_launch=False`` for kernels that must always be visible to such hooks.

.. autoattribute:: Settings.triton_fused_launch

   If ``True`` (default), goes one step further than ``triton_direct_launch``: after one
   priming launch, the argument-dispatch, generated host-wrapper, and launcher layers are
   collapsed into a single cached closure, so repeat calls launch the compiled kernel directly
   from ``Kernel.__call__`` -- roughly halving the remaining per-call host overhead. Builds on
   and shares the safety model of ``triton_direct_launch`` (per-tensor 16-byte alignment and
   ``used_global_vals`` values are folded into the cache key; any deviation, and ``torch.compile``
   tracing, fall back to the full path). Requires ``triton_direct_launch``; Triton-only.

   Output-allocating kernels are supported: because the generated wrapper is codegen'd over fake
   tensors, every host-side allocation is a pure function of input metadata and scalars, so the
   closure rebuilds each allocated output with ``torch.empty_strided`` and reproduces the
   wrapper's return value (a tensor, or a tuple/list). Fusion is automatically disabled for a
   kernel that issues multiple device launches, returns a view over storage it shares with
   another tensor (unreconstructable blind), or takes cooperative-grid launches. Disable
   per-kernel with ``@helion.kernel(triton_fused_launch=False)`` or globally with
   ``HELION_TRITON_FUSED_LAUNCH=0``.
```

### Autotuning Settings

```{eval-rst}
.. autoattribute:: Settings.force_autotune

   Force autotuning even when explicit configs are provided. Default is ``False``. Controlled by ``HELION_FORCE_AUTOTUNE=1``.
   The result is still saved to the cache so subsequent runs can reuse it. Use ``HELION_SKIP_CACHE=1`` instead to skip both reading and writing the cache.

.. autoattribute:: Settings.autotune_force_persistent

   Restrict ``pid_type`` choices to the persistent strategies (``"persistent_blocked"`` or ``"persistent_interleaved"``).
   Default is ``False``. Enable globally with ``HELION_AUTOTUNE_FORCE_PERSISTENT=1`` or per kernel via ``@helion.kernel(..., autotune_force_persistent=True)``.

.. autoattribute:: Settings.autotune_log_level

   Controls verbosity of autotuning output using Python logging levels:

   - ``logging.CRITICAL``: No autotuning output
   - ``logging.WARNING``: Only warnings and errors
   - ``logging.INFO``: Standard progress messages (default)
   - ``logging.DEBUG``: Verbose debugging output

   You can also use ``0`` to completely disable all autotuning output. Controlled by ``HELION_AUTOTUNE_LOG_LEVEL``.

.. autoattribute:: Settings.autotune_log

   When set, Helion writes per-config telemetry (run id, timestamp, config id, generation, status, perf, compile time, config) to ``<value>.csv`` and mirrors the autotune log to ``<value>.log`` (for population-based autotuners: ``PatternSearch``, ``DifferentialEvolution``). Both append, so runs sharing one base path accumulate.
   CSV rows join to ``.meta.jsonl`` via two hashes: ``run_id`` for the invocation — hash(kernel source, shapes, dtypes, hardware, codegen settings) — and ``config_id`` for the config. The codegen settings hashed into ``run_id`` are: backend, dot_precision, fast_math, static_shapes, index_dtype, allow_warp_specialize, triton_do_not_specialize, pallas_interpret, debug_dtype_asserts, persistent_reserved_sms (the full settings are recorded in ``.meta.jsonl``). Same config_id across started/ok/error and re-benchmarks. The full config is written inline in the trailing ``config`` column of the CSV; it is also stored in the ``.meta.jsonl`` configs map, which is written only when ``autotune_log_details`` is enabled (see below). Controlled by HELION_AUTOTUNE_LOG.



.. autoattribute:: Settings.autotune_log_details

   Opt-in to the cost-model dataset sidecar. When enabled (``HELION_AUTOTUNE_LOG_DETAILS=1``) and ``autotune_log`` is set, Helion appends one JSON record per run to ``<autotune_log>.meta.jsonl``: the kernel identity (``run_id``, name, source, shapes, dtypes, hardware), the full ``helion.settings`` (JSON-safe via ``json.dumps(default=str)``, so ``torch.dtype``/callables become strings), and a ``configs`` map from ``config_id`` to the config tested.
   Recover a measured ``(config, perf)`` sample by joining a CSV row to its record: ``meta[run_id]["configs"][row["config_id"]]``. ``run_id`` may recur (re-runs, processes, ``autotune_best_of_k``), but the ``configs`` maps are union-safe (same ``config_id`` implies the same config), so de-duplicating on ``run_id`` is lossless. Searches restricted to user-pinned ``configs`` (without ``force_autotune``) are excluded as a biased slice (``.csv``/``.log`` still written); setting this without ``autotune_log`` collects nothing and warns once.
   Controlled by ``HELION_AUTOTUNE_LOG_DETAILS``.

.. autoattribute:: Settings.autotune_compile_timeout

   Timeout in seconds for Triton compilation during autotuning. Default is ``60``. Controlled by ``HELION_AUTOTUNE_COMPILE_TIMEOUT``.

.. autoattribute:: Settings.autotune_precompile

   Select the autotuner precompile mode, which adds parallelism and
   checks for errors/timeouts. ``"fork"`` (default) is faster but does
   not include the error check run, ``"spawn"`` runs kernel warm-up in a
   fresh process including running to check for errors, or None to
   disables precompile checks altogether. Controlled by
   ``HELION_AUTOTUNE_PRECOMPILE``.

.. autoattribute:: Settings.autotune_random_seed

   Seed used for autotuner random number generation. Defaults to ``HELION_AUTOTUNE_RANDOM_SEED`` if set, otherwise a time-based value.

.. autoattribute:: Settings.autotune_precompile_jobs

   Cap the number of concurrent Triton precompile subprocesses. ``None`` (default) uses the machine CPU count.
   Controlled by ``HELION_AUTOTUNE_PRECOMPILE_JOBS``.
   When using ``"spawn"`` precompile mode, Helion may automatically lower this cap if free GPU memory is limited.

.. autoattribute:: Settings.autotune_max_generations

   Override the default number of generations set for Pattern Search and Differential Evolution Search autotuning algorithms with HELION_AUTOTUNE_MAX_GENERATIONS=N or @helion.kernel(autotune_max_generations=N).

   Lower values result in faster autotuning but may find less optimal configurations.

.. autoattribute:: Settings.autotune_budget_seconds

   Wall-clock budget in seconds for autotuning. When the budget is exceeded, Helion returns the best configuration found so far. Controlled by ``HELION_AUTOTUNE_BUDGET_SECONDS``.

.. autoattribute:: Settings.autotune_ignore_errors

   Continue autotuning even when candidate configurations raise recoverable runtime errors (for example, GPU out-of-memory). Default is ``False``. Controlled by ``HELION_AUTOTUNE_IGNORE_ERRORS``.

.. autoattribute:: Settings.autotune_accuracy_check

   Validate each candidate configuration against a baseline output before accepting it. Default is ``True``. Controlled by ``HELION_AUTOTUNE_ACCURACY_CHECK``.

.. autoattribute:: Settings.autotune_baseline_atol

   Absolute tolerance for baseline output comparison during autotune accuracy checks. Default is ``1e-2``.

.. autoattribute:: Settings.autotune_baseline_rtol

   Relative tolerance for baseline output comparison during autotune accuracy checks. Default is ``1e-2``.

.. autoattribute:: Settings.autotune_search_acf

   List of PTXAS config file paths to search during autotuning. Empty list (default) disables the feature. An empty string entry represents not passing a config. Controlled by ``HELION_AUTOTUNE_SEARCH_ACF`` (comma-separated paths).

.. autoattribute:: Settings.autotune_rebenchmark_threshold

   Controls how aggressively Helion re-runs promising configs to avoid outliers. Default is ``1.5`` (re-benchmark anything within 1.5x of the best).

.. autoattribute:: Settings.autotune_progress_bar

   Toggle the interactive progress bar during autotuning. Default is ``True``. Controlled by ``HELION_AUTOTUNE_PROGRESS_BAR``.

.. autoattribute:: Settings.autotune_config_overrides

   Dict of config key/value pairs to force during autotuning. Useful for disabling problematic candidates or pinning experimental options.
   Provide JSON via ``HELION_AUTOTUNE_CONFIG_OVERRIDES='{"num_warps": 4}'`` for global overrides.

.. autoattribute:: Settings.autotune_effort

   Select the autotuning effort preset. Available values:

   - ``"none"`` – skip autotuning and run the default configuration.
   - ``"quick"`` – limited search for faster runs with decent performance. Uses ``from_best_available`` initial population strategy (no random padding).
   - ``"full"`` – exhaustive autotuning (current default behavior). Uses ``from_random`` initial population strategy.

   Each preset also sets a default initial population strategy (see :doc:`../deployment_autotuning` for details).
   Users can still override individual ``autotune_*`` settings; explicit values win over the preset. Controlled by ``HELION_AUTOTUNE_EFFORT``.

.. autoattribute:: Settings.autotune_best_available_max_configs

   Maximum number of cached configs to use when seeding the initial population with the ``from_best_available`` strategy.
   Default is ``20``. Controlled by ``HELION_BEST_AVAILABLE_MAX_CONFIGS``.

.. autoattribute:: Settings.autotune_best_available_max_cache_scan

   Maximum number of cache files to scan when searching for matching configs in the ``from_best_available`` strategy.
   Default is ``500``. Controlled by ``HELION_BEST_AVAILABLE_MAX_CACHE_SCAN``.

```

### Autotuning Cache

Helion stores the best-performing configs discovered during autotuning in an on-disk cache so subsequent runs can skip the search.

- `HELION_CACHE_DIR`: Override the directory used to store cache entries. Defaults to PyTorch’s `torch._inductor` cache path (typically `/tmp/torchinductor_$USER/helion`).
- `HELION_SKIP_CACHE`: Set to `1` to skip both reading and writing the autotuning cache. Useful for one-off experiments that should not affect cached state. To re-tune and save the result, use `HELION_FORCE_AUTOTUNE=1` instead.
- `TRITON_STORE_BINARY_ONLY`: During autotuning, Helion sets this Triton environment variable to `1` by default, skipping storage of intermediate representations (`.ttir`, `.ttgir`, `.llir`, etc.) and keeping only compiled binaries and metadata. This reduces Triton cache disk usage by approximately 40%. To retain IRs for debugging, set `TRITON_STORE_BINARY_ONLY=0` before running.
- `HELION_KEEP_CACHE`: Set to `1` to keep the backend compile-cache entries for all candidate configs evaluated during autotuning. By default, Helion uses an ephemeral cache directory during autotuning (`TRITON_CACHE_DIR` for the Triton backend, `CUTE_DSL_CACHE_DIR` for the CuTe backend) and only preserves the winning config's cache entry, avoiding significant disk bloat. Enable this if you need to inspect the compiled artifacts of non-winning configs for debugging. (`HELION_KEEP_TRITON_CACHE` is a deprecated alias that still works for the Triton backend.)

See :class:`helion.autotuner.LocalAutotuneCache` for details on cache keys and behavior.

### Debugging and Development

```{eval-rst}
.. autoattribute:: Settings.print_output_code

   Print generated Triton code to stderr. Default is ``False``. Controlled by ``HELION_PRINT_OUTPUT_CODE=1``.

.. autoattribute:: Settings.print_repro

   Print Helion kernel code, config, and caller code to stderr as a standalone repro script. Default is ``False``. Controlled by ``HELION_PRINT_REPRO=1``.

.. autoattribute:: Settings.output_origin_lines

   Annotate generated Triton code with ``# src[<file>:<line>]`` comments indicating the originating Helion statements.
   Default is ``True``. Controlled by ``HELION_OUTPUT_ORIGIN_LINES`` (set to ``0`` to disable).

.. autoattribute:: Settings.ignore_warnings

   List of warning types to suppress during compilation. Default is an empty list.
   Accepts comma-separated warning class names from ``helion.exc`` via ``HELION_IGNORE_WARNINGS`` (for example, ``HELION_IGNORE_WARNINGS=TensorOperationInWrapper``).

.. autoattribute:: Settings.debug_dtype_asserts

   Emit ``tl.static_assert`` dtype checks after each lowering step. Default is ``False``. Controlled by ``HELION_DEBUG_DTYPE_ASSERTS``.
```

### Device Execution Modes

```{eval-rst}
.. autoattribute:: Settings.allow_warp_specialize

   Allow warp specialization for ``tl.range`` calls. Default is ``True``. Controlled by ``HELION_ALLOW_WARP_SPECIALIZE``.

.. autoattribute:: Settings.ref_mode

   Select the reference execution strategy. ``RefMode.OFF`` runs compiled kernels (default); ``RefMode.EAGER`` runs the interpreter for debugging. Controlled by ``HELION_INTERPRET``.
```

### Autotuner Hooks

```{eval-rst}
.. autoattribute:: Settings.autotuner_fn

   Override the callable that constructs autotuner instances. Accepts the same signature as :func:`helion.runtime.settings.default_autotuner_fn`.
   Pass a replacement callable via ``@helion.kernel(..., autotuner_fn=...)`` or ``helion.kernel(autotuner_fn=...)`` at definition time.

.. autoattribute:: Settings.autotune_benchmark_fn

   Custom benchmark function for rebenchmarking during autotuning. Should have the signature
   ``(fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None) -> list[float]``.
   If ``None`` (default), uses the built-in benchmark function.
   Pass a replacement callable via ``@helion.kernel(..., autotune_benchmark_fn=...)`` at definition time.
```

Built-in values for ``HELION_AUTOTUNER`` include ``"LFBOTreeSearch"`` (default), ``"LFBOPatternSearch"``, ``"DESurrogateHybrid"``, ``"PatternSearch"``, ``"DifferentialEvolutionSearch"``, ``"FiniteSearch"``, and ``"RandomSearch"``.

## Environment Variable Reference

| Environment Variable | Maps To | Description |
|----------------------|---------|-------------|
| ``TRITON_F32_DEFAULT`` | ``dot_precision`` | Sets default floating-point precision for Triton dot products (``"tf32"``, ``"tf32x3"``, ``"ieee"``). This variable does not apply to the Pallas backend. |
| ``JAX_DEFAULT_MATMUL_PRECISION`` | ``dot_precision`` | Sets default matmul precision for Pallas dot products (JAX values: ``"default"``, ``"high"``, ``"highest"``, etc., mapped to Helion ``DotPrecision``). This variable does not apply to the Triton backend. |
| ``HELION_INDEX_DTYPE`` | ``index_dtype`` | Choose the index dtype (accepts any ``torch.<dtype>`` name, e.g. ``int64``), or set to ``auto``/unset to allow Helion to pick ``int32`` vs ``int64`` based on input sizes. |
| ``HELION_STATIC_SHAPES`` | ``static_shapes`` | Set to ``0``/``false`` to disable global static shape specialization. |
| ``HELION_FAST_MATH`` | ``fast_math`` | Set to ``1`` to enable fast math approximations (Helion-level and Inductor-level). May reduce numerical precision. |
| ``HELION_PERSISTENT_RESERVED_SMS`` | ``persistent_reserved_sms`` | Reserve this many streaming multiprocessors when launching persistent kernels (``0`` uses all available SMs). |
| ``HELION_TRITON_DIRECT_LAUNCH`` | ``triton_direct_launch`` | Set to ``0`` to disable the Triton direct-launch fast path and route every launch through ``JITFunction.run``. |
| ``HELION_TRITON_FUSED_LAUNCH`` | ``triton_fused_launch`` | Set to ``0`` to disable the fused launch fast path (dispatch + host wrapper + launcher collapsed into one cached closure) and keep only direct launch. |
| ``HELION_FORCE_AUTOTUNE`` | ``force_autotune`` | Force the autotuner to run even when explicit configs are provided. The result is saved to the cache. |
| ``HELION_AUTOTUNE_FORCE_PERSISTENT`` | ``autotune_force_persistent`` | Restrict ``pid_type`` to persistent kernel strategies during config search. |
| ``HELION_DISALLOW_AUTOTUNING`` | ``check_autotuning_disabled`` | Hard-disable autotuning; kernels must supply explicit configs when this is ``1``. |
| ``HELION_AUTOTUNE_COMPILE_TIMEOUT`` | ``autotune_compile_timeout`` | Maximum seconds to wait for Triton compilation during autotuning. |
| ``HELION_AUTOTUNE_LOG_LEVEL`` | ``autotune_log_level`` | Adjust logging verbosity; accepts names like ``INFO`` or numeric levels. |
| ``HELION_AUTOTUNE_LOG`` | ``autotune_log`` | Base filename for per-config CSV telemetry and mirrored autotune logs. |
| ``HELION_AUTOTUNE_PRECOMPILE`` | ``autotune_precompile`` | Select the autotuner precompile mode (``"fork"`` (default), ``"spawn"``, or disable when empty). |
| ``HELION_AUTOTUNE_PRECOMPILE_JOBS`` | ``autotune_precompile_jobs`` | Cap the number of concurrent Triton precompile subprocesses. |
| ``HELION_AUTOTUNE_RANDOM_SEED`` | ``autotune_random_seed`` | Seed used for randomized autotuning searches. |
| ``HELION_AUTOTUNE_MAX_GENERATIONS`` | ``autotune_max_generations`` | Upper bound on generations for Pattern Search and Differential Evolution. |
| ``HELION_AUTOTUNE_BUDGET_SECONDS`` | ``autotune_budget_seconds`` | Wall-clock budget for an autotune run. |
| ``HELION_AUTOTUNE_ACCURACY_CHECK`` | ``autotune_accuracy_check`` | Toggle baseline validation for candidate configs. |
| ``HELION_AUTOTUNE_EFFORT`` | ``autotune_effort`` | Select autotuning preset (``"none"``, ``"quick"``, ``"full"``). |
| ``HELION_AUTOTUNE_SEARCH_ACF`` | ``autotune_search_acf`` | Comma-separated list of PTXAS config file paths to search during autotuning. |
| ``HELION_AUTOTUNER_INITIAL_POPULATION`` | (effort profile) | Override the initial population strategy (``"from_random"``, ``"from_best_available"``). |
| ``HELION_BEST_AVAILABLE_MAX_CONFIGS`` | ``autotune_best_available_max_configs`` | Maximum cached configs to seed when using ``from_best_available`` strategy. |
| ``HELION_BEST_AVAILABLE_MAX_CACHE_SCAN`` | ``autotune_best_available_max_cache_scan`` | Maximum cache files to scan when using ``from_best_available`` strategy. |
| ``HELION_REBENCHMARK_THRESHOLD`` | ``autotune_rebenchmark_threshold`` | Re-run configs whose performance is within a multiplier of the current best. |
| ``HELION_AUTOTUNE_PROGRESS_BAR`` | ``autotune_progress_bar`` | Enable or disable the progress bar UI during autotuning. |
| ``HELION_AUTOTUNE_IGNORE_ERRORS`` | ``autotune_ignore_errors`` | Continue autotuning even when recoverable runtime errors occur. |
| ``HELION_AUTOTUNE_CONFIG_OVERRIDES`` | ``autotune_config_overrides`` | Supply JSON forcing particular autotuner config key/value pairs. |
| ``TRITON_STORE_BINARY_ONLY`` | Triton (autotuning) | Set to ``1`` during autotuning to skip Triton intermediate IRs, reducing cache size ~40%. Set to ``0`` to retain IRs for debugging. |
| ``HELION_CACHE_DIR`` | ``LocalAutotuneCache`` | Override the on-disk directory used for cached autotuning artifacts. |
| ``HELION_SKIP_CACHE`` | ``LocalAutotuneCache`` | When set to ``1``, skip both reading and writing the autotuning cache entirely. |
| ``HELION_ASSERT_CACHE_HIT`` | ``AutotuneCacheBase`` | When set to ``1``, require a cache hit; raises ``CacheAssertionError`` on cache miss with detailed diagnostics. |
| ``HELION_AUTOTUNE_CACHE`` | ``autotune_cache`` | Cache class to use (``"LocalAutotuneCache"`` (default), ``"StrictLocalAutotuneCache"``, ``"RemoteAutotuneCache"``, ``"StrictRemoteAutotuneCache"``, ``"AOTAutotuneCache"``). |
| ``HELION_REMOTE_CACHE_BACKEND`` | (used by ``RemoteAutotuneCache`` and warm-start) | Fully-qualified class path to a ``RemoteCacheBackend`` subclass (e.g. ``mypackage.cache.RedisBackend``); enables remote read-through/write-through caching and, when the backend overrides ``list()``, remote warm-start lookups for ``from_best_available`` / ``helion.from_cache``. |
| ``HELION_BENCHMARK_CUDAGRAPH`` | (benchmarking) | Wrap ``run_example``'s timing loop in CUDA-graph capture/replay so launch overhead is amortized for both helion and the torch baseline. Defaults on inside ``run_example``, off elsewhere. Set ``0`` to opt out; ``1`` to enable globally. Capture silently falls back to non-CG if unavailable (HIP, no CUDA, nested capture). Note: ``=1`` globally also applies during autotune, which is counterproductive — each of thousands of configs triggers a fresh capture (warmup + sync per config, no reuse across configs since launch parameters differ), captured graphs accumulate memory pool allocations, and many configs raise on capture due to shape/stride differences across the search space. Prefer leaving unset (default off in autotune, on in ``run_example``). |
| ``HELION_PRINT_OUTPUT_CODE`` | ``print_output_code`` | Print generated Triton code to stderr for inspection. |
| ``HELION_PRINT_REPRO`` | ``print_repro`` | Print Helion kernel code, config, and caller code to stderr as a standalone repro script. |
| ``HELION_OUTPUT_ORIGIN_LINES`` | ``output_origin_lines`` | Include ``# src[...]`` comments in generated Triton code; set to ``0`` to disable. |
| ``HELION_IGNORE_WARNINGS`` | ``ignore_warnings`` | Comma-separated warning names defined in ``helion.exc`` to suppress. |
| ``HELION_ALLOW_WARP_SPECIALIZE`` | ``allow_warp_specialize`` | Permit warp-specialized code generation for ``tl.range``. |
| ``HELION_DEBUG_DTYPE_ASSERTS`` | ``debug_dtype_asserts`` | Inject dtype assertions after each lowering step. |
| ``HELION_INTERPRET`` | ``ref_mode`` | Run kernels through the reference interpreter when set to ``1`` (maps to ``RefMode.EAGER``). |
| ``HELION_AUTOTUNER`` | ``default_autotuner_fn`` | Select which autotuner implementation to instantiate. Default is ``"LFBOTreeSearch"``. Other options: ``"LFBOPatternSearch"``, ``"DESurrogateHybrid"``, ``"PatternSearch"``, ``"DifferentialEvolutionSearch"``, ``"FiniteSearch"``, ``"RandomSearch"``, ``"LLMGuidedSearch"``, ``"LLMSeededSearch"``, ``"LLMSeededLFBOTreeSearch"``. See :doc:`autotuner` for LLM configuration. |
| ``HELION_BACKEND`` | ``backend`` | Code generation backend (``"triton"`` (default), ``"pallas"``, ``"cute"``, ``"tileir"``). ``"cute"`` requires PyTorch built against CUDA 13 or later. |

## See Also

- {doc}`config` - Kernel optimization parameters
- {doc}`exceptions` - Exception handling and debugging
- {doc}`autotuner` - Autotuning configuration
