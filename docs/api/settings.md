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
    print_output_code=True,            # Debug output
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

   The data type used for index variables in generated code. Default is ``torch.int32``.
   Override via ``HELION_INDEX_DTYPE=int64`` (or any ``torch.<dtype>`` name).

.. autoattribute:: Settings.dot_precision

   Precision mode for dot product operations. Default is ``"tf32"``. Controlled by ``TRITON_F32_DEFAULT`` environment variable.

.. autoattribute:: Settings.static_shapes

   When enabled, tensor shapes are treated as compile-time constants for optimization. Default is ``True``.
   Set ``HELION_STATIC_SHAPES=0`` the default if you need a compiled kernel instance to serve many shape variants.
```

### Autotuning Settings

```{eval-rst}
.. autoattribute:: Settings.force_autotune

   Force autotuning even when explicit configs are provided. Default is ``False``. Controlled by ``HELION_FORCE_AUTOTUNE=1``.

.. autoattribute:: Settings.autotune_log_level

   Controls verbosity of autotuning output using Python logging levels:

   - ``logging.CRITICAL``: No autotuning output
   - ``logging.WARNING``: Only warnings and errors
   - ``logging.INFO``: Standard progress messages (default)
   - ``logging.DEBUG``: Verbose debugging output

   You can also use ``0`` to completely disable all autotuning output. Controlled by ``HELION_AUTOTUNE_LOG_LEVEL``.

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

.. autoattribute:: Settings.autotune_ignore_errors

   Continue autotuning even when candidate configurations raise recoverable runtime errors (for example, GPU out-of-memory). Default is ``False``. Controlled by ``HELION_AUTOTUNE_IGNORE_ERRORS``.

.. autoattribute:: Settings.autotune_accuracy_check

   Validate each candidate configuration against a baseline output before accepting it. Default is ``True``. Controlled by ``HELION_AUTOTUNE_ACCURACY_CHECK``.

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
   - ``"quick"`` – limited search for faster runs with decent performance.
   - ``"full"`` – exhaustive autotuning (current default behavior).

   Users can still override individual ``autotune_*`` settings; explicit values win over the preset. Controlled by ``HELION_AUTOTUNE_EFFORT``.


```

### Autotuning Cache

Helion stores the best-performing configs discovered during autotuning in an on-disk cache so subsequent runs can skip the search.

- `HELION_CACHE_DIR`: Override the directory used to store cache entries. Defaults to PyTorch’s `torch._inductor` cache path (typically `/tmp/torchinductor_$USER/helion`).
- `HELION_SKIP_CACHE`: Set to `1` to ignore cached entries and force the autotuner to re-run even if a matching artifact exists.

See :class:`helion.autotuner.LocalAutotuneCache` for details on cache keys and behavior.

### Debugging and Development

```{eval-rst}
.. autoattribute:: Settings.print_output_code

   Print generated Triton code to stderr. Default is ``False``. Controlled by ``HELION_PRINT_OUTPUT_CODE=1``.

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
```

Built-in values for ``HELION_AUTOTUNER`` include ``"PatternSearch"``, ``"DifferentialEvolutionSearch"``, ``"FiniteSearch"``, and ``"RandomSearch"``.

## Functions

```{eval-rst}

```

## Environment Variable Reference

| Environment Variable | Maps To | Description |
|----------------------|---------|-------------|
| ``TRITON_F32_DEFAULT`` | ``dot_precision`` | Sets default floating-point precision for Triton dot products (``"tf32"``, ``"tf32x3"``, ``"ieee"``). |
| ``HELION_INDEX_DTYPE`` | ``index_dtype`` | Choose the default index dtype (accepts any ``torch.<dtype>`` name, e.g. ``int64``). |
| ``HELION_STATIC_SHAPES`` | ``static_shapes`` | Set to ``0``/``false`` to disable global static shape specialization. |
| ``HELION_FORCE_AUTOTUNE`` | ``force_autotune`` | Force the autotuner to run even when explicit configs are provided. |
| ``HELION_DISALLOW_AUTOTUNING`` | ``check_autotuning_disabled`` | Hard-disable autotuning; kernels must supply explicit configs when this is ``1``. |
| ``HELION_AUTOTUNE_COMPILE_TIMEOUT`` | ``autotune_compile_timeout`` | Maximum seconds to wait for Triton compilation during autotuning. |
| ``HELION_AUTOTUNE_LOG_LEVEL`` | ``autotune_log_level`` | Adjust logging verbosity; accepts names like ``INFO`` or numeric levels. |
| ``HELION_AUTOTUNE_PRECOMPILE`` | ``autotune_precompile`` | Select the autotuner precompile mode (``"fork"`` (default), ``"spawn"``, or disable when empty). |
| ``HELION_AUTOTUNE_PRECOMPILE_JOBS`` | ``autotune_precompile_jobs`` | Cap the number of concurrent Triton precompile subprocesses. |
| ``HELION_AUTOTUNE_RANDOM_SEED`` | ``autotune_random_seed`` | Seed used for randomized autotuning searches. |
| ``HELION_AUTOTUNE_MAX_GENERATIONS`` | ``autotune_max_generations`` | Upper bound on generations for Pattern Search and Differential Evolution. |
| ``HELION_AUTOTUNE_ACCURACY_CHECK`` | ``autotune_accuracy_check`` | Toggle baseline validation for candidate configs. |
| ``HELION_AUTOTUNE_EFFORT`` | ``autotune_effort`` | Select autotuning preset (``"none"``, ``"quick"``, ``"full"``). |
| ``HELION_REBENCHMARK_THRESHOLD`` | ``autotune_rebenchmark_threshold`` | Re-run configs whose performance is within a multiplier of the current best. |
| ``HELION_AUTOTUNE_PROGRESS_BAR`` | ``autotune_progress_bar`` | Enable or disable the progress bar UI during autotuning. |
| ``HELION_AUTOTUNE_IGNORE_ERRORS`` | ``autotune_ignore_errors`` | Continue autotuning even when recoverable runtime errors occur. |
| ``HELION_AUTOTUNE_CONFIG_OVERRIDES`` | ``autotune_config_overrides`` | Supply JSON forcing particular autotuner config key/value pairs. |
| ``HELION_CACHE_DIR`` | ``LocalAutotuneCache`` | Override the on-disk directory used for cached autotuning artifacts. |
| ``HELION_SKIP_CACHE`` | ``LocalAutotuneCache`` | When set to ``1``, ignore cached autotuning entries and rerun searches. |
| ``HELION_ASSERT_CACHE_HIT`` | ``AutotuneCacheBase`` | When set to ``1``, require a cache hit; raises ``CacheAssertionError`` on cache miss with detailed diagnostics. |
| ``HELION_PRINT_OUTPUT_CODE`` | ``print_output_code`` | Print generated Triton code to stderr for inspection. |
| ``HELION_OUTPUT_ORIGIN_LINES`` | ``output_origin_lines`` | Include ``# src[...]`` comments in generated Triton code; set to ``0`` to disable. |
| ``HELION_IGNORE_WARNINGS`` | ``ignore_warnings`` | Comma-separated warning names defined in ``helion.exc`` to suppress. |
| ``HELION_ALLOW_WARP_SPECIALIZE`` | ``allow_warp_specialize`` | Permit warp-specialized code generation for ``tl.range``. |
| ``HELION_DEBUG_DTYPE_ASSERTS`` | ``debug_dtype_asserts`` | Inject dtype assertions after each lowering step. |
| ``HELION_INTERPRET`` | ``ref_mode`` | Run kernels through the reference interpreter when set to ``1`` (maps to ``RefMode.EAGER``). |
| ``HELION_AUTOTUNER`` | ``default_autotuner_fn`` | Select which autotuner implementation to instantiate (``"PatternSearch"``, ``"DifferentialEvolutionSearch"``, ``"FiniteSearch"``, ``"RandomSearch"``). |

## See Also

- {doc}`config` - Kernel optimization parameters
- {doc}`exceptions` - Exception handling and debugging
- {doc}`autotuner` - Autotuning configuration
