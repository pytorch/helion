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
| **Examples** | `print_output_code`, `use_default_config` | `block_sizes`, `num_warps` |
| **When to use** | Development, debugging, environment setup | Performance optimization |

Settings can be configured via:

1. **Environment variables**
2. **Keyword arguments to `@helion.kernel`**
3. **Global defaults via `helion.set_default_settings()`**

## Configuration Examples

### Using Environment Variables

```bash
env HELION_PRINT_OUTPUT_CODE=1  HELION_USE_DEFAULT_CONFIG=1 my_kernel.py
```

### Using Decorator Arguments

```python
import logging
import helion
import helion.language as hl

@helion.kernel(
    use_default_config=True,           # Skip autotuning
    print_output_code=True,            # Debug output
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    for i in hl.grid(x.size(0)):
        result[i] = x[i] * 2
    return result
```

### Global Configuration

```python
import logging
import helion

# Set global defaults
with helion.set_default_settings(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    autotune_log_level=logging.WARNING
):
    # All kernels in this block use these settings
    @helion.kernel
    def kernel1(x): ...

    @helion.kernel
    def kernel2(x): ...
```

## Settings Reference

### Core Compilation Settings

```{eval-rst}
.. currentmodule:: helion

.. autoattribute:: Settings.index_dtype

   The data type used for index variables in generated code. Default is ``torch.int32``.

.. autoattribute:: Settings.dot_precision

   Precision mode for dot product operations. Default is ``"tf32"``. Controlled by ``TRITON_F32_DEFAULT`` environment variable.

.. autoattribute:: Settings.static_shapes

   When enabled, tensor shapes are treated as compile-time constants for optimization. Default is ``False``.
```

### Autotuning Settings

```{eval-rst}
.. autoattribute:: Settings.use_default_config

   Skip autotuning and use default configuration. Default is ``False``. Controlled by ``HELION_USE_DEFAULT_CONFIG=1``.

.. autoattribute:: Settings.force_autotune

   Force autotuning even when explicit configs are provided. Default is ``False``. Controlled by ``HELION_FORCE_AUTOTUNE=1``.

.. autoattribute:: Settings.autotune_log_level

   Controls verbosity of autotuning output using Python logging levels:

   - ``logging.CRITICAL``: No autotuning output
   - ``logging.WARNING``: Only warnings and errors
   - ``logging.INFO``: Standard progress messages (default)
   - ``logging.DEBUG``: Verbose debugging output

   You can also use ``0`` to completely disable all autotuning output.

.. autoattribute:: Settings.autotune_compile_timeout

   Timeout in seconds for Triton compilation during autotuning. Default is ``60``. Controlled by ``HELION_AUTOTUNE_COMPILE_TIMEOUT``.

.. autoattribute:: Settings.autotune_precompile

   Whether to precompile kernels before autotuning. Default is ``True`` on non-Windows systems, ``False`` on Windows.

.. autoattribute:: Settings.autotune_random_seed

   Seed used for autotuner random number generation. Defaults to ``HELION_AUTOTUNE_RANDOM_SEED`` if set, otherwise a time-based value.
```

### Debugging and Development

```{eval-rst}
.. autoattribute:: Settings.print_output_code

   Print generated Triton code to stderr. Default is ``False``. Controlled by ``HELION_PRINT_OUTPUT_CODE=1``.

.. autoattribute:: Settings.ignore_warnings

   List of warning types to suppress during compilation. Default is an empty list.
```

### Advanced Optimization

```{eval-rst}
.. autoattribute:: Settings.allow_warp_specialize

   Allow warp specialization for ``tl.range`` calls. Default is ``True``. Controlled by ``HELION_ALLOW_WARP_SPECIALIZE``.
```

## Functions

```{eval-rst}
.. autofunction:: set_default_settings
```

## See Also

- {doc}`config` - Kernel optimization parameters
- {doc}`exceptions` - Exception handling and debugging
- {doc}`autotuner` - Autotuning configuration
