# API Reference

Complete API documentation for Helion.

## Kernel Creation and Control

Everything you need to create and configure Helion kernels using the {func}`helion.kernel` decorator:

```{toctree}
:maxdepth: 2

kernel
config
settings
```

## Language Constructs

The `helion.language` module contains DSL constructs for authoring kernels:

```{toctree}
:maxdepth: 2

language
```

## Debugging and Utilities

```{toctree}
:maxdepth: 2

exceptions
```

## Advanced Topics

```{toctree}
:maxdepth: 2

autotuner
runtime
```

## Quick Reference

### Main Functions

```{eval-rst}
.. currentmodule:: helion

.. autosummary::
   :toctree: generated/
   :nosignatures:

   kernel
   Config
   Settings
```

### Language Functions

```{eval-rst}
.. currentmodule:: helion.language

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tile
   grid
   static_range
   load
   store
   atomic_add
   atomic_and
   atomic_or
   atomic_xor
   atomic_xchg
   atomic_max
   atomic_min
   atomic_cas
   device_print
   signal
   wait
   stacktensor_like
   zeros
   full
   arange
   subscript
   split
   join
   reduce
   associative_scan
   cumsum
   cumprod
   dot
   inline_asm_elementwise
   inline_triton
   register_block_size
   register_tunable
   constexpr
   specialize
```

### Language Classes

```{eval-rst}
.. currentmodule:: helion.language

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Tile
   StackTensor
```

### Tile Helpers

```{eval-rst}
.. currentmodule:: helion.language

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tile_index
   tile_begin
   tile_end
   tile_block_size
   tile_id
```
