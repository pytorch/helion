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
   load
   store
   atomic_add
   signal
   wait
   stacktensor_like
   zeros
   full
   arange
   reduce
   associative_scan
   register_block_size
   register_reduction_dim
   register_tunable
   constexpr
   specialize
```
