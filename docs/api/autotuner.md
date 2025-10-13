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
