"""Compatibility imports for population tests; helpers live in helion._testing."""

from __future__ import annotations

from helion._testing.cute_population_contracts import _cpu_target
from helion._testing.cute_population_contracts import _target
from helion._testing.cute_population_contracts import checked_initial_population
from helion._testing.cute_population_contracts import tcgen05_mma_support
from helion._testing.cute_population_contracts import with_flat_min_blocks_default

__all__ = [
    "_cpu_target",
    "_target",
    "checked_initial_population",
    "tcgen05_mma_support",
    "with_flat_min_blocks_default",
]
