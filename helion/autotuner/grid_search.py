from __future__ import annotations

import itertools
import logging
import random
from typing import TYPE_CHECKING

from .config_fragment import BaseIntegerFragment
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import ListOf
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .effort_profile import GRID_SEARCH_DEFAULTS
from .finite_search import FiniteSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel

log = logging.getLogger(__name__)


def _enumerate_fragment(spec: ConfigSpecFragment) -> list[object]:
    """Enumerate all possible values for a single config fragment."""
    if isinstance(spec, BooleanFragment):
        return [False, True]

    if isinstance(spec, EnumFragment):
        return list(spec.choices)

    if isinstance(spec, PermutationFragment):
        return [list(p) for p in itertools.permutations(range(spec.length))]

    if isinstance(spec, PowerOfTwoFragment):
        # Includes BlockSizeFragment, NumWarpsFragment
        low_exp = spec.low.bit_length() - 1
        high_exp = spec.high.bit_length() - 1
        return [2**e for e in range(low_exp, high_exp + 1)]

    if isinstance(spec, BaseIntegerFragment):
        # IntegerFragment
        return list(range(spec.low, spec.high + 1))

    if isinstance(spec, ListOf):
        inner_values = _enumerate_fragment(spec.inner)
        return [list(combo) for combo in itertools.product(inner_values, repeat=spec.length)]

    # Fallback: just use the default
    return [spec.default()]


def compute_grid_size(flat_spec: list[ConfigSpecFragment]) -> int:
    """Compute total number of configs in the full grid."""
    total = 1
    for spec in flat_spec:
        total *= len(_enumerate_fragment(spec))
    return total


class GridSearch(FiniteSearch):
    """Exhaustive grid search over the entire configuration space.

    Systematically enumerates all possible configurations by computing the
    Cartesian product of all parameter values. For large search spaces,
    randomly samples a subset controlled by max_configs.

    This is useful when:
    - The search space is small enough for exhaustive evaluation
    - You want deterministic, complete coverage of the config space
    - You suspect heuristic searches are missing the global optimum
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        max_configs: int = GRID_SEARCH_DEFAULTS.max_configs,
        seed: int | None = None,
    ) -> None:
        config_gen = kernel.config_spec.create_config_generation(
            overrides=kernel.settings.autotune_config_overrides or None,
            advanced_controls_files=kernel.settings.autotune_search_acf or None,
        )

        all_values = [_enumerate_fragment(spec) for spec in config_gen.flat_spec]
        grid_size = 1
        for vals in all_values:
            grid_size *= len(vals)

        if grid_size <= max_configs:
            # Exhaustive: enumerate all combinations
            flat_configs = list(itertools.product(*all_values))
            log.info(
                "GridSearch: exhaustive enumeration of %d configs", len(flat_configs)
            )
        else:
            # Sample randomly from the grid
            rng = random.Random(seed)
            indices = rng.sample(range(grid_size), max_configs)
            flat_configs = []
            for idx in indices:
                combo: list[object] = []
                remaining = idx
                for vals in all_values:
                    n = len(vals)
                    combo.append(vals[remaining % n])
                    remaining //= n
                flat_configs.append(tuple(combo))
            log.info(
                "GridSearch: sampled %d / %d configs", max_configs, grid_size
            )

        # Always include the default config at the front
        default_flat = config_gen.default_flat()
        flat_configs.insert(0, tuple(default_flat))

        # Apply shrink_config, convert to Config, and deduplicate
        configs: list[Config] = []
        seen: set[str] = set()
        for flat in flat_configs:
            flat_list = list(flat)
            config_gen.shrink_config(flat_list, 8192)
            config = config_gen.unflatten(flat_list)
            key = config.to_json()
            if key not in seen:
                seen.add(key)
                configs.append(config)

        super().__init__(kernel, args, configs=configs)
        self._grid_size = grid_size
        self._num_configs = len(configs)

        log.info(
            "GridSearch: %d unique configs to benchmark (grid size: %d)",
            self._num_configs,
            self._grid_size,
        )
