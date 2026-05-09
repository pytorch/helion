from __future__ import annotations

from typing import TYPE_CHECKING

from .effort_profile import RANDOM_SEARCH_DEFAULTS
from .finite_search import FiniteSearch
from .observed_heuristics import observed_heuristic_seed_configs_for_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..autotuner.effort_profile import AutotuneEffortProfile
    from ..runtime.config import Config
    from .base_search import _AutotunableKernel
    from helion.runtime.settings import Settings


class RandomSearch(FiniteSearch):
    """
    Implements a random search algorithm for kernel autotuning.

    This class generates a specified number of random configurations
    for a given kernel and evaluates their performance.

    Inherits from:
        FiniteSearch: A base class for finite configuration searches.

    Attributes:
        kernel: The kernel to be tuned (any ``_AutotunableKernel``).
        args: The arguments to be passed to the kernel.
        count: The number of random configurations to generate.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        count: int = RANDOM_SEARCH_DEFAULTS.count,
    ) -> None:
        config_gen = kernel.config_spec.create_config_generation(
            overrides=kernel.settings.autotune_config_overrides or None,
            advanced_controls_files=kernel.settings.autotune_search_acf or None,
            process_group_name=kernel.env.process_group_name,
        )
        random_configs = config_gen.random_population(count)
        seed_configs = observed_heuristic_seed_configs_for_kernel(
            kernel,
            args,
            config_spec=kernel.config_spec,
            max_configs=count,
        )
        configs: list[Config] = []
        seen: set[Config] = set()
        leading_configs = seed_configs or random_configs[:1]
        # Keep the requested population size stable: observed seeds replace the
        # default slot first, and random configs fill any remaining slots.
        for config in [*leading_configs, *random_configs[1:]]:
            if config in seen:
                continue
            seen.add(config)
            configs.append(config)
            if len(configs) >= count:
                break
        attempts = 0
        while len(configs) < count and attempts < 64:
            attempts += 1
            config = config_gen.unflatten(config_gen.random_flat())
            if config in seen:
                continue
            seen.add(config)
            configs.append(config)
        super().__init__(
            kernel,
            args,
            configs=configs,
        )

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        assert profile.random_search is not None
        return {
            "count": profile.random_search.count,
        }
