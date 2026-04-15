from __future__ import annotations

from typing import TYPE_CHECKING

from .effort_profile import RANDOM_SEARCH_DEFAULTS
from .finite_search import FiniteSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..autotuner.effort_profile import AutotuneEffortProfile
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
        super().__init__(
            kernel,
            args,
            configs=kernel.config_spec.create_config_generation(
                overrides=kernel.settings.autotune_config_overrides or None,
                advanced_controls_files=kernel.settings.autotune_search_acf or None,
                process_group_name=kernel.env.process_group_name,
            ).random_population(count),
        )

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        assert profile.random_search is not None
        return {
            "count": profile.random_search.count,
        }
