from __future__ import annotations

from typing import TYPE_CHECKING

from .effort_profile import RANDOM_SEARCH_DEFAULTS
from .finite_search import FiniteSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base_search import _AutotunableKernel


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
                advanced_compiler_configurations=kernel.settings.autotune_search_acc
                or None,
            ).random_population(count),
        )
