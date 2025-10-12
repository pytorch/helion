from __future__ import annotations

from typing import TYPE_CHECKING

from .config_generation import ConfigGeneration
from .effort_profile import RANDOM_SEARCH_DEFAULTS
from .finite_search import FiniteSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel


class RandomSearch(FiniteSearch):
    """
    Implements a random search algorithm for kernel autotuning.

    This class generates a specified number of random configurations
    for a given kernel and evaluates their performance.

    Inherits from:
        FiniteSearch: A base class for finite configuration searches.

    Attributes:
        kernel (BoundKernel): The kernel to be tuned.
        args (Sequence[object]): The arguments to be passed to the kernel.
        count (int): The number of random configurations to generate.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        count: int = RANDOM_SEARCH_DEFAULTS.count,
    ) -> None:
        super().__init__(
            kernel,
            args,
            configs=ConfigGeneration(
                kernel.config_spec,
                overrides=kernel.settings.autotune_config_overrides or None,
            ).random_population(count),
        )
