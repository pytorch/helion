from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .. import exc
from .base_search import BaseSearch

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel


class FiniteSearch(BaseSearch):
    """
    Search over a given list of configs, returning the best one.

    This strategy is similar to triton.Autotune, and is the default if you specify `helion.kernel(configs=[...])`.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        configs: list[Config] | None = None,
    ) -> None:
        super().__init__(kernel, args)
        self.configs: list[Config] = [*(configs or ())]
        if len(self.configs) == 0 and self.kernel.configs:
            self.configs.extend(self.kernel.configs)
        seen: set[Config] = set()
        unique_configs: list[Config] = []
        for config in self.configs:
            if config not in seen:
                seen.add(config)
                unique_configs.append(config)
        num_dupes = len(self.configs) - len(unique_configs)
        if num_dupes > 0:
            log.info(
                "Removed %d duplicate configs from %d total",
                num_dupes,
                len(self.configs),
            )
        self.configs = unique_configs
        if len(self.configs) < 2:
            raise exc.NotEnoughConfigs(len(self.configs))

    def _autotune(self) -> Config:
        best_config = None
        best_time = float("inf")
        for result in self.parallel_benchmark(self.configs, desc="Benchmarking"):
            if result.perf < best_time:
                best_time = result.perf
                best_config = result.config
        assert best_config is not None
        return best_config
