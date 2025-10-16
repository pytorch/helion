from __future__ import annotations

from typing import TYPE_CHECKING

from .. import exc
from .base_search import BaseSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel


class FiniteSearch(BaseSearch):
    """
    Search over a given list of configs, returning the best one.

    This strategy is similar to triton.Autotune, and is the default if you specify `helion.kernel(configs=[...])`.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        configs: list[Config] | None = None,
    ) -> None:
        super().__init__(kernel, args)
        self.configs: list[Config] = [*(configs or ())]
        if len(self.configs) == 0 and self.kernel.configs:
            self.configs.extend(self.kernel.configs)
        if len(self.configs) < 2:
            raise exc.NotEnoughConfigs(len(self.configs))

    def _autotune(self) -> Config:
        best_config = None
        best_time = float("inf")
        for config, _fn, time, _status in self.parallel_benchmark(
            self.configs, desc="Benchmarking"
        ):
            if time < best_time:
                best_time = time
                best_config = config
        assert best_config is not None
        return best_config
