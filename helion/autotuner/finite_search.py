from __future__ import annotations

from typing import TYPE_CHECKING

from .. import exc
from .base_search import BaseSearch
from .config_source import ConfigSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel


class FiniteSearch(BaseSearch):
    """Search over a given list of configs, returning the best one.

    This strategy is similar to triton.Autotune, and is the default if you specify `helion.kernel(configs=[...])`.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        configs: Sequence[Config | ConfigSource] | None = None,
    ) -> None:
        super().__init__(kernel, args)
        raw: list[Config | ConfigSource] = [*(configs or ())]
        if len(raw) == 0 and self.kernel.configs:
            raw = [*self.kernel.configs]
        self.configs: list[Config] = self._resolve_sources(raw)
        if len(self.configs) < 2:
            raise exc.NotEnoughConfigs(len(self.configs))

    def _resolve_sources(self, items: list[Config | ConfigSource]) -> list[Config]:
        """Return a list of Configs in order; inline items are kept as-given, source-produced Configs are deduplicated against the running set."""
        result: list[Config] = []
        seen: set[Config] = set()
        for item in items:
            if isinstance(item, ConfigSource):
                for cfg in item.resolve(self):
                    if cfg not in seen:
                        seen.add(cfg)
                        result.append(cfg)
            else:
                seen.add(item)
                result.append(item)
        return result

    def _autotune(self) -> Config:
        best_config = None
        best_time = float("inf")
        for result in self.benchmark_batch(self.configs, desc="Benchmarking"):
            if result.perf < best_time:
                best_time = result.perf
                best_config = result.config
        assert best_config is not None
        return best_config
