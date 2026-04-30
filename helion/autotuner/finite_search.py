from __future__ import annotations

from typing import TYPE_CHECKING

from .. import exc
from .base_search import BaseSearch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .base_search import _AutotunableKernel
    from .config_generation import ConfigGeneration


class FiniteSearch(BaseSearch):
    """Search over a given list of configs, returning the best one.

    This strategy is similar to triton.Autotune, and is the default if you specify `helion.kernel(configs=[...])`.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        configs: Sequence[Config] | None = None,
    ) -> None:
        super().__init__(kernel, args)
        self.config_gen: ConfigGeneration = self.config_spec.create_config_generation(
            overrides=self.settings.autotune_config_overrides or None,
            advanced_controls_files=self.settings.autotune_search_acf or None,
            process_group_name=kernel.env.process_group_name,
        )
        raw: list[Config] = list(configs if configs is not None else kernel.configs)
        self.configs: list[Config] = raw
        if len(self.configs) < 2:
            raise exc.NotEnoughConfigs(len(self.configs))

    def _autotune(self) -> Config:
        best_config = None
        best_time = float("inf")
        for result in self.benchmark_batch(self.configs, desc="Benchmarking"):
            if result.perf < best_time:
                best_time = result.perf
                best_config = result.config
        assert best_config is not None
        return best_config


class CachedFiniteSearch(FiniteSearch):
    """FiniteSearch seeded with previously-cached best_configs prepended to the explicit config list."""

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        configs: Sequence[Config] | None = None,
        max_configs: int | None = None,
    ) -> None:
        BaseSearch.__init__(self, kernel, args)
        self.config_gen: ConfigGeneration = self.config_spec.create_config_generation(
            overrides=self.settings.autotune_config_overrides or None,
            advanced_controls_files=self.settings.autotune_search_acf or None,
            process_group_name=kernel.env.process_group_name,
        )
        cap = (
            max_configs
            if max_configs is not None
            else self.settings.autotune_best_available_max_configs
        )
        cached: list[Config] = []
        for i, entry in enumerate(self._find_similar_cached_configs(cap)):
            try:
                cached.append(self.config_gen.unflatten(entry.to_mutable_flat_config()))
            except (
                ValueError,
                TypeError,
                KeyError,
                AssertionError,
                exc.InvalidConfig,
            ) as e:
                self.log(f"from_cache: failed to transfer cached config {i + 1}: {e}")
        self.log(f"from_cache: resolved {len(cached)} cached config(s) (cap={cap})")
        explicit: list[Config] = list(
            configs if configs is not None else kernel.configs
        )
        self.configs: list[Config] = [*cached, *explicit]
        if len(self.configs) < 2:
            raise exc.NotEnoughConfigs(len(self.configs))


def from_cache(
    *, max_configs: int | None = None, configs: Sequence[Config] | None = None
) -> Callable[..., CachedFiniteSearch]:
    """Return an autotuner_fn that seeds FiniteSearch with previously-cached best_configs."""

    def _fn(
        bound_kernel: BoundKernel, args: Sequence[object], **kwargs: object
    ) -> CachedFiniteSearch:
        return CachedFiniteSearch(
            bound_kernel, args, configs=configs, max_configs=max_configs
        )

    return _fn
