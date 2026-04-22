from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import contextlib
from typing import TYPE_CHECKING

from .. import exc

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .base_search import BaseSearch


class ConfigSource(ABC):
    """A deferred source of configs resolved at autotune time."""

    @abstractmethod
    def resolve(self, search: BaseSearch) -> list[Config]:
        """Return the list of Configs this source provides."""


class CachedConfigSource(ConfigSource):
    """Load previously-saved best_configs matching the current kernel."""

    def __init__(self, max_configs: int | None = None) -> None:
        self.max_configs = max_configs

    def resolve(self, search: BaseSearch) -> list[Config]:
        """Return matching cached Configs for this kernel (empty on miss)."""
        cap = (
            self.max_configs
            if self.max_configs is not None
            else search.settings.autotune_best_available_max_configs
        )
        entries = search._find_similar_cached_configs(cap)
        configs: list[Config] = []
        for entry in entries:
            with contextlib.suppress(
                ValueError,
                TypeError,
                KeyError,
                AssertionError,
                exc.InvalidConfig,
            ):
                configs.append(
                    search.config_gen.unflatten(entry.to_mutable_flat_config())
                )
        search.log(f"from_cache: resolved {len(configs)} cached config(s) (cap={cap})")
        return configs


def from_cache(*, max_configs: int | None = None) -> CachedConfigSource:
    """Return a ConfigSource that seeds FiniteSearch with previously-cached best_configs."""
    return CachedConfigSource(max_configs=max_configs)
