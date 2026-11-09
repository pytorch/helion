from __future__ import annotations

from collections.abc import ItemsView
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import ValuesView
from typing import Hashable
from typing import TypeVar
from typing import cast

from .config import Config
from .config import _config_key
from .config import _copy_config_structure

_T = TypeVar("_T")


def _stored_config(config: dict[str, object]) -> Config:
    return Config.from_dict(cast("dict[str, object]", _copy_config_structure(config)))


class ConfigCache(MutableMapping[Config, _T]):
    """A config-indexed cache whose identity is stricter than public equality.

    Config's public equality intentionally follows Python: 1 == True == 1.0
    and -0.0 == 0.0. These values can generate different code, so compiled
    callables and their metadata must not use Config's hash/equality directly.
    Keep the Config-facing mapping interface for backend cache management.
    """

    def __init__(
        self, entries: Mapping[Config, _T] | Iterable[tuple[Config, _T]] = ()
    ) -> None:
        self._entries: dict[Hashable, tuple[dict[str, object], _T]] = {}
        if isinstance(entries, ConfigCache):
            # Reuse captured keys instead of looking up mutable opaque leaves.
            self._entries.update(entries._entries)
        else:
            self.update(entries)

    def __getitem__(self, config: Config) -> _T:
        return self._entries[_config_key(config)][1]

    def __setitem__(self, config: Config, value: _T) -> None:
        self._entries[_config_key(config)] = (
            cast("dict[str, object]", _copy_config_structure(config.config)),
            value,
        )

    def __delitem__(self, config: Config) -> None:
        del self._entries[_config_key(config)]

    def __iter__(self) -> Iterator[Config]:
        for config, _ in self._entries.values():
            yield _stored_config(config)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def items(self) -> ItemsView[Config, _T]:
        return _ConfigItems(self)

    def values(self) -> ValuesView[_T]:
        return _ConfigValues(self)

    def popitem(self) -> tuple[Config, _T]:
        _, (config, value) = self._entries.popitem()
        return _stored_config(config), value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.items())!r})"


class _ConfigItems(ItemsView[Config, _T]):
    def __init__(self, cache: ConfigCache[_T]) -> None:
        super().__init__(cache)
        self._cache = cache

    def __iter__(self) -> Iterator[tuple[Config, _T]]:
        # Opaque containers remain caller-owned. Iteration must not look up
        # a stored entry again through an opaque value's possibly new contents.
        for config, value in self._cache._entries.values():
            yield _stored_config(config), value


class _ConfigValues(ValuesView[_T]):
    def __init__(self, cache: ConfigCache[_T]) -> None:
        super().__init__(cache)
        self._cache = cache

    def __iter__(self) -> Iterator[_T]:
        for _, value in self._cache._entries.values():
            yield value
