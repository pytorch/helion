from __future__ import annotations

import abc
import importlib
import json
import logging
import os
from typing import TYPE_CHECKING

from ..runtime.config import Config
from .local_cache import LocalAutotuneCache
from .local_cache import StrictLocalAutotuneCache

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)


class RemoteCacheBackend(abc.ABC):
    """User-provided remote cache backend.

    Subclass this and point ``HELION_REMOTE_CACHE_BACKEND`` at your
    fully-qualified class name (e.g. ``mypackage.cache.RedisBackend``).
    The subclass is responsible for reading its own configuration
    (env vars, config files, etc.).
    """

    @abc.abstractmethod
    def get(self, key: str) -> str | None:
        """Return the cached JSON string for *key*, or ``None`` on miss."""

    @abc.abstractmethod
    def put(self, key: str, data: str) -> None:
        """Store a JSON string under *key*."""

    def list(self, max_results: int | None = None) -> Iterable[str]:
        """Return cached JSON entries, newest first. Empty by default; override to enable warm-start from remote."""
        return ()


_ENV_VAR = "HELION_REMOTE_CACHE_BACKEND"


def _load_remote_backend() -> RemoteCacheBackend:
    value = os.environ.get(_ENV_VAR)
    if value is None or (value := value.strip()) == "":
        raise ValueError(
            f"{_ENV_VAR} must be set when using "
            "RemoteAutotuneCache or StrictRemoteAutotuneCache "
            "(e.g. 'mypackage.module.MyBackend')"
        )
    module_path, class_name = value.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, RemoteCacheBackend)):
        raise TypeError(
            f"{_ENV_VAR} must point to a RemoteCacheBackend subclass, got {cls!r}"
        )
    return cls()


def _load_remote_backend_if_configured() -> RemoteCacheBackend | None:
    """Return the remote backend if HELION_REMOTE_CACHE_BACKEND is set, else None."""
    if not os.environ.get(_ENV_VAR):
        return None
    return _load_remote_backend()


def _remote_get(backend: RemoteCacheBackend, cache_hash: str) -> str | None:
    try:
        return backend.get(cache_hash)
    except Exception:
        log.warning("Remote cache read failed, falling back to local", exc_info=True)
        return None


def _remote_put(backend: RemoteCacheBackend, cache_hash: str, data: str) -> None:
    try:
        backend.put(cache_hash, data)
        log.debug("remote cache put: %s", cache_hash)
    except Exception:
        log.warning("Remote cache write failed (local still written)", exc_info=True)


class RemoteAutotuneCache(LocalAutotuneCache):
    """Local cache extended with a remote read-through / write-through layer.

    On *get*: try remote first, materialize locally on hit, fall back to local.
    On *put*: write to both local and remote.
    """

    def __init__(self, autotuner: BaseSearch) -> None:
        super().__init__(autotuner)
        self._backend = _load_remote_backend()

    def get(self) -> Config | None:
        cache_hash = self.key.stable_hash()
        data = _remote_get(self._backend, cache_hash)
        if data is not None:
            config = Config.from_json(json.loads(data)["config"])
            super().put(config)
            log.debug("remote cache hit: %s", cache_hash)
            return config
        return super().get()

    def put(self, config: Config) -> None:
        super().put(config)
        # Ship the exact bytes super().put() just atomically wrote, so the
        # remote payload mirrors the on-disk shape without re-serialising.
        _remote_put(
            self._backend,
            self.key.stable_hash(),
            self._get_local_cache_path().read_text(),
        )

    def _get_cache_info_message(self) -> str:
        base = super()._get_cache_info_message()
        return f"{base} Remote backend: {type(self._backend).__qualname__}."


class StrictRemoteAutotuneCache(StrictLocalAutotuneCache):
    """Strict local cache extended with a remote read-through / write-through layer."""

    def __init__(self, autotuner: BaseSearch) -> None:
        super().__init__(autotuner)
        self._backend = _load_remote_backend()

    def get(self) -> Config | None:
        cache_hash = self.key.stable_hash()
        data = _remote_get(self._backend, cache_hash)
        if data is not None:
            config = Config.from_json(json.loads(data)["config"])
            super().put(config)
            log.debug("remote cache hit: %s", cache_hash)
            return config
        return super().get()

    def put(self, config: Config) -> None:
        super().put(config)
        # Ship the exact bytes super().put() just atomically wrote.
        _remote_put(
            self._backend,
            self.key.stable_hash(),
            self._get_local_cache_path().read_text(),
        )

    def _get_cache_info_message(self) -> str:
        base = super()._get_cache_info_message()
        return f"{base} Remote backend: {type(self._backend).__qualname__}."
