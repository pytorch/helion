"""
File-based autotune cache for Helion (PR for issue #164).

Stores tuned configs in a single shared JSON file, keyed by a
configurable template built from:

    kernel_name, source_hash, dtypes, shapes, device, backend
    + version stamp: Helion, Triton, CUDA runtime

Version stamp mismatch → cache miss (no error, just re-tunes).
Supports HELION_AUTOTUNE_CACHE_PATH and HELION_AUTOTUNE_CACHE_KEY env-var
overrides.  Writes are atomic (write-then-rename).

Typical use:

    HELION_AUTOTUNE_CACHE=FileAutotuneCache                       \\
    HELION_AUTOTUNE_CACHE_PATH=~/.helion/project_cache.json       \\
    python train.py

On first run it tunes and writes.  Every subsequent run (same
hardware/code) reads from the cache and skips autotuning.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..runtime.config import Config
from .base_cache import AutotuneCacheBase
from .base_cache import CacheKeyBase
from .local_cache import build_loose_cache_key
from .local_cache import get_helion_cache_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base_cache import LooseAutotuneCacheKey
    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)

# Bump this when the on-disk format changes in a breaking way.
_FORMAT_VERSION = 1

# Default file name inside HELION_CACHE_DIR.
_DEFAULT_CACHE_FILE = "autotune_cache.json"

# Template used to produce the per-entry lookup key.
# All field names listed under _build_key_fields() may be referenced.
_DEFAULT_KEY_TEMPLATE = (
    "{version_stamp}-{kernel_name}-{hardware}-{runtime_name}"
    "-{backend}-{config_spec_hash}-{specialization_key_hash}"
)


# ---------------------------------------------------------------------------
# Version stamp helpers
# ---------------------------------------------------------------------------

@functools.cache
def _helion_version() -> str:
    try:
        return importlib.metadata.version("helion")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


@functools.cache
def _triton_version() -> str:
    try:
        import triton  # type: ignore[import-not-found]

        return getattr(triton, "__version__", "unknown")
    except ImportError:
        return "none"


@functools.cache
def _cuda_version() -> str:
    if torch.version.cuda is not None:
        return torch.version.cuda
    if torch.version.hip is not None:
        return f"hip-{torch.version.hip}"
    return "none"


@functools.cache
def build_version_stamp() -> str:
    """Return a short hash that changes when Helion, Triton or CUDA changes."""
    raw = f"{_helion_version()}/{_triton_version()}/{_cuda_version()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class _FileCacheKey(CacheKeyBase):
    """Lightweight key object reconstructed from stored key_fields dicts."""

    def __init__(self, fields: dict[str, str]) -> None:
        for attr, val in fields.items():
            setattr(self, attr, val)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# File-based cache
# ---------------------------------------------------------------------------

class FileAutotuneCache(AutotuneCacheBase):
    """
    Autotune cache that stores all configs for a project in one JSON file.

    Key differences from ``LocalAutotuneCache``:

    * **Single shared file** – easy to commit, copy across machines.
    * **Configurable key template** – override via ``autotune_cache_key``
      setting or ``HELION_AUTOTUNE_CACHE_KEY`` env var.
    * **Version stamp** – if Helion, Triton or CUDA version changes the
      entry is treated as a miss and autotuning runs again.
    * **Atomic writes** – write to a ``.tmp`` file then ``os.replace()``.

    Enable with::

        HELION_AUTOTUNE_CACHE=FileAutotuneCache
        HELION_AUTOTUNE_CACHE_PATH=/path/to/cache.json   # optional
    """

    def __init__(self, autotuner: BaseSearch) -> None:
        super().__init__(autotuner)
        self.loose_key: LooseAutotuneCacheKey = build_loose_cache_key(
            self.kernel, self.args
        )
        self.cache_path: Path = self._resolve_cache_path()
        self.key_fields: dict[str, str] = self._build_key_fields()
        self.cache_key: str = self._render_cache_key()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_cache_path(self) -> Path:
        path_setting = self.autotuner.settings.autotune_cache_path
        if path_setting is not None:
            return Path(path_setting).expanduser()
        return get_helion_cache_dir() / _DEFAULT_CACHE_FILE

    def _build_key_fields(self) -> dict[str, str]:
        """Return a dict of all substitutable template fields."""
        k = self.loose_key
        return {
            "kernel_name": self.kernel.kernel.name,
            "kernel_source_hash": k.kernel_source_hash,
            "hardware": k.hardware,
            "runtime_name": k.runtime_name,
            "backend": k.backend,
            "config_spec_hash": k.config_spec_hash,
            "specialization_key_hash": _sha256(repr(k.specialization_key)),
            "extra_results_hash": _sha256(repr(k.extra_results)),
            "stable_hash": k.stable_hash(),
            "version_stamp": build_version_stamp(),
            # Human-readable extras (not recommended for keying, but available)
            "helion_version": _helion_version(),
            "triton_version": _triton_version(),
            "cuda_version": _cuda_version(),
        }

    def _render_cache_key(self) -> str:
        template = (
            self.autotuner.settings.autotune_cache_key or _DEFAULT_KEY_TEMPLATE
        )
        try:
            return template.format(**self.key_fields)
        except KeyError as exc:
            available = ", ".join(sorted(self.key_fields))
            raise ValueError(
                f"Unknown field {exc.args[0]!r} in autotune_cache_key template. "
                f"Available fields: {available}"
            ) from exc

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _read_entries(self) -> dict[str, dict[str, object]]:
        """Return raw entry dict from disk; {} on any error."""
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("Could not read autotune cache %s: %s", self.cache_path, exc)
            return {}
        if not isinstance(data, dict) or data.get("format_version") != _FORMAT_VERSION:
            log.debug(
                "Autotune cache %s has unexpected format/version, treating as empty",
                self.cache_path,
            )
            return {}
        entries = data.get("entries", {})
        return entries if isinstance(entries, dict) else {}

    def _write_entries(self, entries: dict[str, dict[str, object]]) -> None:
        """Atomically write entries to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "helion_version": _helion_version(),
            "triton_version": _triton_version(),
            "cuda_version": _cuda_version(),
            "entries": entries,
        }
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.cache_path)

    # ------------------------------------------------------------------
    # AutotuneCacheBase interface
    # ------------------------------------------------------------------

    def get(self) -> Config | None:
        entries = self._read_entries()
        raw = entries.get(self.cache_key)
        if raw is None or not isinstance(raw, dict) or "config" not in raw:
            return None
        try:
            return Config.from_json(raw["config"])
        except (ValueError, TypeError) as exc:
            log.warning(
                "Corrupt cache entry for key %s in %s: %s",
                self.cache_key,
                self.cache_path,
                exc,
            )
            return None

    def put(self, config: Config) -> None:
        entries = self._read_entries()
        entries[self.cache_key] = {
            "config": config.to_json(),
            "key_fields": self.key_fields,
        }
        self._write_entries(entries)

    def _get_cache_info_message(self) -> str:
        return (
            f"Cache file: {self.cache_path}. "
            "Override path with HELION_AUTOTUNE_CACHE_PATH. "
            "Delete the file or change hardware/code to invalidate."
        )

    def _get_cache_key(self) -> CacheKeyBase:
        return self.loose_key

    def _list_cache_entries(self) -> Sequence[tuple[str, CacheKeyBase]]:
        entries = self._read_entries()
        results: list[tuple[str, CacheKeyBase]] = []
        for key, raw in entries.items():
            if not isinstance(raw, dict):
                continue
            kf: dict[str, str] = {
                str(k): str(v)
                for k, v in raw.get("key_fields", {}).items()
            }
            name = kf.get("kernel_name", "unknown")
            desc = f"{key}  (kernel={name})"
            results.append((desc, _FileCacheKey(kf)))
        return results
