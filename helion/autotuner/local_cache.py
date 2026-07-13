from __future__ import annotations

import atexit
import contextlib
import dataclasses
import hashlib
import inspect
import itertools
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import tempfile
import textwrap
from typing import TYPE_CHECKING
import uuid

import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir

from .._compat import extract_device
from .._compat import get_device_name
from ..runtime.config import Config
from .base_cache import AutotuneCacheBase
from .base_cache import LooseAutotuneCacheKey
from .base_cache import StrictAutotuneCacheKey

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing import Callable

    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)


def get_helion_cache_dir() -> Path:
    """Return the root directory for all Helion caches."""
    if (user_path := os.environ.get("HELION_CACHE_DIR")) is not None:
        return Path(user_path)
    return Path(cache_dir()) / "helion"


def helion_triton_cache_dir(device_index: int) -> str:
    """Return per-device Triton cache directory under Helion's cache root."""
    return str(get_helion_cache_dir() / "triton" / str(device_index))


def helion_cute_cache_dir(device_index: int) -> str:
    """Return per-device CuTe DSL cache directory under Helion's cache root."""
    return str(get_helion_cache_dir() / "cute" / str(device_index))


_npu_volatile_triton_cache: str | None = None


def npu_volatile_triton_cache_dir() -> str:
    """Per-process temp Triton cache on NPU (stale hits seen under ``helion/triton/<i>``).

    Opt out: ``HELION_NPU_PERSISTENT_TRITON_CACHE=1``.
    """
    global _npu_volatile_triton_cache
    if _npu_volatile_triton_cache is None:
        _npu_volatile_triton_cache = tempfile.mkdtemp(
            prefix="helion_npu_triton_volatile_",
        )

        def _cleanup_volatile_triton_cache() -> None:
            shutil.rmtree(_npu_volatile_triton_cache, ignore_errors=True)

        atexit.register(_cleanup_volatile_triton_cache)
    return _npu_volatile_triton_cache


def npu_persistent_triton_cache_requested() -> bool:
    """When true, NPU uses :func:`helion_triton_cache_dir` like other devices."""
    v = os.environ.get("HELION_NPU_PERSISTENT_TRITON_CACHE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def triton_env_is_helion_autotune_ephemeral() -> bool:
    """True while :meth:`helion.runtime.kernel.BoundKernel._ephemeral_triton_cache` is active."""
    d = os.environ.get("TRITON_CACHE_DIR", "")
    return "helion_autotune_" in d


def should_force_npu_volatile_triton_cache(device: torch.device) -> bool:
    """NPU default: volatile disk root, unless persistent requested or autotune ephemeral is active."""
    if device.type != "npu":
        return False
    if npu_persistent_triton_cache_requested():
        return False
    if triton_env_is_helion_autotune_ephemeral():
        return False
    return True


_per_config_triton_cache_override: str | None = None


@contextlib.contextmanager
def per_config_triton_cache_for_autotune(path: str) -> Iterator[None]:
    """Make ``compile_config`` use *path* as ``TRITON_CACHE_DIR`` (overrides NPU volatile root)."""
    global _per_config_triton_cache_override
    prev = _per_config_triton_cache_override
    _per_config_triton_cache_override = path
    try:
        yield
    finally:
        _per_config_triton_cache_override = prev


def get_per_config_triton_cache_override() -> str | None:
    return _per_config_triton_cache_override


def autotune_fresh_triton_subdir_per_benchmark() -> bool:
    """Env ``HELION_AUTOTUNE_FRESH_TRITON_SUBDIR_PER_BENCHMARK``: isolated disk cache per search candidate."""
    v = os.environ.get(
        "HELION_AUTOTUNE_FRESH_TRITON_SUBDIR_PER_BENCHMARK", ""
    ).strip().lower()
    return v in ("1", "true", "yes", "on")


def autotune_stable_triton_subdir_per_config() -> bool:
    """Env ``HELION_AUTOTUNE_TRITON_SUBDIR_STABLE`` (use with *FRESH_TRITON_SUBDIR*).

    Use ``TRITON_CACHE_DIR = <parent>/<sha256(config.config)>`` instead of a fresh temp dir per visit.
    Different Helion configs never share one Triton cache tree (avoids cross-config confusion); the same
    config reuses its tree so Triton can hit its own on-disk cache. Skip ``rmtree`` after each benchmark;
    on NPU the parent still lives under the process volatile root removed at exit.
    """
    v = os.environ.get("HELION_AUTOTUNE_TRITON_SUBDIR_STABLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def autotune_triton_candidate_parent_path() -> Path:
    """Directory containing one subdirectory per autotune candidate (stable hash or temp name)."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return Path(npu_volatile_triton_cache_dir()) / "autotune_candidates"
    return Path(tempfile.gettempdir()) / "helion_autotune_candidates"


def triton_cache_dir_for_autotune_candidate(config: Config, *, stable: bool) -> str:
    """Return ``TRITON_CACHE_DIR`` for this candidate: isolated from other configs' trees."""
    parent = autotune_triton_candidate_parent_path()
    parent.mkdir(parents=True, exist_ok=True)
    if stable:
        payload = json.dumps(config.config, sort_keys=True, default=str).encode()
        digest = hashlib.sha256(payload).hexdigest()[:48]
        p = parent / digest
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    return tempfile.mkdtemp(prefix="cfg_", dir=str(parent))


@dataclasses.dataclass(frozen=True)
class SavedBestConfig:
    """A parsed cache entry from a .best_config file."""

    hardware: str
    specialization_key: str
    config: Config
    config_spec_hash: str
    flat_config: tuple[object, ...] | None

    def to_mutable_flat_config(self) -> list[object]:
        """Return the stored flat_config as a mutable list."""
        assert self.flat_config is not None
        return list(self.flat_config)


def parse_cache_entry(raw: str) -> SavedBestConfig:
    """Parse a serialized .best_config JSON string into a SavedBestConfig.

    Raises:
        ValueError: if the payload is malformed or missing required fields.
    """
    try:
        data = json.loads(raw)
        fields = data["key"]["fields"]
        raw_flat = data.get("flat_config")
        if isinstance(raw_flat, str):
            flat_config: tuple[object, ...] | None = tuple(json.loads(raw_flat))
        elif raw_flat is not None:
            flat_config = tuple(raw_flat)
        else:
            flat_config = None
        return SavedBestConfig(
            hardware=fields.get("hardware", ""),
            specialization_key=fields.get("specialization_key", ""),
            config=Config.from_json(data["config"]),
            config_spec_hash=fields.get("config_spec_hash", ""),
            flat_config=flat_config,
        )
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"malformed cache entry: {e}") from e


def iter_cache_entries(
    cache_path: Path, *, max_scan: int | None = None
) -> Iterator[SavedBestConfig]:
    """Yield parsed cache entries from *cache_path*, newest first.

    Corrupt or unparsable files are skipped with a warning.
    """
    if not cache_path.exists():
        return

    files = list(cache_path.glob("*.best_config"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in itertools.islice(files, max_scan):
        # parse_cache_entry consolidates KeyError/TypeError/JSONDecodeError into ValueError.
        try:
            yield parse_cache_entry(p.read_text())
        except (OSError, ValueError) as e:
            log.warning("Skipping corrupt cache file %s: %s", p.name, e)


class LocalAutotuneCache(AutotuneCacheBase):
    """
    This class implements the local autotune cache, storing the
    best config artifact on the local file system either by default
    on torch's cache directory, or at a user specified HELION_CACHE_DIR
    directory.
    It uses the LooseAutotuneCacheKey implementation for the cache key
    which takes into account device and source code properties, but does
    not account for library level code changes such as Triton, Helion or
    PyTorch. Use StrictLocalAutotuneCache to consider these properties.
    """

    def __init__(
        self,
        autotuner: BaseSearch,
        *,
        autotuner_factory: Callable[[], BaseSearch] | None = None,
    ) -> None:
        super().__init__(autotuner, autotuner_factory=autotuner_factory)
        self.key = self._generate_key()

    def _generate_key(self) -> LooseAutotuneCacheKey:
        in_memory_cache_key = self.kernel.kernel._create_bound_kernel_cache_key(
            self.kernel,
            tuple(self.args),
            self.kernel.kernel._base_specialization_key(self.args),
        )
        kernel_source = textwrap.dedent(inspect.getsource(self.kernel.kernel.fn))
        kernel_source_hash = hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()

        dev = extract_device(self.args)
        assert dev is not None

        hardware = get_device_name(dev)
        runtime_name = None

        if (
            dev.type == "xpu"
            and getattr(torch, "xpu", None) is not None
            and torch.xpu.is_available()
        ):
            runtime_name = torch.xpu.get_device_properties(dev).driver_version
        elif dev.type == "cuda" and torch.cuda.is_available():
            if torch.version.cuda is not None:
                runtime_name = str(torch.version.cuda)
            elif torch.version.hip is not None:
                runtime_name = torch.version.hip
        elif dev.type == "mps":
            # Include OS version as Metal runtime is part of OS
            runtime_name = platform.mac_ver()[0] or "mps"
        elif dev.type == "tpu":
            hardware = "tpu"
            try:
                import torch_tpu  # type: ignore[import-not-found]

                runtime_name = getattr(torch_tpu, "__version__", "unknown")
            except ImportError:
                runtime_name = "unknown"
        elif dev.type == "npu":
            hardware = "npu"
            try:
                import torch_npu  # type: ignore[import-not-found]

                runtime_name = torch_npu.npu.get_device_name()
            except ImportError:
                runtime_name = "unknown"
        elif dev.type == "cpu" and self.kernel.kernel.settings.backend == "pallas":
            hardware = "pallas_interpret"
            runtime_name = "interpret"
        elif dev.type == "mtia":
            hardware = hardware or "mtia"
            try:
                runtime_name = str(torch.mtia.get_device_properties(dev))
            except Exception:
                runtime_name = "unknown"

        assert hardware is not None and runtime_name is not None
        config_spec_hash = self.kernel.config_spec.structural_fingerprint_hash(
            advanced_controls_files=self.autotuner.settings.autotune_search_acf or None
        )
        return LooseAutotuneCacheKey(
            specialization_key=in_memory_cache_key.specialization_key,
            extra_results=in_memory_cache_key.extra_results,
            kernel_source_hash=kernel_source_hash,
            hardware=hardware,
            runtime_name=runtime_name,
            backend=self.kernel.env.backend.name,
            config_spec_hash=config_spec_hash,
            extra_cache_key=self.kernel.extra_cache_key(),
            best_of_k=self.autotuner.settings.autotune_best_of_k,
        )

    def _get_local_cache_path(self) -> Path:
        return get_helion_cache_dir() / f"{self.key.stable_hash()}.best_config"

    def get(self) -> Config | None:
        path = self._get_local_cache_path()
        try:
            data = json.loads(path.read_text())
            return Config.from_json(data["config"])
        except Exception:
            return None

    def put(self, config: Config) -> None:
        path = self._get_local_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save both config and key for better debugging
        # Store key as dict for safer reconstruction (avoids eval)
        key_dict = {
            "type": type(self.key).__name__,
            "fields": {k: str(v) for k, v in vars(self.key).items()},
        }

        data: dict[str, object] = {
            "config": config.to_json(),
            "key": key_dict,
        }

        config_gen = self.kernel.config_spec.create_config_generation(
            advanced_controls_files=self.autotuner.settings.autotune_search_acf or None
        )
        data["flat_config"] = json.dumps(config_gen.flatten(config))

        backend_cache_key = self.kernel.backend_cache_key(config)
        if backend_cache_key is None:
            # Config may have been minimized (default values stripped),
            # so it won't match the full config in _compile_cache.
            # Expand it back by merging with defaults.
            default = self.kernel.config_spec.default_config()
            # pyrefly: ignore [bad-argument-type]
            full_config = Config(**(default.config | config.config))
            backend_cache_key = self.kernel.backend_cache_key(full_config)
        if backend_cache_key is not None:
            data["backend_cache_key"] = backend_cache_key

        # Atomic write
        tmp = path.parent / f"tmp.{uuid.uuid4()!s}"
        tmp.write_text(json.dumps(data, indent=2))
        os.rename(str(tmp), str(path))

    def _get_cache_info_message(self) -> str:
        cache_dir = self._get_local_cache_path().parent
        return f"Cache directory: {cache_dir}. To re-tune and update the cache, set HELION_FORCE_AUTOTUNE=1."

    def _get_cache_key(self) -> LooseAutotuneCacheKey:
        return self.key

    def _list_cache_entries(self) -> Sequence[tuple[str, LooseAutotuneCacheKey]]:
        """List all cache entries in the cache directory."""
        cache_dir = self._get_local_cache_path().parent
        if not cache_dir.exists():
            return []

        current_key_hash = self.key.stable_hash()
        entries: list[tuple[str, LooseAutotuneCacheKey]] = []
        for cache_file in cache_dir.glob("*.best_config"):
            try:
                data = json.loads(cache_file.read_text())
                file_hash = cache_file.stem

                if file_hash == current_key_hash:
                    continue

                key_data = data["key"]

                # Create a simple namespace object that has the same attributes
                # for comparison purposes (we don't need the full key object)
                class CachedKey:
                    def __init__(self, fields: dict[str, str]) -> None:
                        for name, value in fields.items():
                            setattr(self, name, value)

                cached_key = CachedKey(key_data["fields"])
                entries.append((cache_file.name, cached_key))  # type: ignore[arg-type]
            except Exception:
                pass

        return entries


class StrictLocalAutotuneCache(LocalAutotuneCache):
    """
    Stricter implementation of the local autotune cache, which takes into
    account library level code changes such as Triton, Helion or PyTorch.
    """

    def _generate_key(self) -> StrictAutotuneCacheKey:
        loose_key = super()._generate_key()
        return StrictAutotuneCacheKey(**vars(loose_key))
