from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import textwrap
from typing import TYPE_CHECKING
import uuid

import torch
from torch._inductor.runtime.cache_dir_utils import (
    cache_dir,  # pyright: ignore[reportPrivateImportUsage]
)

from ..runtime.config import Config
from .base_cache import AutotuneCacheBase
from .base_cache import LooseAutotuneCacheKey
from .base_cache import StrictAutotuneCacheKey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)


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

    def __init__(self, autotuner: BaseSearch) -> None:
        super().__init__(autotuner)
        self.key = self._generate_key()

    def _generate_key(self) -> LooseAutotuneCacheKey:
        in_memory_cache_key = self.kernel.kernel._create_bound_kernel_cache_key(
            self.kernel,
            tuple(self.args),
            self.kernel.kernel.specialization_key(self.args),
        )
        kernel_source = textwrap.dedent(inspect.getsource(self.kernel.kernel.fn))
        kernel_source_hash = hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()

        hardware = None
        runtime_name = None

        for arg in self.args:
            if isinstance(arg, torch.Tensor):
                nms = torch.xpu if torch.xpu.is_available() else torch.cuda
                device_properties = nms.get_device_properties(arg.device)
                if torch.version.cuda is not None:  # pyright: ignore[reportAttributeAccessIssue]
                    hardware = device_properties.name
                    runtime_name = str(torch.version.cuda)
                elif torch.version.hip is not None:  # pyright: ignore[reportAttributeAccessIssue]
                    hardware = device_properties.gcnArchName
                    runtime_name = torch.version.hip  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    hardware = device_properties.name
                    runtime_name = device_properties.driver_version  # pyright: ignore[reportAttributeAccessIssue]

        assert hardware is not None and runtime_name is not None
        return LooseAutotuneCacheKey(
            specialization_key=in_memory_cache_key.specialization_key,
            extra_results=in_memory_cache_key.extra_results,
            kernel_source_hash=kernel_source_hash,
            hardware=hardware,
            runtime_name=runtime_name,
        )

    def _get_local_cache_path(self) -> Path:
        if (user_path := os.environ.get("HELION_CACHE_DIR", None)) is not None:
            cache_path = Path(user_path)
        else:
            cache_path = Path(cache_dir()) / "helion"

        return cache_path / f"{self.key.stable_hash()}.best_config"

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

        data = {
            "config": config.to_json(),
            "key": key_dict,
        }

        # Atomic write
        tmp = path.parent / f"tmp.{uuid.uuid4()!s}"
        tmp.write_text(json.dumps(data, indent=2))
        os.rename(str(tmp), str(path))

    def _get_cache_info_message(self) -> str:
        cache_dir = self._get_local_cache_path().parent
        return f"Cache directory: {cache_dir}. To run autotuning again, delete the cache directory or set HELION_SKIP_CACHE=1."

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
