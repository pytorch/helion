from __future__ import annotations

import hashlib
import inspect
import logging
import os
from pathlib import Path
import textwrap
from typing import TYPE_CHECKING

import torch
from torch._inductor.runtime.cache_dir_utils import (
    cache_dir,  # pyright: ignore[reportPrivateImportUsage]
)

from ..runtime.config import Config
from .base_cache import AutotuneCacheBase
from .base_cache import LooseAutotuneCacheKey
from .base_cache import StrictAutotuneCacheKey

if TYPE_CHECKING:
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
            return Config.load(path)
        except Exception:
            return None

    def put(self, config: Config) -> None:
        path = self._get_local_cache_path()
        config.save(path)


class StrictLocalAutotuneCache(LocalAutotuneCache):
    """
    Stricter implementation of the local autotune cache, which takes into
    account library level code changes such as Triton, Helion or PyTorch.
    """

    def _generate_key(self) -> StrictAutotuneCacheKey:
        loose_key = super()._generate_key()
        return StrictAutotuneCacheKey(**vars(loose_key))
