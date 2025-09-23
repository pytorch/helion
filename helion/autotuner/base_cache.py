from __future__ import annotations

import abc
import dataclasses
import functools
import hashlib
import logging
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Hashable

from torch._inductor.codecache import build_code_hash
from torch._inductor.codecache import torch_key

from .._utils import counters
from .base_search import BaseAutotuner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)


class AutotuneCacheMeta(abc.ABCMeta):
    """Metaclass that enables the Cache[Search] syntax for autotuner cache classes."""

    def __getitem__(
        cls, search_cls: type[BaseSearch]
    ) -> Callable[[BoundKernel, Sequence[Any]], BaseAutotuner]:
        """Enable Cache[Search] syntax to create a factory function.

        Args:
            search_cls: The search class to use with this cache

        Returns:
            A factory function that creates cache instances with the specified search
        """

        def factory(kernel: BoundKernel, args: Sequence[Any]) -> BaseAutotuner:
            return cls(search_cls(kernel, args))  # type: ignore[misc]

        return factory


@functools.cache
def helion_key() -> str:
    here = os.path.abspath(__file__)
    helion_path = os.path.dirname(os.path.dirname(here))

    combined_hash = hashlib.sha256()
    build_code_hash([helion_path], "", combined_hash)
    return combined_hash.hexdigest()


@functools.cache
def torch_key_wrapper() -> str:
    return torch_key().hex()


@functools.cache
def triton_key_wrapper() -> str:
    from torch._inductor.runtime.triton_compat import triton_key

    return triton_key()


class CacheKeyBase:
    """
    Base class to provide utility functions to all cache key dataclasses
    """

    def stable_hash(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class BoundKernelInMemoryCacheKey(CacheKeyBase):
    """
    Default in memory cache key.

    This key includes:

    specialization_key: Information about all kernel inputs.
                        For tensors this means their device, shape, size etc.
    extra_results: Information regarding `hl.specialize` decisions
    """

    specialization_key: tuple[Hashable, ...]
    extra_results: tuple[Hashable, ...]


@dataclasses.dataclass(frozen=True)
class LooseAutotuneCacheKey(BoundKernelInMemoryCacheKey):
    """
    Autotune Cache key to use for most use cases.

    This key includes (in addition to BoundKernelInMemoryCacheKey):

    kernel_source_hash: Hash of source code of input Helion kernel
    hardware: Hardware of the input device
    runtime_name: Version of the cuda/rocm arch
    """

    kernel_source_hash: str
    hardware: str
    runtime_name: str

    def stable_hash(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class StrictAutotuneCacheKey(LooseAutotuneCacheKey):
    """
    Autotune Cache key to use for utmost strictness in terms of re-autotuning
    when library source code changes.

    This key includes (in addition to StrictAutotuneCacheKey):

    helion_key: Hash of source code of Helion
    torch_key: Hash of source code of PyTorch
    triton_key: Hash of source code of Triton
    """

    helion_key: str = dataclasses.field(default_factory=helion_key)
    torch_key: str = dataclasses.field(default_factory=torch_key_wrapper)
    triton_key: str = dataclasses.field(default_factory=triton_key_wrapper)


class AutotuneCacheBase(BaseAutotuner, abc.ABC, metaclass=AutotuneCacheMeta):
    """
    Abstract base class that all autotune caches need to implement.
    Any user defined cache will need to extend this class, and
    provide implementations for get and put methods.
    """

    def __init__(self, autotuner: BaseSearch) -> None:
        self.autotuner = autotuner
        self.kernel = self.autotuner.kernel
        self.args = self.autotuner.args

    @abc.abstractmethod
    def get(self) -> Config | None:
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, config: Config) -> None:
        raise NotImplementedError

    def autotune(self) -> Config:
        if os.environ.get("HELION_SKIP_CACHE", "") not in {"", "0", "false", "False"}:
            return self.autotuner.autotune()

        if (config := self.get()) is not None:
            counters["autotune"]["cache_hit"] += 1
            log.debug("cache hit: %s", str(config))
            return config

        counters["autotune"]["cache_miss"] += 1
        log.debug("cache miss")

        config = self.autotuner.autotune()

        self.put(config)
        counters["autotune"]["cache_put"] += 1
        log.debug("cache put: %s", str(config))

        return config
