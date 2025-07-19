from __future__ import annotations

import abc
import dataclasses
import functools
import hashlib
import logging
import os
from typing import TYPE_CHECKING
from typing import Hashable
from typing import Sequence

from torch._inductor.codecache import build_code_hash
from torch._inductor.codecache import torch_key
from torch._inductor.runtime.triton_compat import triton_key

from .._utils import counters

if TYPE_CHECKING:
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)


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


class AutotuneCacheBase(abc.ABC):
    """
    Abstract base class that all autotune caches need to implement.
    Any user defined cache will need to extend this class, and
    provide implementations for get and put methods.
    """

    def __init__(
        self, kernel: BoundKernel, args: Sequence[object], autotuner: BaseSearch
    ) -> None:
        self.autotuner = autotuner
        self.kernel = kernel
        self.args = args

    @abc.abstractmethod
    def get(self) -> Config | None:
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, config: Config) -> None:
        raise NotImplementedError

    def autotune(self) -> Config:
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
