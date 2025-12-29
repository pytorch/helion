from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import functools
import hashlib
import logging
import os
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Hashable

from torch._inductor.codecache import build_code_hash
from torch._inductor.codecache import torch_key

from .. import exc
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

    full_key = triton_key()
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


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

    def _get_cache_info_message(self) -> str:
        """Return a message describing where the cache is and how to clear it."""
        return ""

    @abc.abstractmethod
    def _get_cache_key(self) -> CacheKeyBase:
        """Return the cache key for this cache instance."""
        raise NotImplementedError

    @abc.abstractmethod
    def _list_cache_entries(self) -> Sequence[tuple[str, CacheKeyBase]]:
        """Return a sequence of (description, key) tuples for all cache entries."""
        raise NotImplementedError

    def autotune(self, *, skip_cache: bool = False) -> Config:
        if skip_cache or os.environ.get("HELION_SKIP_CACHE", "") not in {
            "",
            "0",
            "false",
            "False",
        }:
            return self.autotuner.autotune()

        if (config := self.get()) is not None:
            counters["autotune"]["cache_hit"] += 1
            log.debug("cache hit: %s", str(config))
            kernel_decorator = self.kernel.format_kernel_decorator(
                config, self.autotuner.settings
            )
            print(f"Using cached config:\n\t{kernel_decorator}", file=sys.stderr)
            cache_info = self._get_cache_info_message()
            self.autotuner.log(
                f"Found cached config for {self.kernel.kernel.name}, skipping autotuning.\n{cache_info}"
            )
            return config

        counters["autotune"]["cache_miss"] += 1
        log.debug("cache miss")

        if os.environ.get("HELION_ASSERT_CACHE_HIT") == "1":
            current_key = self._get_cache_key()
            print("\n" + "=" * 80, file=sys.stderr)
            print("HELION_ASSERT_CACHE_HIT: Cache miss detected!", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"\nKernel: {self.kernel.kernel.name}", file=sys.stderr)
            print(f"\nCurrent cache key:\n{current_key}", file=sys.stderr)

            cache_entries = self._list_cache_entries()
            if cache_entries:
                print(
                    f"\n{len(cache_entries)} other cache entries exist (but don't match):",
                    file=sys.stderr,
                )
                for i, (desc, cached_key) in enumerate(cache_entries, 1):
                    print(f"\n[Entry {i}] {desc}", file=sys.stderr)
                    print("  Key differences:", file=sys.stderr)
                    has_diff = False
                    for field_name in vars(current_key):
                        current_val = str(getattr(current_key, field_name))
                        cached_val = str(getattr(cached_key, field_name, "<missing>"))
                        if current_val != cached_val:
                            has_diff = True
                            print(f"    {field_name}:", file=sys.stderr)
                            print(f"      Current:  {current_val}", file=sys.stderr)
                            print(f"      Cached:   {cached_val}", file=sys.stderr)
                    if not has_diff:
                        print(
                            "    (no differences found, likely a hash collision)",
                            file=sys.stderr,
                        )
            else:
                print("\nNo existing cache entries found.", file=sys.stderr)

            print("=" * 80 + "\n", file=sys.stderr)
            raise exc.CacheAssertionError(self.kernel.kernel.name)

        self.autotuner.log("Starting autotuning process, this may take a while...")

        config = self.autotuner.autotune()

        self.put(config)
        counters["autotune"]["cache_put"] += 1
        log.debug("cache put: %s", str(config))

        return config
