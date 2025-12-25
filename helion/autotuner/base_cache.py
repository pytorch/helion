from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import functools
import hashlib
import logging
import os
import platform
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Hashable

import torch
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


# Known compute capabilities in descending order (newest first)
# This allows fallback to older architectures when heuristics aren't available
_CUDA_COMPUTE_CAPS: list[str] = [
    "sm100",
    "sm90",
    "sm89",
    "sm87",
    "sm86",
    "sm80",
    "sm75",
    "sm72",
    "sm70",
]

_ROCM_ARCHS: list[str] = [
    "gfx950",
    "gfx942",
    "gfx941",
    "gfx940",
    "gfx90a",
    "gfx908",
    "gfx906",
    "gfx900",
]


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """
    Hardware information for cache keys and heuristic file discovery.

    Attributes:
        device_kind: Device type ('cuda', 'rocm', 'xpu', 'cpu')
        hardware_name: Device name (e.g., 'NVIDIA H100', 'gfx90a', 'cpu')
        runtime_version: Runtime version (e.g., '12.4', 'gfx90a', 'x86_64')
        compute_capability: Compute capability for heuristics (e.g., 'sm90', 'gfx90a')
    """

    device_kind: str
    hardware_name: str
    runtime_version: str
    compute_capability: str

    @property
    def hardware_id(self) -> str:
        """Get a unique identifier string for this hardware."""
        safe_name = self.hardware_name.replace(" ", "_")
        return f"{self.device_kind}_{safe_name}_{self.runtime_version}"

    def get_compatible_compute_ids(self) -> list[str]:
        """
        Get a list of compatible compute IDs for fallback, ordered from current to oldest.

        For CUDA/ROCm, returns the current compute capability followed by all older
        compatible architectures. This allows using heuristics tuned on older hardware
        when newer hardware-specific heuristics aren't available.
        """
        if self.device_kind == "cuda":
            arch_list = _CUDA_COMPUTE_CAPS
        elif self.device_kind == "rocm":
            arch_list = _ROCM_ARCHS
        else:
            return [self.compute_capability]

        try:
            current_idx = arch_list.index(self.compute_capability)
            return arch_list[current_idx:]
        except ValueError:
            return [self.compute_capability, *arch_list]


@functools.cache
def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
    """
    Get hardware information for the current or specified device.

    This is the single source of truth for hardware detection, used by both
    local cache and AOT cache.

    Args:
        device: Optional device to get info for. If None, uses first available GPU or CPU.

    Returns:
        HardwareInfo with device details for caching and heuristic lookup.
    """
    # CPU fallback
    if device is not None and device.type == "cpu":
        return HardwareInfo(
            device_kind="cpu",
            hardware_name="cpu",
            runtime_version=platform.machine().lower(),
            compute_capability=platform.machine().lower(),
        )

    # XPU (Intel) path
    if (
        device is not None
        and device.type == "xpu"
        and getattr(torch, "xpu", None) is not None
        and torch.xpu.is_available()
    ):
        props = torch.xpu.get_device_properties(device)
        return HardwareInfo(
            device_kind="xpu",
            hardware_name=props.name,
            runtime_version=props.driver_version,
            compute_capability=props.name,  # XPU doesn't have compute capability
        )

    # CUDA/ROCm path
    if torch.cuda.is_available():
        dev = (
            device
            if device is not None and device.type == "cuda"
            else torch.device("cuda:0")
        )
        props = torch.cuda.get_device_properties(dev)

        if torch.version.cuda is not None:
            return HardwareInfo(
                device_kind="cuda",
                hardware_name=props.name,
                runtime_version=str(torch.version.cuda),
                compute_capability=f"sm{props.major}{props.minor}",
            )
        if torch.version.hip is not None:
            return HardwareInfo(
                device_kind="rocm",
                hardware_name=props.gcnArchName,
                runtime_version=torch.version.hip,
                compute_capability=props.gcnArchName,
            )

    # CPU fallback
    return HardwareInfo(
        device_kind="cpu",
        hardware_name="cpu",
        runtime_version=platform.machine().lower(),
        compute_capability=platform.machine().lower(),
    )


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
            # Suppress verbose output in AOT evaluate mode (quiet by default when using heuristics)
            aot_mode = os.environ.get("HELION_AOT_MODE", "evaluate").lower()
            aot_verbose = os.environ.get("HELION_AOT_VERBOSE", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if aot_mode != "evaluate" or aot_verbose:
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
