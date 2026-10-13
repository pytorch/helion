"""Read-only real-parent target snapshot for one guarded source interval.

This module imports no torch/SDK and executes no hardware query at import time.
Only capture(), explicitly called by the owning parent, resolves actual metadata.
"""

from __future__ import annotations

from collections import Counter
import dataclasses
import importlib
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from types import ModuleType

    import torch
    from torch._dynamo.device_interface import CudaInterface

    from ..._compiler.cute.mma_support import CuteMmaSupport

if TYPE_CHECKING:
    from contextlib import ExitStack


@dataclasses.dataclass(frozen=True)
class _DeviceProperties:
    # No retained C++ property wrapper is reachable from this view. Unknown
    # non-primitive properties deliberately decline instead of invoking a getter.
    fields: tuple[tuple[str, str | int | bool | float], ...]

    def __getattr__(self, name: str) -> str | int | bool | float:
        for key, value in self.fields:
            if key == name:
                return value
        raise AttributeError(name)


class TargetMetadata:
    def __init__(
        self,
        torch: ModuleType,
        interface: type[CudaInterface],
        properties: torch._C._CudaDeviceProperties,
        support_modules: tuple[ModuleType, ...],
        support: CuteMmaSupport,
    ) -> None:
        self.torch = torch
        self.interface = interface
        self.properties = properties
        self._properties_owner = properties
        self.support_modules = support_modules
        self.support = support
        self.parent_pid = os.getpid()
        self.index = torch.cuda.current_device()
        self.available = torch.cuda.is_available()
        self.count = torch.cuda.device_count()
        self.capability = torch.cuda.get_device_capability(self.index)
        self.name = torch.cuda.get_device_name(self.index)
        assert self.available and 0 <= self.index < self.count
        assert (properties.major, properties.minor) == self.capability
        assert properties.name == self.name
        self.queries = Counter()
        self.property_values = self._property_values()
        self.frozen_properties = _DeviceProperties(
            tuple(
                (name, value)
                for name in self.property_values
                if type(value := getattr(properties, name)) in (str, int, bool, float)
            )
        )
        self._frozen_fields = self.frozen_properties.fields
        self.support_values = dataclasses.asdict(support)

    @classmethod
    def capture(cls, torch: ModuleType, device: object) -> TargetMetadata:
        """Resolve the installed support probes and selected device in parent."""
        from torch._dynamo.device_interface import CudaInterface

        modules = tuple(
            importlib.import_module(name)
            for name in (
                "helion._compiler.cute.mma_support",
                "helion._compiler.cute.chained_matmul",
                "helion._compiler.cute.cute_mma",
            )
        )
        original_support = modules[0].get_cute_mma_support
        assert all(m.get_cute_mma_support is original_support for m in modules)
        properties = torch.cuda.get_device_properties(device)
        support = original_support()
        result = cls(torch, CudaInterface, properties, modules, support)
        result._index(device)
        return result

    def _property_values(self) -> dict[str, object]:
        # Only the original parent may evaluate runtime-backed getters, even
        # after fork when a wrapper looks like ordinary metadata.
        assert os.getpid() == self.parent_pid
        values: dict[str, object] = {}
        for name in dir(self.properties):
            if name.startswith("_"):
                continue
            value = getattr(self.properties, name)
            if callable(value):
                raise TypeError("Unsupported callable device property: " + name)
            values[name] = (
                value if type(value) in (str, int, bool, float) else str(value)
            )
        return values

    def _index(self, device: object) -> int:
        if device is None:
            return self.index
        if type(device) is int:
            index = device
        elif isinstance(device, (str, self.torch.device)):
            parsed = self.torch.device(device)
            if parsed.type != "cuda":
                raise ValueError("Source target metadata requires CUDA")
            index = self.index if parsed.index is None else parsed.index
        else:
            raise TypeError("Unsupported source target device")
        if index != self.index:
            raise ValueError("Source target metadata is bound to one device")
        return index

    def _answer(self, name: str, value: object, device: object = None) -> object:
        self._index(device)
        self.queries[name] += 1
        return value

    def get_properties(self, device: object = None) -> object:
        return self._answer("properties", self.frozen_properties, device)

    def get_capability(self, device: object = None) -> object:
        return self._answer("capability", self.capability, device)

    def get_name(self, device: object = None) -> object:
        return self._answer("name", self.name, device)

    def current_device(self) -> object:
        return self._answer("current_device", self.index)

    def is_available(self) -> object:
        return self._answer("availability", self.available)

    def device_count(self) -> object:
        return self._answer("device_count", self.count)

    def get_support(self) -> object:
        return self._answer("mma_support", self.support)

    def check(self) -> None:
        assert self.properties is self._properties_owner
        assert self.frozen_properties.fields == self._frozen_fields
        if os.getpid() == self.parent_pid:
            assert self._property_values() == self.property_values
        assert dataclasses.asdict(self.support) == self.support_values

    def record(self) -> dict[str, object]:
        self.check()
        return {
            "visible_index": self.index,
            "availability_in_parent": self.available,
            "device_count_in_parent": self.count,
            "properties": self.property_values,
            "mma_support": self.support_values,
        }

    def install(self, stack: ExitStack) -> None:
        # Scope is compiler target metadata only. These are not initialized-state
        # answers: is_initialized and _is_in_bad_fork remain process-local.
        for name, value in (
            ("is_available", self.is_available),
            ("current_device", self.current_device),
            ("device_count", self.device_count),
            ("get_device_properties", self.get_properties),
            ("get_device_capability", self.get_capability),
            ("get_device_name", self.get_name),
        ):
            stack.enter_context(patch.object(self.torch.cuda, name, value))
        # Torch's device interface retains aliases from module import time;
        # patch these exact readonly bindings too, without changing its caches.
        for owner, entries in (
            (
                self.interface,
                (
                    ("current_device", self.current_device),
                    ("device_count", self.device_count),
                    ("get_device_properties", self.get_properties),
                ),
            ),
            (
                self.interface.Worker,
                (
                    ("current_device", self.current_device),
                    ("get_device_properties", self.get_properties),
                ),
            ),
        ):
            for name, value in entries:
                stack.enter_context(patch.object(owner, name, staticmethod(value)))
        for module in self.support_modules:
            stack.enter_context(
                patch.object(module, "get_cute_mma_support", self.get_support)
            )
