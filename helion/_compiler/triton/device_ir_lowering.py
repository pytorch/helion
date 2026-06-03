"""Triton-specific device IR lowering pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..device_ir_lowering import DeviceIRLowering
from ..device_ir_lowering import register_atomic_tunables
from ..device_ir_lowering import register_load_store_tunables
from ..device_ir_lowering import register_tensor_descriptor_layout_guards

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from ..host_function import HostFunction


class TritonDeviceIRLowering(DeviceIRLowering):
    """Triton-specific overrides for the device IR lowering pipeline.

    Adds Triton-specific tunable registration: load/store eviction
    policies and indexing types, atomic indexing types, and tensor
    descriptor layout guards.
    """

    def trace(self, device_ir: DeviceIR, func: HostFunction) -> None:
        super().trace(device_ir, func)

    def transform(self, device_ir: DeviceIR, func: HostFunction) -> None:
        super().transform(device_ir, func)

    def lower(self, device_ir: DeviceIR) -> None:
        super().lower(device_ir)

    def optimize(self, device_ir: DeviceIR) -> None:
        super().optimize(device_ir)

    def register(self, device_ir: DeviceIR) -> None:
        super().register(device_ir)
        register_load_store_tunables(device_ir)
        register_atomic_tunables(device_ir)
        register_tensor_descriptor_layout_guards(device_ir)
