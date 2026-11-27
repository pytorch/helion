"""Single paired-kernel effect adapter for guarded capture PDL overlap."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any
from typing import cast

import torch

from .capture_pdl import CaptureGuard
from .capture_pdl import LaunchEffects
from .capture_pdl import StorageRange
from .capture_pdl_cuda import CaptureSnapshot
from .capture_pdl_cuda import CudaCaptureDriver
from .capture_pdl_cuda import finish_capture_launch
from .launch_caches import LAST_LAUNCH_CACHE_ATTRIBUTE
from .launch_caches import track_cuda_graph_cache_entry

_LOCAL = threading.local()


@dataclass(frozen=True)
class PendingCaptureLaunch:
    driver: CudaCaptureDriver
    guard: CaptureGuard
    before: CaptureSnapshot
    effects: LaunchEffects
    parameter_sizes: tuple[int, ...]
    device: torch.device
    # The launch device's SM count for a kernel that reads its head start
    # from the wait-mode argument, else zero.
    head_start_sms: int = 0

    def finish(self) -> None:
        # Both snapshots and the parameter update must use the launch context,
        # even when the caller restored a different current device after enqueue.
        with torch.cuda.device(self.device):
            finish_capture_launch(
                self.driver,
                self.guard,
                self.before,
                self.effects,
                self.parameter_sizes,
                head_start_sms=self.head_start_sms,
            )
        nodes = self.guard.captures.get(self.before.key)
        if nodes is not None:
            # Owned cute_cuda_graph scopes release this metadata at graph reset.
            # Raw torch.cuda.graph scopes retain only the guard's bounded maps;
            # no tensor/descriptor allocation is kept alive by the proof state.
            track_cuda_graph_cache_entry(
                self.guard, "captures", self.before.key, nodes, ()
            )


def _launch_stream_capturing(tensor: torch.Tensor) -> bool:
    """Whether the current stream of ``tensor``'s device is capturing a CUDA
    graph.

    Query the actual stream rather than a Python capture flag: raw
    torch.cuda.graph scopes are supported, including on a non-current device.
    Only that case pays for switching the device.
    """
    index = tensor.get_device()
    if index == torch.cuda.current_device():
        return torch.cuda.is_current_stream_capturing()
    with torch.cuda.device(index):
        return torch.cuda.is_current_stream_capturing()


def prepare_capture_launch(
    kernel: object, args: tuple[object, ...], grid: tuple[int, ...]
) -> PendingCaptureLaunch | None:
    """Only a warmed, fully audited single-kernel wrapper may enter the guard."""
    plans = getattr(cast("Any", kernel), "_helion_cute_wrapper_plans", ())
    if (
        len(plans) != 2
        or plans[0].get("kind") != "helion_paged_attention"
        or plans[0].get("capture_pdl_contract") != "paired-output-stores-v1"
        or plans[1].get("kind") != "helion_mma_dag_alias"
    ):
        return None
    indices = plans[1]["tensor_idx"]
    if not indices:
        return None
    first = args[indices[0]]
    if not isinstance(first, torch.Tensor) or not first.is_cuda:
        return None
    # Eager relaunches leave here. The complete proof is capture-only and an
    # ordinary launch retains its early wait, but every launch of a kernel
    # with the contract comes through this function, so the capture test
    # precedes the remaining validation, the storage walk and the graph
    # inspection.
    if not _launch_stream_capturing(first):
        return None
    if (
        getattr(cast("Any", kernel), "_helion_cute_use_pdl", False) is not True
        or getattr(cast("Any", kernel), LAST_LAUNCH_CACHE_ATTRIBUTE, None) is None
    ):
        return None
    # The ordinary launcher elides empty grids. Without this gate, another
    # thread's one new node could be mistaken for our nonexistent launch.
    # Decline symbolic/custom grid values rather than evaluating them twice.
    if not 1 <= len(grid) <= 3 or any(
        type(size) is not int or size <= 0 for size in grid
    ):
        return None
    suffix = plans[0].get("capture_pdl_parameter_sizes")
    if (
        not isinstance(suffix, tuple)
        or len(suffix) < 2
        or any(type(size) is not int for size in suffix)
        or suffix[-1] != 4
        or any(size != 128 for size in suffix[:-1])
    ):
        return None
    stored = plans[1]["stored"]
    loaded = plans[1]["capture_loaded"]
    if len(indices) != len(stored) or len(indices) != len(loaded):
        return None
    reads: list[StorageRange] = []
    writes: list[StorageRange] = []
    device: torch.device | None = None
    for index, is_store, is_load in zip(indices, stored, loaded, strict=True):
        tensor = args[index]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return None
        if device is None:
            device = tensor.device
        elif device != tensor.device:
            return None
        storage = tensor.untyped_storage()
        begin = storage.data_ptr()
        assert device.index is not None
        interval = StorageRange(device.index, begin, begin + storage.nbytes())
        if is_store:
            writes.append(interval)
        if is_load:
            reads.append(interval)
    if device is None:
        return None
    # Static tensor layouts lower to one pointer argument. Unbaked layouts
    # have a structured CuTe ABI and are deliberately outside this guarded route.
    last = getattr(cast("Any", kernel), LAST_LAUNCH_CACHE_ATTRIBUTE)
    parameter_sizes: list[int] = []
    for entry in last.launch.schema:
        if entry[0] == "tensor" and len(entry) in (5, 6):
            parameter_sizes.append(8)
        elif entry[0] == "scalar":
            parameter_sizes.append({"bool": 1, "int": 8, "float": 4}[entry[1]])
        elif entry[0] != "scalar_constexpr":
            return None
    # The compiler proves this suffix from the final wrapper-only signature;
    # the CUDA adapter also checks each actual parameter size before copying.
    parameter_sizes.extend(suffix)
    # This routine only runs on an already initialized/warmed CUDA path.
    if not hasattr(_LOCAL, "driver"):
        try:
            _LOCAL.driver = CudaCaptureDriver()
        except (AttributeError, OSError):
            # Older/missing versioned driver entry points cannot establish the
            # edge-data/ABI proof. They retain the ordinary early wait.
            _LOCAL.driver = None
        _LOCAL.guard = CaptureGuard()
    driver: CudaCaptureDriver | None = _LOCAL.driver
    if driver is None:
        return None
    guard: CaptureGuard = _LOCAL.guard
    with torch.cuda.device(device):
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        before = driver.snapshot(stream)
    if before is None:
        return None
    head_start_sms = 0
    if plans[0].get("capture_pdl_head_start") is True:
        # The boundary prefetch of a one-wave grid: its first CTAs serve the
        # units that wait for the SMs of the launch ahead.
        head_start_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return PendingCaptureLaunch(
        driver,
        guard,
        before,
        LaunchEffects(tuple(reads), tuple(writes)),
        tuple(parameter_sizes),
        device,
        head_start_sms,
    )
