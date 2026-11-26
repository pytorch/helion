"""Check the exact CUDA launch that a multi-block-per-SM barrier would issue."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING
from typing import Any

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Callable

from cuda.bindings import driver  # pyrefly: ignore[missing-module-attribute]


def _checked(rc: tuple[Any, ...]) -> tuple[Any, ...]:
    if rc[0] != driver.CUresult.CUDA_SUCCESS:
        status, message = driver.cuGetErrorString(rc[0])
        detail = (
            message.decode(errors="replace")
            if status == driver.CUresult.CUDA_SUCCESS and message
            else str(rc[0])
        )
        raise RuntimeError(f"CUDA cooperative residency query failed: {detail}")
    return rc[1:]


def stream_resources(stream: object) -> tuple[int, int]:
    """Require an ordinary full-device stream in the probing thread's context.

    Even on a cache hit, a green-context stream may target a different subset
    of SMs on the same CUDA device. A probe stream created in the thread's
    context cannot attest its CUfunction or its SM assignment.
    """
    (current,) = _checked(driver.cuCtxGetCurrent())
    context, green = _checked(driver.cuStreamGetCtx_v2(stream))
    if (
        not int(current)
        or int(context) != int(current)
        or (green is not None and int(green))
    ):
        raise exc.InvalidConfig(
            "checked cooperative grid requires a regular stream in the current CUDA context"
        )
    (device,) = _checked(driver.cuCtxGetDevice())
    (sms,) = _checked(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
        )
    )
    (resources,) = _checked(
        driver.cuCtxGetDevResource(
            context, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    if not sms or int(resources.sm.smCount) != sms:
        raise exc.InvalidConfig(
            "checked cooperative grid requires the ordinary full SM allocation"
        )
    # Context IDs cannot be recycled as context handles can be.
    (identity,) = _checked(driver.cuCtxGetId(context))
    return int(identity), int(sms)


def check_cooperative_residency(
    compiled: Callable[..., object], args: tuple[object, ...], sms: int
) -> None:
    """Capture the actual launch once without running it or replaying the graph.

    The node supplies its own CUfunction, dynamic shared memory, block shape
    and cooperative bit even on a reload. An occupancy guess based on the DSL
    source (or a register limit alone) is insufficient for a spinning barrier.
    The generic wrapper receives grid_x/y/z separately, uses them only in
    its launch grid, and passes none to the device as a runtime argument.
    Capture a single block so CUDA can provide this node even for a request
    larger than the residency limit. The probe graph is never submitted.
    CUDA still admits the real cooperative launch under concurrent workloads.
    """
    width, height, depth = args[-4:-1]
    if type(width) is not int or width <= 0 or (height, depth) != (1, 1):
        raise RuntimeError("Expected a positive one-dimensional cooperative grid")
    (device,) = _checked(driver.cuCtxGetDevice())
    (cooperative,) = _checked(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device
        )
    )
    if not cooperative:
        raise exc.InvalidConfig(
            "the executing device does not support cooperative launch"
        )
    (capture,) = _checked(driver.cuStreamIsCapturing(args[-1]))
    if capture != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        raise exc.InvalidConfig(
            "Warm up the checked cooperative grid before graph capture, including autotuner-selected grids"
        )
    (stream,) = _checked(
        driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
    )
    graph = None
    try:
        _checked(
            driver.cuStreamBeginCapture(
                stream, driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            )
        )
        try:
            compiled(*args[:-4], 1, 1, 1, stream)
        finally:
            (graph,) = _checked(driver.cuStreamEndCapture(stream))
        _, count = _checked(driver.cuGraphGetNodes(graph, 0))
        if count != 1:
            raise RuntimeError("Expected a single cooperative CUDA kernel")
        nodes, _ = _checked(driver.cuGraphGetNodes(graph, 1))
        (params,) = _checked(driver.cuGraphKernelNodeGetParams(nodes[0]))
        (attribute,) = _checked(
            driver.cuGraphKernelNodeGetAttribute(
                nodes[0], driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
            )
        )
        grid = (params.gridDimX, params.gridDimY, params.gridDimZ)
        if not attribute.cooperative or grid != (1, 1, 1):
            raise RuntimeError("Expected the actual one-dimensional cooperative grid")
        handle = params.func
        if not int(handle):
            (handle,) = _checked(driver.cuKernelGetFunction(params.kern))
        threads = prod((params.blockDimX, params.blockDimY, params.blockDimZ))
        (blocks,) = _checked(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                handle, threads, params.sharedMemBytes
            )
        )
        if width > blocks * sms:
            raise exc.InvalidConfig(
                f"Cooperative grid of {width} blocks exceeds actual residency "
                f"of {blocks} blocks per SM on {sms} SMs"
            )
    finally:
        try:
            if graph is not None:
                _checked(driver.cuGraphDestroy(graph))
        finally:
            _checked(driver.cuStreamDestroy(stream))


def stride_from_schema(schema: tuple[tuple[object, ...], ...], index: object) -> int:
    """Read the actual grid-stride constexpr from this executable's specialization."""
    if type(index) is int and 0 <= index < len(schema):
        entry = schema[index]
        if len(entry) == 4 and entry[:2] == ("scalar_constexpr", "int"):
            value = entry[3]
            if type(value) is int and value > 0:
                return value
    raise exc.InvalidConfig(
        "checked cooperative grid requires an integer phase-stride specialization"
    )


class ProvenCooperativeGrids:
    """Context-local admission of one grid on one compiled function."""

    def __init__(self, expected_grid_x: int) -> None:
        self.width = expected_grid_x
        # A racing first compilation can produce two distinct loaded host
        # callables. Retain the exact object so its Python id cannot be reused.
        self.proven: dict[tuple[int, int, int], Callable[..., object]] = {}

    def check(self, compiled: Callable[..., object], args: tuple[object, ...]) -> None:
        if args[-4:-1] != (self.width, 1, 1):
            raise exc.InvalidConfig(
                "physical grid must match the specialized phase stride"
            )
        context, sms = stream_resources(args[-1])
        key = (id(compiled), context, sms)
        if self.proven.get(key) is not compiled:
            check_cooperative_residency(compiled, args, sms)
            self.proven[key] = compiled
