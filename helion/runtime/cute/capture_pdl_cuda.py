"""Capture-only CUDA adapter for the audited PDL wait-mode kernel ABI.

CUDA bindings releases predating the v3 capture API omit dependency edge data,
so this adapter uses the versioned driver entry points directly. Importing it
does not load the driver or initialize CUDA. Unsupported queries leave the
captured kernel's default EARLY mode untouched.

The before/after whole-graph snapshots establish which node this launch added,
including concurrent submissions on other streams. Their aggregate capture-time
cost is quadratic in graph size, bounded by MAX_GRAPH_NODES. This cost is
paid only for opted-in capture launches, never during graph replay or eager
execution; avoiding it requires another equally strong node-identity proof.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import threading
from typing import Protocol

from .capture_pdl import CaptureGuard
from .capture_pdl import CaptureKey
from .capture_pdl import Dependency
from .capture_pdl import LaunchEffects
from .capture_pdl import WaitMode


@dataclass(frozen=True)
class CaptureSnapshot:
    key: CaptureKey
    nodes: frozenset[int]


class CaptureDriver(Protocol):
    def snapshot(self, stream: int) -> CaptureSnapshot | None: ...

    def kernel_dependencies(self, node: int) -> tuple[Dependency, ...] | None: ...

    def set_wait_mode(
        self, node: int, mode: WaitMode, parameter_sizes: tuple[int, ...]
    ) -> bool: ...


def finish_capture_launch(
    driver: CaptureDriver,
    guard: CaptureGuard,
    before: CaptureSnapshot,
    effects: LaunchEffects,
    parameter_sizes: tuple[int, ...],
) -> WaitMode:
    """Patch only a uniquely identified new kernel; otherwise retain EARLY.

    The caller's warmed, audited wrapper enqueues exactly one kernel, with a
    final dynamic Int32 wait-mode parameter initialized to zero. Comparing the
    whole graph's node sets also detects concurrent submissions to other streams
    in this capture. No decision relies on a pre-enqueue stream-tail snapshot.
    """
    after = driver.snapshot(before.key.stream)
    if after is None or after.key != before.key or not before.nodes <= after.nodes:
        return WaitMode.EARLY
    added = after.nodes - before.nodes
    if len(added) != 1:
        return WaitMode.EARLY
    (node,) = added
    dependencies = driver.kernel_dependencies(node)
    if dependencies is None:
        return WaitMode.EARLY
    decision = guard.decide(before.key, node, dependencies, effects)
    if decision.mode == WaitMode.BEFORE_WRITES and not driver.set_wait_mode(
        node, decision.mode, parameter_sizes
    ):
        # Never install proof state for a failed or unavailable ABI patch.
        return WaitMode.EARLY
    guard.record(before.key, node, decision)
    return decision.mode


class _Edge(ctypes.Structure):
    _fields_ = [
        ("from_port", ctypes.c_ubyte),
        ("to_port", ctypes.c_ubyte),
        ("kind", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 5),
    ]


class _KernelParams(ctypes.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v2, including the library-kernel/context fields.
    _fields_ = [
        ("func", ctypes.c_void_p),
        ("grid_x", ctypes.c_uint),
        ("grid_y", ctypes.c_uint),
        ("grid_z", ctypes.c_uint),
        ("block_x", ctypes.c_uint),
        ("block_y", ctypes.c_uint),
        ("block_z", ctypes.c_uint),
        ("shared_bytes", ctypes.c_uint),
        ("parameters", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("kernel", ctypes.c_void_p),
        ("context", ctypes.c_void_p),
    ]


class CudaCaptureDriver:
    """Versioned CUDA calls, constructed only on an initialized CUDA path."""

    MAX_GRAPH_NODES = 4096

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
            "cuStreamGetCaptureInfo_v3": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
                ctypes.POINTER(ctypes.POINTER(_Edge)),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cuGraphGetNodes": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cuGraphNodeGetType": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
            "cuGraphNodeGetDependencies_v2": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_Edge),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cuGraphKernelNodeGetParams_v2": [
                ctypes.c_void_p,
                ctypes.POINTER(_KernelParams),
            ],
            "cuGraphKernelNodeSetParams_v2": [
                ctypes.c_void_p,
                ctypes.POINTER(_KernelParams),
            ],
            "cuKernelGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            "cuFuncGetParamInfo": [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    def snapshot(self, stream: int) -> CaptureSnapshot | None:
        context = ctypes.c_void_p()
        if self.library.cuCtxGetCurrent(ctypes.byref(context)) or not context.value:
            return None
        status = ctypes.c_int()
        capture_id = ctypes.c_uint64()
        graph = ctypes.c_void_p()
        dependencies = ctypes.POINTER(ctypes.c_void_p)()
        edges = ctypes.POINTER(_Edge)()
        count = ctypes.c_size_t()
        result = self.library.cuStreamGetCaptureInfo_v3(
            stream,
            ctypes.byref(status),
            ctypes.byref(capture_id),
            ctypes.byref(graph),
            ctypes.byref(dependencies),
            ctypes.byref(edges),
            ctypes.byref(count),
        )
        if result or status.value != 1 or not graph.value:
            return None
        # CaptureInfo's borrowed arrays expire at the next stream API. Consume
        # their edge restrictions immediately; actual-node edges are read anew
        # after enqueue, so these borrowed addresses are never retained.
        for index in range(count.value):
            edge = edges[index]
            if not Dependency(
                dependencies[index],
                edge.from_port,
                edge.to_port,
                edge.kind,
                bytes(edge.reserved),
            ).supported():
                return None
        count = ctypes.c_size_t()
        if self.library.cuGraphGetNodes(graph, None, ctypes.byref(count)):
            return None
        if count.value > self.MAX_GRAPH_NODES:
            return None
        nodes = (ctypes.c_void_p * count.value)()
        capacity = count.value
        if capacity:
            if self.library.cuGraphGetNodes(graph, nodes, ctypes.byref(count)):
                return None
            if count.value > capacity:
                return None
        return CaptureSnapshot(
            CaptureKey(
                context.value,
                threading.get_ident(),
                stream,
                capture_id.value,
                graph.value,
            ),
            frozenset(node for node in nodes[: count.value] if node is not None),
        )

    def kernel_dependencies(self, node: int) -> tuple[Dependency, ...] | None:
        kind = ctypes.c_int()
        if self.library.cuGraphNodeGetType(node, ctypes.byref(kind)) or kind.value != 0:
            return None
        count = ctypes.c_size_t()
        if self.library.cuGraphNodeGetDependencies_v2(
            node, None, None, ctypes.byref(count)
        ):
            return None
        if count.value > self.MAX_GRAPH_NODES:
            return None
        if count.value == 0:
            return ()
        dependencies = (ctypes.c_void_p * count.value)()
        edges = (_Edge * count.value)()
        capacity = count.value
        if (
            self.library.cuGraphNodeGetDependencies_v2(
                node, dependencies, edges, ctypes.byref(count)
            )
            or count.value > capacity
        ):
            return None
        return tuple(
            Dependency(
                dependencies[index],
                edges[index].from_port,
                edges[index].to_port,
                edges[index].kind,
                bytes(edges[index].reserved),
            )
            for index in range(count.value)
        )

    def set_wait_mode(
        self, node: int, mode: WaitMode, parameter_sizes: tuple[int, ...]
    ) -> bool:
        # The warmed wrapper schema and compiler's final argument contract
        # supply the exact ABI. Do not discover its length by intentionally
        # issuing failing driver queries while capture is active.
        if not parameter_sizes or parameter_sizes[-1] != 4:
            return False
        params = _KernelParams()
        if self.library.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params)):
            return False
        function = ctypes.c_void_p(params.func)
        if not function.value:
            if not params.kernel or self.library.cuKernelGetFunction(
                ctypes.byref(function), params.kernel
            ):
                return False
        offsets: list[tuple[int, int]] = []
        for index, expected in enumerate(parameter_sizes):
            offset, size = ctypes.c_size_t(), ctypes.c_size_t()
            result = self.library.cuFuncGetParamInfo(
                function, index, ctypes.byref(offset), ctypes.byref(size)
            )
            if result or size.value != expected:
                return False
            offsets.append((offset.value, size.value))
        mode_value = ctypes.c_int32(mode)
        if params.parameters:
            pointers = tuple(params.parameters[: len(offsets)])
            if any(pointer is None for pointer in pointers) or params.extra:
                return False
            if ctypes.c_int32.from_address(pointers[-1]).value != 0:
                return False
            # GetParams' arrays AND argument bytes belong to the node and expire
            # when SetParams modifies it. Copy both levels before that call.
            values = [
                ctypes.create_string_buffer(ctypes.string_at(pointer, size), size)
                for pointer, (_offset, size) in zip(pointers, offsets, strict=True)
            ]
            ctypes.memmove(values[-1], ctypes.byref(mode_value), 4)
            copied = (ctypes.c_void_p * len(values))(
                *(ctypes.addressof(value) for value in values)
            )
            params.parameters = copied
            result = self.library.cuGraphKernelNodeSetParams_v2(
                node, ctypes.byref(params)
            )
            if result:
                # An actual mutation failure cannot be silently treated as a
                # proved EARLY node: abort the capture instead of replaying it.
                raise RuntimeError(f"CUDA graph wait-mode update failed: {result}")
            return True
        # This adapter declines packed launch buffers. The capture
        # remains correct with its original mode; add this ABI only with a
        # separate byte-layout proof if the installed CuTe uses it.
        return False
