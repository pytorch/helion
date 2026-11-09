"""The metadata-guarded, zero-marshalling relaunch of a compiled CuTe kernel.

:class:`CuteFastRelaunch` is the per-kernel state the launcher builds after a
kernel's first successful launch (``launcher._cute_build_fast_relaunch``
probes the DSL's marshalled argument slots): on a relaunch whose tensor
metadata, scalars, grid and block match, only the data pointers and the
stream are poked into the pre-packed C argument array before the compiled
entry point runs.  This module holds the relaunch state machine and its
guards (the one-call C++ tensor guard, the compiled metadata lambda, the
packed-argument copy); the probe itself stays with the launcher because it
reads the launcher's test seams.  Leaf module: imports ``arg_signature``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import logging
import math
import operator
import sys
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from .arg_signature import FastpathTensorGuard
    from .arg_signature import _CuteLastScalarArgGuard
    from .host_wrappers.mma_dag import MmaDagAliasGuard

log: logging.Logger = logging.getLogger(__name__)

FASTPATH_MISS: tuple[bool, None] = (False, None)
# Lowest value the fast relaunch probe treats as a host heap address it may
# dereference. Python's mmap-backed allocator places ctypes cells above this on
# 64-bit Linux, the only platform the CUDA-only fast path runs on; by-value
# integers such as tensor extents stay below it. macOS arm64 user space sits
# entirely below this floor, so the probe never dereferences there.
HOST_CELL_ADDRESS_FLOOR = 1 << 40


class CuteFastRelaunch:
    """Metadata-guarded, zero-marshalling relaunch of a compiled cute kernel.

    The CuTe DSL's per-call path (schema/arg cache key construction plus
    ``generate_execution_args`` marshalling: ctypes storage allocation and
    pointer extraction, ~25-45us of Python) recomputes almost exactly the
    same values on every call — for a given compiled kernel only the tensor
    DATA POINTERS and the CUDA stream can change between calls once the
    tensor metadata (dtype/shape/stride/device), the scalars, the grid, and
    the block are pinned.  Real workloads allocate fresh output tensors on
    every call, so pointer-keyed caches miss constantly; for O(30us)
    memory-bound kernels the marshalling then starves the GPU.

    This caches the marshalled ``exe_args`` once and per call only:

    1. checks metadata and compiled pointer alignment (no pointer equality),
    2. writes each tensor arg's ``data_ptr()`` into its probe-verified
       ``exe_args`` slot (tensor pointers marshal by value),
    3. refreshes the CUDA stream slot(s), and
    4. launches: a C-API ``JitExecutor`` is entered directly through its
       ``capi_func`` with a fastpath-owned copy of the packed ``void*``
       array it would build from ``exe_args`` on every call (the pointer
       and stream slots are written into that copy in steps 2-3), and its
       ``cuda_result`` is checked as ``run_compiled_program`` would; any
       other executor still goes through ``run_compiled_program``.

    Slot discovery is probe-verified at build time: tensor slots by
    re-marshalling with per-tensor shifted pointers, stream slots by
    re-marshalling with a different stream handle (handling both by-value
    slots and by-reference ctypes cells).  Anything unexpected falls back
    to the full DSL call path.

    The current stream is sampled fresh on every call via
    ``torch._C._cuda_getCurrentRawStream`` (same primitive the Triton
    launcher uses), so CUDA-graph capture streams are honored.
    """

    __slots__ = (
        "alias_guard",
        "alias_pick",
        "arg_count",
        "block",
        "by_ref_writers",
        "by_val_slots",
        "capi_func",
        "compile_options",
        "constexpr_flags",
        "cuda_error",
        "cuda_result",
        "cxx_check",
        "cxx_guards",
        "device_index",
        "exe_args",
        "executor",
        "grid",
        "keepalive",
        "last_raw",
        "lock",
        "packed_args",
        "pick_tensors",
        "pointer_indices",
        "pointer_masks",
        "pointer_writers",
        "pointer_slots",
        "raw_grid",
        "raw_kwargs",
        "scalar_fast",
        "scalar_guards",
        "tensor_guard",
        "tensor_guards",
        "tensor_slots",
    )

    def __init__(
        self,
        *,
        executor: object,
        exe_args: list[object],
        tensor_guards: tuple[FastpathTensorGuard, ...],
        scalar_guards: tuple[_CuteLastScalarArgGuard, ...],
        constexpr_flags: tuple[bool, ...],
        tensor_slots: tuple[tuple[int, int | None, object | None], ...],
        by_ref_writers: list[object],
        by_val_slots: list[int],
        arg_count: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        compile_options: str | None,
        device_index: int,
        last_raw: int,
        keepalive: tuple[object, ...],
        packed_launch: PackedLaunch | None = None,
        alias_guard: MmaDagAliasGuard | None = None,
        cxx_guards: object | None = None,
    ) -> None:
        self.executor = executor
        self.alias_guard = alias_guard
        # One C++ check of every tensor's dtype / device / shape / stride
        # (``cxx_tensor_guards``); the Python guards decide when it
        # misses (it also pins requires_grad and the dispatch keys).
        self.cxx_guards = cxx_guards
        self.cxx_check: Callable[..., bool] | None = (
            None if cxx_guards is None else cast("Any", cxx_guards).check
        )
        self.pick_tensors = tuple_picker(tuple(g[0] for g in tensor_guards))
        # Scalars: an exact (type, value) match; the normalizing guard
        # decides otherwise (Sym* values, NaN).
        self.scalar_fast = tuple(
            (
                guard.index,
                SCALAR_TYPES[guard.scalar_kind],
                float.fromhex(cast("str", guard.scalar_value))
                if guard.scalar_kind == "float"
                else guard.scalar_value,
            )
            for guard in scalar_guards
        )
        # The data pointers are gathered once, in ``tensor_slots`` order,
        # for the compiled alignment check, the alias guard and the patch.
        self.pointer_indices = tuple(index for index, _slot, _writer in tensor_slots)
        alignments = {g[0]: g[6] for g in tensor_guards}
        self.pointer_masks = tuple(
            alignments.get(index, 1) - 1 for index in self.pointer_indices
        )
        # The by-reference cells and the by-value slots the pointers patch, as
        # two plain lists paired with the gathered pointers.
        self.pointer_writers = tuple(
            (position, writer)
            for position, (_index, _slot, writer) in enumerate(tensor_slots)
            if writer is not None
        )
        self.pointer_slots = tuple(
            (position, slot)
            for position, (_index, slot, writer) in enumerate(tensor_slots)
            if writer is None
        )
        self.alias_pick: Callable[[Sequence[int]], tuple[int, ...]] | None = None
        if alias_guard is not None and all(
            index in self.pointer_indices for index in alias_guard.indices
        ):
            self.alias_pick = tuple_picker(
                tuple(self.pointer_indices.index(i) for i in alias_guard.indices)
            )
        # The launch as the generated host code spells it (grid tuple and
        # keyword arguments), recorded by the entry after the first launch
        # so the relaunch compares them without normalizing.
        self.raw_grid: tuple[int, ...] | None = None
        self.raw_kwargs: dict[str, object] | None = None
        self.exe_args = exe_args
        self.tensor_guards = tensor_guards
        self.tensor_guard = compile_fastpath_tensor_guard(tensor_guards)
        self.scalar_guards = scalar_guards
        self.constexpr_flags = constexpr_flags
        self.tensor_slots = tensor_slots
        self.by_ref_writers = by_ref_writers
        self.by_val_slots = by_val_slots
        self.arg_count = arg_count
        self.grid = grid
        self.block = block
        self.compile_options = compile_options
        self.device_index = device_index
        self.last_raw = last_raw
        self.keepalive = keepalive
        self.lock = threading.Lock()
        if packed_launch is None:
            self.packed_args = None
            self.capi_func = None
            self.cuda_result = None
            self.cuda_error = None
        else:
            self.packed_args = packed_launch.packed_args
            self.capi_func = packed_launch.capi_func
            self.cuda_result = packed_launch.cuda_result
            self.cuda_error = packed_launch.cuda_error

    def try_launch(
        self,
        args: tuple[object, ...],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        compile_options: str | None,
    ) -> tuple[bool, object]:
        if (
            len(args) != self.arg_count
            or grid != self.grid
            or block != self.block
            or compile_options != self.compile_options
        ):
            return FASTPATH_MISS
        return self.launch(args)

    def try_launch_raw(
        self,
        args: tuple[object, ...],
        grid: tuple[int, ...],
        kwargs: dict[str, object],
    ) -> tuple[bool, object]:
        """``try_launch`` on the launch as the host code spells it (before
        the entry normalizes grid, block and options)."""
        if (
            grid != self.raw_grid
            or kwargs != self.raw_kwargs
            or len(args) != self.arg_count
        ):
            return FASTPATH_MISS
        return self.launch(args)

    def launch(self, args: tuple[object, ...]) -> tuple[bool, object]:
        """Guards, then the relaunch, for a launch whose grid / block /
        options and argument count already matched."""
        # One C++ check of every tensor (``cxx_tensor_guards``); when it
        # is absent or misses, the compiled metadata guard (dtype, shape,
        # stride, device, compiled pointer alignment) decides.
        cxx_check = self.cxx_check
        if (
            cxx_check is None or not cxx_check(*self.pick_tensors(args))
        ) and not self.tensor_guard(args):
            return FASTPATH_MISS
        for position, (index, kind, value) in enumerate(self.scalar_fast):
            arg = args[index]
            if (
                type(arg) is not kind
                or arg != value
                # -0.0 == 0.0, but a float travels by value and is pinned in
                # the argument array: the sign of a zero must match too.
                or (
                    kind is float
                    and arg == 0.0
                    and math.copysign(1.0, arg)
                    != math.copysign(1.0, cast("float", value))
                )
            ) and not self.scalar_guards[position].matches(args, self.constexpr_flags):
                return FASTPATH_MISS
        # (No ``typing.cast`` on this path: it is a Python call per use, and
        # the pointer gather and the slot pokes run once per tensor.)
        pointers = [
            args[index].data_ptr()  # pyrefly: ignore[missing-attribute]
            for index in self.pointer_indices
        ]
        # The compiled schema's pointer alignment (no pointer equality).
        if any(map(operator.and_, pointers, self.pointer_masks)):
            return FASTPATH_MISS
        if self.alias_guard is not None:
            if self.alias_pick is not None:
                self.alias_guard.check_begins(self.alias_pick(pointers))
            else:
                self.alias_guard.check(args)
        raw = torch._C._cuda_getCurrentRawStream(self.device_index)
        with self.lock:
            # The slots are poked in the pre-packed C array of a CuTe DSL
            # ``JitExecutor``; an executor that is not one (test doubles,
            # profiling wrappers) has no such array and keeps packing
            # ``exe_args`` itself through ``run_compiled_program``.
            packed_args = self.packed_args
            slots: Any = self.exe_args if packed_args is None else packed_args
            for position, writer in self.pointer_writers:
                writer.value = pointers[position]  # pyrefly: ignore[missing-attribute]
            for position, slot in self.pointer_slots:
                slots[slot] = pointers[position]
            if raw != self.last_raw:
                for stream_writer in self.by_ref_writers:
                    stream_writer.value = raw  # pyrefly: ignore[missing-attribute]
                for stream_slot in self.by_val_slots:
                    slots[stream_slot] = raw
                self.last_raw = raw
            if packed_args is None:
                return (
                    True,
                    cast("Any", self.executor).run_compiled_program(self.exe_args),
                )
            capi_func: Any = self.capi_func
            try:
                capi_func(packed_args)
            except Exception as error:
                raise_direct_launch_crash(error)
            cuda_result: Any = self.cuda_result
            if cuda_result is None:
                return (True, None)
            error_code = cuda_result.value
            if error_code == 0:
                return (True, 0)
            raise cast("Any", self.cuda_error)(error_code)


SCALAR_TYPES: dict[str, type] = {"bool": bool, "int": int, "float": float}


def tuple_picker(
    indices: tuple[int, ...],
) -> Callable[[Sequence[Any]], tuple[Any, ...]]:
    """``lambda items: tuple(items[i] for i in indices)`` at C speed."""
    if len(indices) > 1:
        return cast(
            "Callable[[Sequence[Any]], tuple[Any, ...]]", operator.itemgetter(*indices)
        )
    if len(indices) == 1:
        (index,) = indices
        return lambda items: (items[index],)
    return lambda items: ()


def cxx_tensor_guards(
    args: tuple[object, ...],
    tensor_guards: Sequence[tuple[Any, ...]],
) -> object | None:
    """Dynamo's C++ tensor guard over the launch's tensor arguments: one
    call checks every tensor's type, dtype, device, requires_grad, dispatch
    keys, sizes and strides against this launch's.  None when the guard is
    unavailable or does not accept these tensors (the Python guards then
    decide alone)."""
    try:
        from torch._C._dynamo.guards import TensorGuards
    except ImportError:  # pragma: no cover - older torch
        return None
    tensors = [args[guard[0]] for guard in tensor_guards]
    if not tensors or not all(type(t) is torch.Tensor for t in tensors):
        return None
    try:
        # (The stub types the per-tensor lists as flat; the binding takes
        # one list of concrete sizes / strides per tensor.)
        guards = TensorGuards(
            *tensors,
            dynamic_dims_sizes=cast("Any", [list(guard[4]) for guard in tensor_guards]),
            dynamic_dims_strides=cast(
                "Any", [list(guard[5]) for guard in tensor_guards]
            ),
        )
        if not guards.check(*tensors):
            return None
    except Exception:
        log.debug("CuTe fastpath: C++ tensor guards unavailable", exc_info=True)
        return None
    return guards


@dataclass(frozen=True)
class PackedLaunch:
    """A private copy of the DSL executor's packed C argument array.

    ``JitExecutor.run_compiled_program`` re-packs every ``exe_args`` entry into
    a thread-local ``c_void_p`` array on each call (type dispatch plus a
    ``ctypes.c_void_p(...)`` round trip per entry, ~3us for a flash kernel).
    Only the tensor-pointer and stream slots change between fast relaunches,
    so the array is packed once and those slots are poked in place before the
    C entry point is invoked directly.  The copy is private: the executor's
    thread-local buffer is rewritten by any slow-path launch of the same
    kernel, which would otherwise clobber the cached scalar slots.
    """

    packed_args: object
    capi_func: object
    cuda_result: object | None
    cuda_error: object


def packed_launch(executor: object, exe_args: list[object]) -> object | None:
    """Pre-pack ``exe_args`` for a real CuTe DSL ``JitExecutor``; else ``None``.

    The DSL modules are looked up rather than imported: a real executor can
    only exist once ``cutlass.base_dsl.jit_executor`` is loaded, and test
    doubles must not pull the DSL in.
    """
    jit_executor = sys.modules.get("cutlass.base_dsl.jit_executor")
    if jit_executor is None or not isinstance(executor, jit_executor.JitExecutor):
        return None
    try:
        create_cuda_runtime_error = sys.modules[
            "cutlass.base_dsl.runtime.cuda"
        ].create_cuda_runtime_error
        source = executor._get_invoke_packed_args(list(exe_args))
        total = len(exe_args) + executor._num_extra_args
        packed_args = (ctypes.c_void_p * total)()
        ctypes.memmove(packed_args, source, ctypes.sizeof(ctypes.c_void_p) * total)
        return PackedLaunch(
            packed_args=packed_args,
            capi_func=executor.capi_func,
            cuda_result=executor.cuda_result,
            cuda_error=create_cuda_runtime_error,
        )
    except Exception:
        # A packing-internals change in cutlass gives up only the direct C
        # entry; the fastpath keeps the ``run_compiled_program`` launch.
        log.debug("CuTe direct C-API launch unavailable", exc_info=True)
        return None


def compile_fastpath_tensor_guard(
    tensor_guards: tuple[FastpathTensorGuard, ...],
) -> Callable[[tuple[object, ...]], bool]:
    """Fold the per-tensor metadata checks into one straight-line expression.

    Same facts as the tuple form (dtype, shape, stride, device, compiled
    pointer alignment); evaluating them as a single generated lambda avoids
    the per-tensor loop and tuple unpacking on every relaunch.
    """
    namespace: dict[str, object] = {"Tensor": torch.Tensor}
    checks: list[str] = []
    for (
        index,
        device_type,
        device_index,
        dtype,
        shape,
        stride,
        alignment,
    ) in tensor_guards:
        namespace[f"dtype_{index}"] = dtype
        namespace[f"shape_{index}"] = torch.Size(shape)
        namespace[f"stride_{index}"] = tuple(stride)
        namespace[f"device_{index}"] = (
            torch.device(device_type)
            if device_index is None
            else torch.device(device_type, device_index)
        )
        tensor = f"t{index}"
        item = [
            f"isinstance(({tensor} := args[{index}]), Tensor)",
            f"{tensor}.dtype is dtype_{index}",
            f"{tensor}.size() == shape_{index}",
            f"{tensor}.stride() == stride_{index}",
            f"{tensor}.device == device_{index}",
        ]
        if alignment > 1:
            item.append(f"{tensor}.data_ptr() % {int(alignment)} == 0")
        checks.append(f"({' and '.join(item)})")
    return eval(f"lambda args: {' and '.join(checks) if checks else 'True'}", namespace)


def raise_direct_launch_crash(error: Exception) -> NoReturn:
    """Wrap a C-API entry failure as ``run_compiled_program`` does."""
    from cutlass.base_dsl.common import DSLRuntimeError

    raise DSLRuntimeError("Runtime Crash", cause=error)
