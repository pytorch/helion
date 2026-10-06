"""Native launcher for concretely exported CuTe kernels; no Helion dependency."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

# pyrefly: ignore [missing-import]
from cuda.bindings.driver import CUstream  # noqa: F401  # Emitted wrapper annotation.
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cutlass.torch as cutlass_torch
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

_STANDALONE_LAUNCHES: dict[str, tuple[Any, ...]] = {}
_STANDALONE_COMPILED: dict[tuple[str, int], Any] = {}
_STANDALONE_DTYPES = {
    "torch.float16": cutlass.Float16,
    "torch.bfloat16": cutlass.BFloat16,
    "torch.float32": cutlass.Float32,
    "torch.float64": cutlass.Float64,
    "torch.int8": cutlass.Int8,
    "torch.int16": cutlass.Int16,
    "torch.int32": cutlass.Int32,
    "torch.int64": cutlass.Int64,
    "torch.uint8": cutlass.Uint8,
    "torch.uint32": cutlass.Uint32,
    "torch.uint64": cutlass.Int64,
    "torch.bool": cutlass.Uint8,
    "torch.float8_e4m3fn": cutlass.Float8E4M3FN,
    "torch.float8_e5m2": cutlass.Float8E5M2,
    "torch.float4_e2m1fn_x2": cutlass.Uint8,
}


def _standalone_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    return max(
        1, torch.cuda.get_device_properties(device).multi_processor_count - reserved_sms
    )


helion = types.SimpleNamespace(
    runtime=types.SimpleNamespace(get_num_sm=_standalone_num_sm)
)


def _standalone_validate(arg: object, entry: tuple[Any, ...]) -> None:
    kind = entry[0]
    if kind == "tensor":
        _, dtype, rank, shape, strides, alignment = entry
        if (
            not isinstance(arg, torch.Tensor)
            or not arg.is_cuda
            or str(arg.dtype) != dtype
            or tuple(arg.shape) != shape
            or tuple(arg.stride()) != strides
            or arg.data_ptr() % alignment
        ):
            raise ValueError("Argument does not match the exported CuTe tensor layout")
    elif kind in ("constant", "scalar_constexpr"):
        if arg != entry[-1]:
            raise ValueError("Argument does not match the exported CuTe constant")
    elif kind == "device":
        if not isinstance(arg, torch.device) or str(arg) != entry[1]:
            raise ValueError("Argument does not match the exported CuTe device")
    elif kind in ("tuple", "list"):
        if (
            not isinstance(arg, (tuple, list))
            or type(arg).__name__ != kind
            or len(arg) != len(entry[1])
        ):
            raise ValueError("Argument does not match the exported CuTe container")
        _standalone_validate_args(arg, entry[1])
    elif kind == "dict":
        if not isinstance(arg, dict) or set(arg) != {key for key, _ in entry[1]}:
            raise ValueError("Argument does not match the exported CuTe dictionary")
        for key, value in entry[1]:
            _standalone_validate(arg[key], value)
    elif type(arg).__name__ != entry[1]:
        raise ValueError("Argument does not match the exported CuTe scalar type")


def _standalone_validate_args(
    args: Sequence[object], schema: Sequence[tuple[Any, ...]]
) -> None:
    for entry, arg in zip(schema, args, strict=True):
        _standalone_validate(arg, entry)


def _standalone_validate_alignment(
    args: Sequence[torch.Tensor], residues: tuple[int, ...]
) -> None:
    if any(
        tensor.data_ptr() % 16 != residue
        for tensor, residue in zip(args, residues, strict=True)
    ):
        raise ValueError(
            "Input pointer alignment does not match the exported CuTe workload"
        )


def _standalone_validate_aliases(
    args: Sequence[torch.Tensor], expected: tuple[bool, ...]
) -> None:
    spans = []
    for tensor in args:
        storage = tensor.untyped_storage()
        start = storage.data_ptr()
        spans.append((tensor.device, start, start + storage.nbytes()))
    disjoint = tuple(
        left[1] == left[2]
        or right[1] == right[2]
        or left[0] != right[0]
        or left[2] <= right[1]
        or right[2] <= left[1]
        for index, left in enumerate(spans)
        for right in spans[index + 1 :]
    )
    if disjoint != expected:
        raise ValueError("Tensor aliasing does not match the exported CuTe workload")


def _default_cute_launcher(
    kernel: object, grid: tuple[int, ...], *args: object, **kwargs: object
) -> None:
    kernel_name = cast("Any", kernel).__name__
    schema, wrapper, block, options = _STANDALONE_LAUNCHES[kernel_name]
    requested_block = tuple(cast("tuple[int, ...]", kwargs.get("block", (256, 1, 1))))
    requested_block = (*requested_block, *(1 for _ in range(3 - len(requested_block))))
    if requested_block != block or kwargs.get("cute_compile_options") != options:
        raise ValueError("Launch options do not match the exported CuTe wrapper")
    grid = (*grid, *(1 for _ in range(3 - len(grid))))
    if any(value <= 0 for value in grid):
        return
    launch_args: list[object] = []
    devices = set()
    for entry, arg in zip(schema, args, strict=True):
        _standalone_validate(arg, entry)
        kind = entry[0]
        if kind == "tensor":
            assert isinstance(arg, torch.Tensor)
            _, dtype, rank, shape, strides, alignment = entry
            devices.add(arg.device)
            launch_args.append(
                make_ptr(
                    _STANDALONE_DTYPES[dtype],
                    arg.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=alignment,
                )
            )
        elif kind != "scalar_constexpr":
            launch_args.append(arg)
    if len(devices) != 1:
        raise ValueError("Standalone CuTe launch tensors must share one CUDA device")
    device = devices.pop()
    launch_args.extend(grid)
    with torch.cuda.device(device):
        launch_args.append(cutlass_torch.current_stream())
        key = (kernel_name, torch.cuda.current_device())
        compiled = _STANDALONE_COMPILED.get(key)
        if compiled is None:
            compile_options = options or ""
            if "--enable-tvm-ffi" not in compile_options:
                compile_options = (compile_options + " --enable-tvm-ffi").strip()
            compiled = cute.compile(wrapper, *launch_args, options=compile_options)
            _STANDALONE_COMPILED[key] = compiled
        compiled(*launch_args)
