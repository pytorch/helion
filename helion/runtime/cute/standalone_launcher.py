"""Native launcher for concretely exported CuTe kernels; no Helion dependency."""

from __future__ import annotations

import math
import shlex
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

# pyrefly: ignore [missing-import]
import cuda.bindings.driver as _standalone_cuda_driver

# pyrefly: ignore [missing-import]
from cuda.bindings.driver import CUstream  # noqa: F401  # Emitted wrapper annotation.
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cutlass.torch as cutlass_torch
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

# Frozen copies of ``compile_cache.FULL_RANGE_DIVISION_COMPILE_OPTION`` and of
# the two division options a launch may name (a test keeps the two launchers'
# option handling equal).
_STANDALONE_DIVISION_OPTION = "nvvm-options=-prec-div=0"
_STANDALONE_DIVISION_OPTIONS = (
    _STANDALONE_DIVISION_OPTION,
    "nvvm-options=-prec-div=1",
)
_STANDALONE_LAUNCHES: dict[str, tuple[Any, ...]] = {}
_STANDALONE_COMPILED: dict[tuple[str, int], Any] = {}
# A frozen copy of ``helion._compiler.cute.dtypes.CUTLASS_POINTER_DTYPE_NAME``
# keyed by ``str(dtype)`` (this module stays Helion-free; a test keeps them equal).
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
    "torch.float8_e8m0fnu": cutlass.Float8E8M0FNU,
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
        expected = entry[-1]
        if arg != expected and not (
            isinstance(arg, float)
            and isinstance(expected, float)
            and math.isnan(arg)
            and math.isnan(expected)
        ):
            raise ValueError("Argument does not match the exported CuTe constant")
    elif kind == "device":
        if not isinstance(arg, torch.device) or str(arg) != entry[1]:
            raise ValueError("Argument does not match the exported CuTe device")
    elif kind in ("tuple", "list"):
        if not isinstance(arg, tuple if kind == "tuple" else list) or len(arg) != len(
            entry[1]
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


# The compiler's persistent vec-alignment classifier keys pointer residues
# modulo this many bytes (``_CUTE_ALIGNMENT_MAX_BYTES`` in
# helion/_compiler/cute/memory_ops.py); the exported residues are its.
_STANDALONE_ALIGNMENT_BYTES = 32


def _standalone_validate_alignment(
    args: Sequence[torch.Tensor], residues: tuple[int, ...]
) -> None:
    if any(
        tensor.data_ptr() % _STANDALONE_ALIGNMENT_BYTES != residue
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


def _standalone_validate_mma_dag_aliases(
    args: Sequence[object], indices: tuple[int, ...], stored: tuple[bool, ...]
) -> None:
    """The cute_mma_dag alias guard: the byte span of every tensor argument
    the kernel stores to is disjoint from every other tensor argument's
    (the kernel was compiled against distinct storages and reads them as
    such; two loads may alias)."""
    spans: list[tuple[int, int] | None] = []
    for index in indices:
        tensor = args[index]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Argument does not match the exported CuTe tensor layout")
        if tensor.numel() == 0:
            spans.append(None)
            continue
        last = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        begin = tensor.data_ptr()
        spans.append((begin, begin + (last + 1) * tensor.element_size()))
    for store, is_store in enumerate(stored):
        span = spans[store]
        if not is_store or span is None:
            continue
        for other in (*spans[:store], *spans[store + 1 :]):
            if other is not None and other[0] < span[1] and span[0] < other[1]:
                raise ValueError(
                    "Tensor arguments alias; the exported CuTe kernel reads and "
                    "writes them as distinct tensors"
                )


# Looked up once: every launch checks the context, the fast relaunch included.
_STANDALONE_CUDA_SUCCESS = _standalone_cuda_driver.CUresult.CUDA_SUCCESS
_standalone_cu_ctx_get_current = _standalone_cuda_driver.cuCtxGetCurrent


def _standalone_ensure_cuda_context() -> None:
    """Make the current device's primary CUDA context current on this thread,
    before every launch: the compiled host code of a launch that sets function
    attributes aborts without one, and a thread can lack one at any launch."""
    # Helion's own launcher runs this check too, and its
    # helion.runtime.cute.launcher._ensure_cuda_context says why.
    success = _STANDALONE_CUDA_SUCCESS
    error, context = _standalone_cu_ctx_get_current()
    if error == success and int(context) == 0:
        error, device = _standalone_cuda_driver.cuDeviceGet(torch.cuda.current_device())
        if error == success:
            error, context = _standalone_cuda_driver.cuDevicePrimaryCtxRetain(device)
        if error == success:
            (error,) = _standalone_cuda_driver.cuCtxSetCurrent(context)
    if error != success:
        raise RuntimeError(f"No CUDA context for a CuTe launch: {error}")


def _standalone_compile(
    wrapper: object, launch_args: Sequence[object], options: str | None
) -> Callable[..., object]:
    """Compile under the options Helion's launcher compiles with
    (``helion.runtime.cute.compile_cache.merge_default_compile_options`` and
    ``compile_with_options``; a test keeps the defaults equal): the FFI
    bridge, and fp32 ``/`` as ``div.full.f32`` unless the kernel chose the
    IEEE division.  The DSL's option string has no spelling for a libNVVM
    option, so those go in as raw pass options, next to the DSL's option
    objects for the options an exported kernel launches with."""
    from cutlass.base_dsl.compiler import ExtraCompilerOpts

    tokens = shlex.split(options or "")
    if "--enable-tvm-ffi" not in tokens:
        tokens.append("--enable-tvm-ffi")
    if not any(token in _STANDALONE_DIVISION_OPTIONS for token in tokens):
        tokens.append(_STANDALONE_DIVISION_OPTION)
    compile_options: list[object] = []
    division = []
    remaining = iter(tokens)
    for token in remaining:
        if token in _STANDALONE_DIVISION_OPTIONS:
            division.append(token)
        elif token == "--enable-tvm-ffi":
            compile_options.append(cute.EnableTVMFFI(True))
        elif token == "--generate-line-info":
            compile_options.append(cute.GenerateLineInfo(True))
        elif token.startswith("--ptxas-options="):
            compile_options.append(cute.PtxasOptions(token.partition("=")[2]))
        elif token == "--opt-level":
            level = next(remaining, "")
            if not level.isdecimal():
                raise ValueError(
                    f"CuTe compile option --opt-level needs a level: {options!r}"
                )
            compile_options.append(cute.OptLevel(int(level)))
        else:
            raise ValueError(f"Unsupported CuTe compile option: {token!r}")
    if len(division) > 1:
        raise ValueError(f"More than one division option: {options!r}")
    compile_options.append(ExtraCompilerOpts(" ".join(division)))
    return cute.compile[tuple(compile_options)](wrapper, *launch_args)


def _default_cute_launcher(
    kernel: object, grid: tuple[int, ...], *args: object, **kwargs: object
) -> None:
    kernel_name = cast("Any", kernel).__name__
    schema, wrapper, block, options, alias = _STANDALONE_LAUNCHES[kernel_name]
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
    if alias is not None:
        _standalone_validate_mma_dag_aliases(args, *alias)
    if len(devices) != 1:
        raise ValueError("Standalone CuTe launch tensors must share one CUDA device")
    device = devices.pop()
    launch_args.extend(grid)
    with torch.cuda.device(device):
        _standalone_ensure_cuda_context()
        launch_args.append(cutlass_torch.current_stream())
        key = (kernel_name, torch.cuda.current_device())
        compiled = _STANDALONE_COMPILED.get(key)
        if compiled is None:
            compiled = _standalone_compile(wrapper, launch_args, options)
            _STANDALONE_COMPILED[key] = compiled
        compiled(*launch_args)
