"""Runtime bridge for the CUTLASS grouped GEMM reference path.

Generated Helion grouped GEMM lowering must not import this module or the
trimmed CUTLASS kernel it loads. This benchmark bridge exists because
the Blackwell grouped GEMM comparisons use it as an opt-in CUTLASS CuTeDSL
comparison/reference launch for heterogeneous NT groups.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import Sequence
from typing import cast
import weakref

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class _PreparedExecutor(Protocol):
    def run_compiled_program(self, exe_args: list[object]) -> int | None: ...


class _CompiledCuteFunction(Protocol):
    _default_executor: _PreparedExecutor | None

    def to(self, device: object = None) -> _PreparedExecutor: ...

    def generate_execution_args(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[list[object], list[object]]: ...


@dataclass
class _PointerBuffers:
    host: torch.Tensor
    cute_tensor: object
    torch_tensor: torch.Tensor
    event: torch.cuda.Event | None = None
    event_stream: int | None = None
    addresses: tuple[int, ...] | None = None


@dataclass
class _CompiledGroupedKernel:
    initial: tuple[object, object, object]
    dim_tensor: object
    dim_torch: torch.Tensor
    stride_tensor: object
    stride_torch: torch.Tensor
    tensormap: object
    tensormap_torch: torch.Tensor
    pointer_pool: list[_PointerBuffers]
    compiled: Callable[..., object]


@dataclass
class _PreparedExecution:
    executor: _PreparedExecutor
    exe_args: list[object]
    adapted_args: list[object]
    stream: object


@dataclass
class _PreparedBlackwellGroupedLaunch:
    compiled: _CompiledGroupedKernel
    pointer_buffers: _PointerBuffers
    tensor_refs: tuple[weakref.ReferenceType[torch.Tensor], ...]
    stream_executions: OrderedDict[int, _PreparedExecution]
    captured: bool = False


@dataclass
class _LastBlackwellPreparedLaunch:
    key: tuple[object, ...]
    prepared_ref: weakref.ReferenceType[_PreparedBlackwellGroupedLaunch]
    tensor_versions: tuple[int, ...]
    tensor_addresses: tuple[int, ...]


@dataclass
class _RetiredCompiledGroupedKernel:
    compiled: _CompiledGroupedKernel
    events: tuple[torch.cuda.Event, ...]


class BlackwellGroupedGemmProblem(NamedTuple):
    m: int
    n: int
    k: int
    l_mode: int = 1


class _BlackwellGroupedGemmMetadata(NamedTuple):
    device: torch.device
    problems: tuple[BlackwellGroupedGemmProblem, ...]
    strides_abc: tuple[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        ...,
    ]
    a_dtype: type[cutlass.Numeric]
    c_dtype: type[cutlass.Numeric]
    a_mode0_major: bool
    b_mode0_major: bool
    c_mode0_major: bool


_BLACKWELL_GROUPED_COMPILED_CACHE_MAX = 8
_BLACKWELL_GROUPED_COMPILED_CACHE: OrderedDict[
    tuple[object, ...],
    _CompiledGroupedKernel,
] = OrderedDict()
_BLACKWELL_GROUPED_RETIRED_COMPILED: list[_RetiredCompiledGroupedKernel] = []
_BLACKWELL_GROUPED_CAPTURED_GRAPH_LAUNCHES: OrderedDict[
    tuple[int, int],
    tuple[_CompiledGroupedKernel, _PointerBuffers],
] = OrderedDict()
_BLACKWELL_GROUPED_PREPARED_CACHE_MAX = 32
_BLACKWELL_GROUPED_PREPARED_CACHE: OrderedDict[
    tuple[object, ...],
    _PreparedBlackwellGroupedLaunch,
] = OrderedDict()
_BLACKWELL_GROUPED_RETIRED_PREPARED: list[_PreparedBlackwellGroupedLaunch] = []
_BLACKWELL_GROUPED_LAST_PREPARED: _LastBlackwellPreparedLaunch | None = None
_CUTLASS_GROUPED_REFERENCE_MODULE: object | None = None


def _cutlass_grouped_reference() -> object:
    """Load the trimmed CUTLASS reference kernel only for the bridge path."""
    global _CUTLASS_GROUPED_REFERENCE_MODULE
    if _CUTLASS_GROUPED_REFERENCE_MODULE is None:
        from . import _cutlass_grouped_gemm_kernel

        _CUTLASS_GROUPED_REFERENCE_MODULE = _cutlass_grouped_gemm_kernel
    return _CUTLASS_GROUPED_REFERENCE_MODULE


def _new_pointer_buffers(num_segments: int, device: torch.device) -> _PointerBuffers:
    host = torch.empty((num_segments, 3), dtype=torch.int64, pin_memory=True)
    torch_tensor = torch.empty((num_segments, 3), device=device, dtype=torch.int64)
    cute_tensor = cutlass_torch.from_dlpack(torch_tensor, assumed_align=16)
    cute_tensor.element_type = cutlass.Int64
    return _PointerBuffers(host, cute_tensor, torch_tensor)


def _cuda_graph_capture_active() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _retain_blackwell_grouped_captured_launch(
    compiled: _CompiledGroupedKernel,
    buffers: _PointerBuffers,
) -> None:
    # CUDA graph replay needs the compiled launcher and pointer metadata alive
    # after Python returns from capture. There is no Python-visible lifetime
    # hook for every graph that may replay this launch, so eviction is unsafe.
    key = (id(compiled), id(buffers))
    _BLACKWELL_GROUPED_CAPTURED_GRAPH_LAUNCHES[key] = (compiled, buffers)
    _BLACKWELL_GROUPED_CAPTURED_GRAPH_LAUNCHES.move_to_end(key)


def _acquire_pointer_buffers_from_pool(
    compiled: _CompiledGroupedKernel,
    *,
    num_entries: int,
    device: torch.device,
) -> tuple[_PointerBuffers, bool]:
    if _cuda_graph_capture_active():
        buffers = _new_pointer_buffers(num_entries, device)
        _retain_blackwell_grouped_captured_launch(compiled, buffers)
        return buffers, True

    for buffers in compiled.pointer_pool:
        if buffers.event is None or buffers.event.query():
            return buffers, False

    buffers = _new_pointer_buffers(num_entries, device)
    compiled.pointer_pool.append(buffers)
    return buffers, False


def _record_pointer_buffers_use(
    buffers: _PointerBuffers,
    *,
    graph_capture: bool,
    stream: torch.cuda.Stream,
) -> None:
    if graph_capture:
        return
    if buffers.event is None:
        buffers.event = torch.cuda.Event()
    buffers.event.record(stream)
    buffers.event_stream = int(stream.cuda_stream)


def _record_tensor_use(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    if tensor.is_cuda:
        tensor.record_stream(stream)


def _retire_finished_compiled_entries(
    retired_entries: list[_RetiredCompiledGroupedKernel],
) -> None:
    if not retired_entries:
        return
    retired_entries[:] = [
        retired
        for retired in retired_entries
        if not all(event.query() for event in retired.events)
    ]


def _retire_compiled_entry(
    compiled: _CompiledGroupedKernel,
    retired_entries: list[_RetiredCompiledGroupedKernel],
) -> None:
    events = tuple(
        event
        for buffers in compiled.pointer_pool
        if (event := buffers.event) is not None and not event.query()
    )
    if events:
        retired_entries.append(
            _RetiredCompiledGroupedKernel(compiled=compiled, events=events)
        )


def _torch_to_cutlass_grouped_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"unsupported grouped GEMM dtype {dtype}")


def _major_mode(
    tensor: torch.Tensor,
    *,
    mode0: str,
    mode1: str,
    tensor_name: str,
    group: int,
) -> str:
    if int(tensor.stride(1)) == 1:
        return mode1
    if int(tensor.stride(0)) == 1:
        return mode0
    raise ValueError(
        f"{tensor_name}[{group}] must have a contiguous {mode0} or {mode1} mode"
    )


def _require_same_major(
    actual: str,
    expected: str,
    *,
    tensor_name: str,
    group: int,
) -> None:
    if actual != expected:
        raise ValueError(
            f"{tensor_name}[{group}] has {actual}-major layout, expected "
            f"{expected}-major to match the other groups"
        )


def _check_16b_alignment(
    tensor: torch.Tensor,
    cutlass_dtype: type[cutlass.Numeric],
    *,
    major: str,
    mode0: str,
    tensor_name: str,
    group: int,
) -> None:
    if int(tensor.data_ptr()) % 16 != 0:
        raise ValueError(f"{tensor_name}[{group}] data pointer must be 16-byte aligned")
    major_dim = 0 if major == mode0 else 1
    elements_per_16b = 16 * 8 // cutlass_dtype.width
    if int(tensor.size(major_dim)) % elements_per_16b != 0:
        raise ValueError(
            f"{tensor_name}[{group}] contiguous dimension must be 16-byte "
            f"aligned for {cutlass_dtype}"
        )
    leading_stride_bytes = int(tensor.stride(1 - major_dim)) * tensor.element_size()
    if leading_stride_bytes % 16 != 0:
        raise ValueError(
            f"{tensor_name}[{group}] leading stride must be 16-byte aligned "
            f"for {cutlass_dtype}"
        )


def _check_strides_fit_int32(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    group: int,
) -> None:
    for stride in tensor.stride():
        if stride < 0 or stride > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"{tensor_name}[{group}] strides must be non-negative int32 values"
            )


def _validate_blackwell_grouped_config(
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    *,
    use_2cta_instrs: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    mma_tiler = tuple(int(dim) for dim in mma_tiler_mn)
    cluster_shape = tuple(int(dim) for dim in cluster_shape_mn)
    if len(mma_tiler) != 2:
        raise ValueError("mma_tiler_mn must contain exactly two values")
    if len(cluster_shape) != 2:
        raise ValueError("cluster_shape_mn must contain exactly two values")
    mma_m, mma_n = mma_tiler
    if not (
        (not use_2cta_instrs and mma_m in (64, 128))
        or (use_2cta_instrs and mma_m in (128, 256))
    ):
        raise ValueError(f"invalid grouped GEMM MMA tiler M {mma_m}")
    if mma_n not in range(32, 257, 32):
        raise ValueError(f"invalid grouped GEMM MMA tiler N {mma_n}")
    if cluster_shape[0] % (2 if use_2cta_instrs else 1) != 0:
        raise ValueError("cluster_shape_m must align with use_2cta_instrs")

    def is_power_of_2(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    if (
        cluster_shape[0] * cluster_shape[1] > 16
        or not is_power_of_2(cluster_shape[0])
        or not is_power_of_2(cluster_shape[1])
    ):
        raise ValueError(f"invalid grouped GEMM cluster shape {cluster_shape}")
    return (mma_tiler[0], mma_tiler[1]), (cluster_shape[0], cluster_shape[1])


def _blackwell_grouped_gemm_nt_metadata(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
) -> _BlackwellGroupedGemmMetadata:
    if not a_groups:
        raise ValueError("grouped GEMM requires at least one group")
    if len(a_groups) != len(b_groups) or len(a_groups) != len(out_groups):
        raise ValueError("A, B, and output group counts must match")

    first_a = a_groups[0]
    first_b = b_groups[0]
    first_out = out_groups[0]
    if first_a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "grouped GEMM A/B dtype must be torch.float16 or torch.bfloat16"
        )
    if first_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("grouped GEMM output dtype must be fp16, bf16, or fp32")
    if first_a.device.type != "cuda":
        raise ValueError("grouped GEMM tensors must be on CUDA")
    if first_b.device != first_a.device or first_out.device != first_a.device:
        raise ValueError("all grouped GEMM tensors must be on the same CUDA device")

    a_dtype = _torch_to_cutlass_grouped_dtype(first_a.dtype)
    c_dtype = _torch_to_cutlass_grouped_dtype(first_out.dtype)
    a_major = _major_mode(
        first_a,
        mode0="m",
        mode1="k",
        tensor_name="A",
        group=0,
    )
    b_major = _major_mode(
        first_b,
        mode0="n",
        mode1="k",
        tensor_name="B",
        group=0,
    )
    c_major = _major_mode(
        first_out,
        mode0="m",
        mode1="n",
        tensor_name="out",
        group=0,
    )

    problems: list[BlackwellGroupedGemmProblem] = []
    strides_abc: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for group, (a, b, out) in enumerate(
        zip(a_groups, b_groups, out_groups, strict=True)
    ):
        if a.ndim != 2:
            raise ValueError(f"A[{group}] must have shape [M, K]")
        if b.ndim != 2:
            raise ValueError(f"B[{group}] must have shape [N, K]")
        if out.ndim != 2:
            raise ValueError(f"out[{group}] must have shape [M, N]")
        if a.dtype != first_a.dtype or b.dtype != first_a.dtype:
            raise ValueError("all grouped GEMM A/B tensors must share one dtype")
        if out.dtype != first_out.dtype:
            raise ValueError("all grouped GEMM output tensors must share one dtype")
        if b.device != first_a.device or a.device != first_a.device:
            raise ValueError("all grouped GEMM A/B tensors must be on one device")
        if out.device != first_a.device:
            raise ValueError("all grouped GEMM output tensors must be on one device")

        m = int(a.size(0))
        k = int(a.size(1))
        n = int(b.size(0))
        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError("grouped GEMM problem dimensions must be positive")
        if int(b.size(1)) != k:
            raise ValueError(f"K dimension mismatch for group {group}")
        if tuple(int(size) for size in out.shape) != (m, n):
            raise ValueError(f"out[{group}] must have shape [{m}, {n}]")

        group_a_major = _major_mode(
            a,
            mode0="m",
            mode1="k",
            tensor_name="A",
            group=group,
        )
        group_b_major = _major_mode(
            b,
            mode0="n",
            mode1="k",
            tensor_name="B",
            group=group,
        )
        group_c_major = _major_mode(
            out,
            mode0="m",
            mode1="n",
            tensor_name="out",
            group=group,
        )
        _require_same_major(group_a_major, a_major, tensor_name="A", group=group)
        _require_same_major(group_b_major, b_major, tensor_name="B", group=group)
        _require_same_major(group_c_major, c_major, tensor_name="out", group=group)
        _check_16b_alignment(
            a,
            a_dtype,
            major=group_a_major,
            mode0="m",
            tensor_name="A",
            group=group,
        )
        _check_16b_alignment(
            b,
            a_dtype,
            major=group_b_major,
            mode0="n",
            tensor_name="B",
            group=group,
        )
        _check_16b_alignment(
            out,
            c_dtype,
            major=group_c_major,
            mode0="m",
            tensor_name="out",
            group=group,
        )
        _check_strides_fit_int32(a, tensor_name="A", group=group)
        _check_strides_fit_int32(b, tensor_name="B", group=group)
        _check_strides_fit_int32(out, tensor_name="out", group=group)
        problems.append(BlackwellGroupedGemmProblem(m, n, k, 1))
        strides_abc.append(
            (
                (int(a.stride(0)), int(a.stride(1))),
                (int(b.stride(0)), int(b.stride(1))),
                (int(out.stride(0)), int(out.stride(1))),
            )
        )

    return _BlackwellGroupedGemmMetadata(
        device=first_a.device,
        problems=tuple(problems),
        strides_abc=tuple(strides_abc),
        a_dtype=a_dtype,
        c_dtype=c_dtype,
        a_mode0_major=a_major == "m",
        b_mode0_major=b_major == "n",
        c_mode0_major=c_major == "m",
    )


def _compute_blackwell_total_clusters(
    problems: tuple[BlackwellGroupedGemmProblem, ...],
    *,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
) -> int:
    cta_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
    cta_n = mma_tiler_mn[1]
    cluster_m = cta_m * cluster_shape_mn[0]
    cluster_n = cta_n * cluster_shape_mn[1]
    return sum(
        ((problem.m + cluster_m - 1) // cluster_m)
        * ((problem.n + cluster_n - 1) // cluster_n)
        for problem in problems
    )


def _blackwell_grouped_initial_tensor(
    dtype: type[cutlass.Numeric],
    *,
    mode0_major: bool,
) -> object:
    min_size = 16 * 8 // dtype.width
    cutlass_grouped = cast("Any", _cutlass_grouped_reference())
    return cutlass_grouped.create_tensor_and_stride(
        1,
        min_size,
        min_size,
        mode0_major,
        dtype,
    )[2]


def _blackwell_grouped_cache_key(
    metadata: _BlackwellGroupedGemmMetadata,
    *,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
) -> tuple[object, ...]:
    return (
        int(metadata.device.index or 0),
        metadata.problems,
        metadata.strides_abc,
        metadata.a_dtype,
        metadata.c_dtype,
        metadata.a_mode0_major,
        metadata.b_mode0_major,
        metadata.c_mode0_major,
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
    )


def _compile_blackwell_grouped_kernel(
    metadata: _BlackwellGroupedGemmMetadata,
    *,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
) -> _CompiledGroupedKernel:
    from helion.runtime import _ensure_cute_dsl_arch_env

    _ensure_cute_dsl_arch_env(())
    initial = (
        _blackwell_grouped_initial_tensor(
            metadata.a_dtype,
            mode0_major=metadata.a_mode0_major,
        ),
        _blackwell_grouped_initial_tensor(
            metadata.a_dtype,
            mode0_major=metadata.b_mode0_major,
        ),
        _blackwell_grouped_initial_tensor(
            metadata.c_dtype,
            mode0_major=metadata.c_mode0_major,
        ),
    )
    dim_tensor, dim_torch = cast(
        "tuple[object, torch.Tensor]",
        cutlass_torch.cute_tensor_like(
            cast("Any", torch.tensor(metadata.problems, dtype=torch.int32)),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        ),
    )
    stride_tensor, stride_torch = cast(
        "tuple[object, torch.Tensor]",
        cutlass_torch.cute_tensor_like(
            cast("Any", torch.tensor(metadata.strides_abc, dtype=torch.int32)),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        ),
    )

    hardware = cutlass_utils.HardwareInfo()
    sm_count = hardware.get_max_active_clusters(1)
    max_active_clusters = hardware.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    cutlass_grouped = cast("Any", _cutlass_grouped_reference())
    tensormap_shape = (
        sm_count,
        cutlass_grouped.GroupedGemmKernel.num_tensormaps,
        cutlass_grouped.GroupedGemmKernel.bytes_per_tensormap // 8,
    )
    tensormap, tensormap_torch = cast(
        "tuple[object, torch.Tensor]",
        cutlass_torch.cute_tensor_like(
            cast("Any", torch.empty(tensormap_shape, dtype=torch.int64)),
            cutlass.Int64,
            is_dynamic_layout=False,
        ),
    )

    pointer_buffers = _new_pointer_buffers(len(metadata.problems), metadata.device)
    grouped_kernel = cutlass_grouped.GroupedGemmKernel(
        cutlass.Float32,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        cutlass_utils.TensorMapUpdateMode.SMEM,
    )
    total_clusters = _compute_blackwell_total_clusters(
        metadata.problems,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    compiled = cute.compile(
        grouped_kernel,
        initial[0],
        initial[1],
        initial[2],
        len(metadata.problems),
        dim_tensor,
        stride_tensor,
        pointer_buffers.cute_tensor,
        total_clusters,
        tensormap,
        max_active_clusters,
        cutlass_torch.default_stream(),
        options="--opt-level 2",
    )
    return _CompiledGroupedKernel(
        initial=initial,
        dim_tensor=dim_tensor,
        dim_torch=dim_torch,
        stride_tensor=stride_tensor,
        stride_torch=stride_torch,
        tensormap=tensormap,
        tensormap_torch=tensormap_torch,
        pointer_pool=[],
        compiled=compiled,
    )


def _retire_finished_blackwell_grouped_compiled() -> None:
    _retire_finished_compiled_entries(_BLACKWELL_GROUPED_RETIRED_COMPILED)


def _retire_blackwell_grouped_compiled(
    compiled: _CompiledGroupedKernel,
) -> None:
    _retire_compiled_entry(compiled, _BLACKWELL_GROUPED_RETIRED_COMPILED)


def _get_compiled_blackwell_grouped_kernel(
    metadata: _BlackwellGroupedGemmMetadata,
    *,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
) -> _CompiledGroupedKernel:
    _retire_finished_blackwell_grouped_compiled()
    key = _blackwell_grouped_cache_key(
        metadata,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    cached = _BLACKWELL_GROUPED_COMPILED_CACHE.get(key)
    if cached is not None:
        _BLACKWELL_GROUPED_COMPILED_CACHE.move_to_end(key)
        return cached

    compiled = _compile_blackwell_grouped_kernel(
        metadata,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    _BLACKWELL_GROUPED_COMPILED_CACHE[key] = compiled
    _BLACKWELL_GROUPED_COMPILED_CACHE.move_to_end(key)
    while (
        len(_BLACKWELL_GROUPED_COMPILED_CACHE) > _BLACKWELL_GROUPED_COMPILED_CACHE_MAX
    ):
        evicted = _BLACKWELL_GROUPED_COMPILED_CACHE.popitem(last=False)[1]
        _retire_blackwell_grouped_compiled(evicted)
    return compiled


def _acquire_blackwell_pointer_buffers(
    compiled: _CompiledGroupedKernel,
    *,
    num_groups: int,
    device: torch.device,
) -> tuple[_PointerBuffers, bool]:
    return _acquire_pointer_buffers_from_pool(
        compiled,
        num_entries=num_groups,
        device=device,
    )


def _update_blackwell_pointer_buffers(
    buffers: _PointerBuffers,
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
) -> None:
    addresses: list[int] = []
    for a, b, out in zip(a_groups, b_groups, out_groups, strict=True):
        addresses.extend((int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr())))
    current_addresses = tuple(addresses)
    if buffers.addresses == current_addresses:
        return
    for index, offset in enumerate(range(0, len(addresses), 3)):
        buffers.host[index, 0] = addresses[offset]
        buffers.host[index, 1] = addresses[offset + 1]
        buffers.host[index, 2] = addresses[offset + 2]
    buffers.addresses = current_addresses
    buffers.torch_tensor.copy_(buffers.host, non_blocking=True)


def _blackwell_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        int(tensor.device.index or 0),
    )


def _blackwell_prepared_cache_key(
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
    *,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
) -> tuple[object, ...]:
    return (
        tuple(_blackwell_tensor_signature(tensor) for tensor in a_groups),
        tuple(_blackwell_tensor_signature(tensor) for tensor in b_groups),
        tuple(_blackwell_tensor_signature(tensor) for tensor in out_groups),
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
    )


def _blackwell_prepared_tensor_versions(
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> tuple[int, ...]:
    return tuple(
        int(tensor._version)
        for tensors in (a_groups, b_groups, out_groups)
        for tensor in tensors
    )


def _blackwell_prepared_tensor_addresses(
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> tuple[int, ...]:
    return tuple(
        int(tensor.data_ptr())
        for tensors in (a_groups, b_groups, out_groups)
        for tensor in tensors
    )


def _get_last_blackwell_prepared_launch(
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
    *,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
) -> _PreparedBlackwellGroupedLaunch | None:
    last = _BLACKWELL_GROUPED_LAST_PREPARED
    if last is None:
        return None
    key = last.key
    prepared = last.prepared_ref()
    if prepared is None:
        return None
    if (
        key[3] != mma_tiler_mn
        or key[4] != cluster_shape_mn
        or key[5] != use_2cta_instrs
        or not _blackwell_prepared_refs_match(
            prepared,
            a_groups,
            b_groups,
            out_groups,
        )
        or _blackwell_prepared_tensor_versions(a_groups, b_groups, out_groups)
        != last.tensor_versions
        or _blackwell_prepared_tensor_addresses(a_groups, b_groups, out_groups)
        != last.tensor_addresses
        or _BLACKWELL_GROUPED_PREPARED_CACHE.get(key) is not prepared
    ):
        return None
    _BLACKWELL_GROUPED_PREPARED_CACHE.move_to_end(key)
    return prepared


def _blackwell_prepared_tensor_refs(
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> tuple[weakref.ReferenceType[torch.Tensor], ...]:
    return tuple(weakref.ref(tensor) for tensor in (*a_groups, *b_groups, *out_groups))


def _blackwell_prepared_refs_match(
    prepared: _PreparedBlackwellGroupedLaunch,
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> bool:
    offset = 0
    tensor_refs = prepared.tensor_refs
    for tensors in (a_groups, b_groups, out_groups):
        for tensor in tensors:
            if offset >= len(tensor_refs) or tensor_refs[offset]() is not tensor:
                return False
            offset += 1
    return offset == len(tensor_refs)


def _set_last_blackwell_prepared_launch(
    key: tuple[object, ...],
    prepared: _PreparedBlackwellGroupedLaunch,
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> None:
    global _BLACKWELL_GROUPED_LAST_PREPARED
    _BLACKWELL_GROUPED_LAST_PREPARED = _LastBlackwellPreparedLaunch(
        key=key,
        prepared_ref=weakref.ref(prepared),
        tensor_versions=_blackwell_prepared_tensor_versions(
            a_groups,
            b_groups,
            out_groups,
        ),
        tensor_addresses=_blackwell_prepared_tensor_addresses(
            a_groups,
            b_groups,
            out_groups,
        ),
    )


def _retire_finished_blackwell_prepared_launches() -> None:
    if not _BLACKWELL_GROUPED_RETIRED_PREPARED or _cuda_graph_capture_active():
        return
    _BLACKWELL_GROUPED_RETIRED_PREPARED[:] = [
        prepared
        for prepared in _BLACKWELL_GROUPED_RETIRED_PREPARED
        if ((event := prepared.pointer_buffers.event) is not None and not event.query())
    ]


def _retire_blackwell_prepared_launch(
    prepared: _PreparedBlackwellGroupedLaunch,
) -> None:
    event = prepared.pointer_buffers.event
    if _cuda_graph_capture_active() or (event is not None and not event.query()):
        _BLACKWELL_GROUPED_RETIRED_PREPARED.append(prepared)


def _get_cached_blackwell_prepared_launch(
    key: tuple[object, ...],
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> _PreparedBlackwellGroupedLaunch | None:
    global _BLACKWELL_GROUPED_LAST_PREPARED
    _retire_finished_blackwell_prepared_launches()
    prepared = _BLACKWELL_GROUPED_PREPARED_CACHE.get(key)
    if prepared is None:
        return None
    if not _blackwell_prepared_refs_match(prepared, a_groups, b_groups, out_groups):
        del _BLACKWELL_GROUPED_PREPARED_CACHE[key]
        if (
            _BLACKWELL_GROUPED_LAST_PREPARED is not None
            and _BLACKWELL_GROUPED_LAST_PREPARED.key == key
        ):
            _BLACKWELL_GROUPED_LAST_PREPARED = None
        _retire_blackwell_prepared_launch(prepared)
        return None
    _BLACKWELL_GROUPED_PREPARED_CACHE.move_to_end(key)
    _set_last_blackwell_prepared_launch(
        key,
        prepared,
        a_groups,
        b_groups,
        out_groups,
    )
    return prepared


def _cache_blackwell_prepared_launch(
    key: tuple[object, ...],
    prepared: _PreparedBlackwellGroupedLaunch,
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> None:
    global _BLACKWELL_GROUPED_LAST_PREPARED
    _BLACKWELL_GROUPED_PREPARED_CACHE[key] = prepared
    _BLACKWELL_GROUPED_PREPARED_CACHE.move_to_end(key)
    _set_last_blackwell_prepared_launch(
        key,
        prepared,
        a_groups,
        b_groups,
        out_groups,
    )
    while (
        len(_BLACKWELL_GROUPED_PREPARED_CACHE) > _BLACKWELL_GROUPED_PREPARED_CACHE_MAX
    ):
        _old_key, evicted = _BLACKWELL_GROUPED_PREPARED_CACHE.popitem(last=False)
        if (
            _BLACKWELL_GROUPED_LAST_PREPARED is not None
            and _BLACKWELL_GROUPED_LAST_PREPARED.key == _old_key
        ):
            _BLACKWELL_GROUPED_LAST_PREPARED = None
        _retire_blackwell_prepared_launch(evicted)


def _get_blackwell_prepared_executor(
    compiled: _CompiledGroupedKernel,
) -> _PreparedExecutor:
    jit_func = cast("_CompiledCuteFunction", compiled.compiled)
    executor = jit_func._default_executor
    if executor is None:
        executor = jit_func.to(None)
        jit_func._default_executor = executor
    return executor


def _new_blackwell_prepared_execution(
    compiled: _CompiledGroupedKernel,
    buffers: _PointerBuffers,
) -> _PreparedExecution:
    stream = cutlass_torch.current_stream()
    jit_func = cast("_CompiledCuteFunction", compiled.compiled)
    exe_args, adapted_args = jit_func.generate_execution_args(
        compiled.initial[0],
        compiled.initial[1],
        compiled.initial[2],
        compiled.dim_tensor,
        compiled.stride_tensor,
        buffers.cute_tensor,
        compiled.tensormap,
        stream,
    )
    return _PreparedExecution(
        executor=_get_blackwell_prepared_executor(compiled),
        exe_args=exe_args,
        adapted_args=adapted_args,
        stream=stream,
    )


def _launch_blackwell_prepared(
    prepared: _PreparedBlackwellGroupedLaunch,
    a_groups: tuple[torch.Tensor, ...],
    b_groups: tuple[torch.Tensor, ...],
    out_groups: tuple[torch.Tensor, ...],
) -> None:
    torch_stream = torch.cuda.current_stream()
    stream_key = int(torch_stream.cuda_stream)
    buffers = prepared.pointer_buffers
    graph_capture = _cuda_graph_capture_active()
    if (
        not graph_capture
        and buffers.event is not None
        and buffers.event_stream != stream_key
        and not buffers.event.query()
    ):
        torch_stream.wait_event(buffers.event)

    execution = prepared.stream_executions.get(stream_key)
    if execution is None:
        execution = _new_blackwell_prepared_execution(prepared.compiled, buffers)
        prepared.stream_executions[stream_key] = execution
    else:
        prepared.stream_executions.move_to_end(stream_key)

    execution.executor.run_compiled_program(execution.exe_args)
    if graph_capture:
        _retain_blackwell_grouped_captured_launch(prepared.compiled, buffers)
        prepared.captured = True
    _record_pointer_buffers_use(
        buffers,
        graph_capture=graph_capture,
        stream=torch_stream,
    )
    _record_blackwell_launch_tensor_uses(
        prepared.compiled,
        buffers,
        a_groups,
        b_groups,
        out_groups,
        torch_stream,
    )


def _record_blackwell_launch_tensor_uses(
    compiled: _CompiledGroupedKernel,
    buffers: _PointerBuffers,
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    stream: torch.cuda.Stream,
) -> None:
    # Internal metadata and pointer buffers are owned by cache entries that are
    # retired behind the pointer-buffer event. Only caller-owned tensors need
    # record_stream to protect allocator lifetime after this wrapper returns.
    for tensor in (*a_groups, *b_groups, *out_groups):
        _record_tensor_use(tensor, stream)


def blackwell_grouped_gemm_nt(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor] | None = None,
    *,
    out_dtype: torch.dtype | None = None,
    mma_tiler_mn: Sequence[int] = (128, 64),
    cluster_shape_mn: Sequence[int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Run Blackwell CuTeDSL grouped GEMM for independent NT problems.

    Each group computes ``out_g = A_g @ B_g.T`` with ``A_g`` shaped ``[M, K]``,
    ``B_g`` shaped ``[N, K]``, and ``out_g`` shaped ``[M, N]``.  The groups may
    have different ``M``, ``N``, and ``K`` sizes.  A/B must share fp16 or bf16
    dtype; outputs may be fp16, bf16, or fp32.  All groups must share one
    majorness per operand, and each data pointer, contiguous dimension, and
    leading stride must be 16-byte aligned.
    """
    a_tuple = tuple(a_groups)
    b_tuple = tuple(b_groups)
    prepared_key: tuple[object, ...] | None = None
    mma_tiler, cluster_shape = _validate_blackwell_grouped_config(
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    if out_groups is None:
        if not a_tuple:
            raise ValueError("grouped GEMM requires at least one group")
        if len(a_tuple) != len(b_tuple):
            raise ValueError("A and B group counts must match")
        output_dtype = a_tuple[0].dtype if out_dtype is None else out_dtype
        out_tuple = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=a.device,
                dtype=output_dtype,
            )
            for a, b in zip(a_tuple, b_tuple, strict=True)
        )
    else:
        if out_dtype is not None:
            raise ValueError("out_dtype cannot be provided with explicit out_groups")
        out_tuple = tuple(out_groups)
        prepared = _get_last_blackwell_prepared_launch(
            a_tuple,
            b_tuple,
            out_tuple,
            mma_tiler_mn=mma_tiler,
            cluster_shape_mn=cluster_shape,
            use_2cta_instrs=use_2cta_instrs,
        )
        if prepared is not None:
            with torch.cuda.device(prepared.compiled.dim_torch.device):
                _launch_blackwell_prepared(prepared, a_tuple, b_tuple, out_tuple)
            return out_tuple
        prepared_key = _blackwell_prepared_cache_key(
            a_tuple,
            b_tuple,
            out_tuple,
            mma_tiler_mn=mma_tiler,
            cluster_shape_mn=cluster_shape,
            use_2cta_instrs=use_2cta_instrs,
        )
        prepared = _get_cached_blackwell_prepared_launch(
            prepared_key,
            a_tuple,
            b_tuple,
            out_tuple,
        )
        if prepared is not None:
            with torch.cuda.device(prepared.compiled.dim_torch.device):
                _launch_blackwell_prepared(prepared, a_tuple, b_tuple, out_tuple)
            return out_tuple

    metadata = _blackwell_grouped_gemm_nt_metadata(a_tuple, b_tuple, out_tuple)
    with torch.cuda.device(metadata.device):
        compiled = _get_compiled_blackwell_grouped_kernel(
            metadata,
            mma_tiler_mn=mma_tiler,
            cluster_shape_mn=cluster_shape,
            use_2cta_instrs=use_2cta_instrs,
        )
        if prepared_key is None:
            pointer_buffers, graph_capture = _acquire_blackwell_pointer_buffers(
                compiled,
                num_groups=len(metadata.problems),
                device=metadata.device,
            )
        else:
            pointer_buffers = _new_pointer_buffers(
                len(metadata.problems),
                metadata.device,
            )
            graph_capture = _cuda_graph_capture_active()
        _update_blackwell_pointer_buffers(
            pointer_buffers,
            a_tuple,
            b_tuple,
            out_tuple,
        )
        torch_stream = torch.cuda.current_stream()
        if prepared_key is None:
            compiled.compiled(
                compiled.initial[0],
                compiled.initial[1],
                compiled.initial[2],
                compiled.dim_tensor,
                compiled.stride_tensor,
                pointer_buffers.cute_tensor,
                compiled.tensormap,
                cutlass_torch.current_stream(),
            )
            _record_pointer_buffers_use(
                pointer_buffers,
                graph_capture=graph_capture,
                stream=torch_stream,
            )
            _record_blackwell_launch_tensor_uses(
                compiled,
                pointer_buffers,
                a_tuple,
                b_tuple,
                out_tuple,
                torch_stream,
            )
        else:
            prepared = _PreparedBlackwellGroupedLaunch(
                compiled=compiled,
                pointer_buffers=pointer_buffers,
                tensor_refs=_blackwell_prepared_tensor_refs(
                    a_tuple,
                    b_tuple,
                    out_tuple,
                ),
                stream_executions=OrderedDict(),
            )
            _cache_blackwell_prepared_launch(
                prepared_key,
                prepared,
                a_tuple,
                b_tuple,
                out_tuple,
            )
            _launch_blackwell_prepared(prepared, a_tuple, b_tuple, out_tuple)
    return out_tuple
