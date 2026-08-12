"""CUDA-graph-compatible cuBLAS grouped GEMM benchmark adapter."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from itertools import pairwise
import os
from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

CUBLAS_GROUPED_BASELINE = "cublas_grouped_batched_ex_operator"

_CUBLAS_STATUS_SUCCESS = 0
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUDA_R_16F = 2
_CUDA_R_16BF = 14
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_POINTER_MODE_HOST = 0
_CUBLAS_WORKSPACE_BYTES = 32 * 1024 * 1024
_CUBLAS_WORKSPACE_ALIGNMENT = 256


@dataclass
class PreparedLaunch:
    call: Callable[[], object]
    owners: tuple[object, ...]

    def __call__(self) -> object:
        return self.call()


def _check_cublas(status: int, operation: str) -> None:
    if status != _CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"{operation} failed with cuBLAS status {status}")


class _CublasGroupedGemm:
    """Process-scoped cuBLAS grouped launch with persistent device metadata."""

    def __init__(
        self,
        problems: tuple[tuple[int, int, int, int], ...],
        group_a: tuple[torch.Tensor, ...],
        group_b: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        import torch

        if not problems:
            raise ValueError("cuBLAS comparison requires at least one group")
        if any(batch != 1 for _m, _n, _k, batch in problems):
            raise ValueError("cuBLAS comparison requires L=1 problems")
        if not (len(problems) == len(group_a) == len(group_b) == len(outputs)):
            raise ValueError("cuBLAS comparison requires one A/B/C tensor per group")
        tensors = (*group_a, *group_b, *outputs)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("cuBLAS comparison requires CUDA tensors")
        device = group_a[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("cuBLAS comparison requires tensors on one CUDA device")
        for (m, n, k, _batch), a, b, output in zip(
            problems, group_a, group_b, outputs, strict=True
        ):
            if (a.shape, b.shape, output.shape) != ((m, k), (n, k), (m, n)):
                raise ValueError(
                    "cuBLAS comparison tensor shapes do not match problems"
                )
            if any(not tensor.is_contiguous() for tensor in (a, b, output)):
                raise ValueError("cuBLAS comparison requires contiguous tensors")
            if k % 8 or any(tensor.data_ptr() % 16 for tensor in (a, b, output)):
                raise ValueError("cuBLAS comparison requires 16-byte alignment")
        output_ranges = sorted(
            (
                output.data_ptr(),
                output.data_ptr() + output.numel() * output.element_size(),
            )
            for output in outputs
        )
        if any(
            previous_end > current_start
            for (_previous_start, previous_end), (
                current_start,
                _current_end,
            ) in pairwise(output_ranges)
        ):
            raise ValueError("cuBLAS comparison output tensors must not overlap")
        common_dtype = group_a[0].dtype
        if any(
            tensor.dtype != common_dtype
            for tensors in (group_a, group_b, outputs)
            for tensor in tensors
        ):
            raise ValueError("cuBLAS comparison requires one common dtype")
        data_types = {
            torch.float16: _CUDA_R_16F,
            torch.bfloat16: _CUDA_R_16BF,
        }
        if common_dtype not in data_types:
            raise ValueError("cuBLAS comparison supports FP16 and BF16")
        cuda_version = torch.version.cuda
        if cuda_version is None:
            raise RuntimeError("cuBLAS comparison requires a CUDA PyTorch build")
        library_name = os.environ.get(
            "HELION_CUBLAS_LIBRARY",
            f"libcublas.so.{cuda_version.partition('.')[0]}",
        )
        library = ctypes.CDLL(library_name)
        void_pointer = ctypes.c_void_p
        int_pointer = ctypes.POINTER(ctypes.c_int)
        library.cublasCreate_v2.argtypes = [ctypes.POINTER(void_pointer)]
        library.cublasCreate_v2.restype = ctypes.c_int
        library.cublasDestroy_v2.argtypes = [void_pointer]
        library.cublasDestroy_v2.restype = ctypes.c_int
        library.cublasSetStream_v2.argtypes = [void_pointer, void_pointer]
        library.cublasSetStream_v2.restype = ctypes.c_int
        library.cublasSetWorkspace_v2.argtypes = [
            void_pointer,
            void_pointer,
            ctypes.c_size_t,
        ]
        library.cublasSetWorkspace_v2.restype = ctypes.c_int
        library.cublasSetPointerMode_v2.argtypes = [void_pointer, ctypes.c_int]
        library.cublasSetPointerMode_v2.restype = ctypes.c_int
        library.cublasGetVersion_v2.argtypes = [
            void_pointer,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.cublasGetVersion_v2.restype = ctypes.c_int
        grouped = library.cublasGemmGroupedBatchedEx
        grouped.argtypes = [
            void_pointer,
            int_pointer,
            int_pointer,
            int_pointer,
            int_pointer,
            int_pointer,
            void_pointer,
            void_pointer,
            ctypes.c_int,
            int_pointer,
            void_pointer,
            ctypes.c_int,
            int_pointer,
            void_pointer,
            void_pointer,
            ctypes.c_int,
            int_pointer,
            ctypes.c_int,
            int_pointer,
            ctypes.c_int,
        ]
        grouped.restype = ctypes.c_int

        handle = void_pointer()
        _check_cublas(library.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate")
        _check_cublas(
            library.cublasSetPointerMode_v2(handle, _CUBLAS_POINTER_MODE_HOST),
            "cublasSetPointerMode",
        )
        version = ctypes.c_int()
        _check_cublas(
            library.cublasGetVersion_v2(handle, ctypes.byref(version)),
            "cublasGetVersion",
        )

        group_count = len(problems)
        int_array = ctypes.c_int * group_count
        float_array = ctypes.c_float * group_count
        # cuBLAS is column-major. Swapping A/B computes the transpose of each
        # row-major A @ B.T into the row-major output buffer without copies.
        self.transa = int_array(*(_CUBLAS_OP_T for _ in problems))
        self.transb = int_array(*(_CUBLAS_OP_N for _ in problems))
        self.m = int_array(*(n for _m, n, _k, _l in problems))
        self.n = int_array(*(m for m, _n, _k, _l in problems))
        self.k = int_array(*(k for _m, _n, k, _l in problems))
        self.lda = int_array(*(k for _m, _n, k, _l in problems))
        self.ldb = int_array(*(k for _m, _n, k, _l in problems))
        self.ldc = int_array(*(n for _m, n, _k, _l in problems))
        self.group_sizes = int_array(*(1 for _ in problems))
        self.alpha = float_array(*(1.0 for _ in problems))
        self.beta = float_array(*(0.0 for _ in problems))
        # cublasGemmGroupedBatchedEx consumes device arrays of operand pointers.
        self.a_pointers = torch.tensor(
            [tensor.data_ptr() for tensor in group_b],
            device=device,
            dtype=torch.int64,
        )
        self.b_pointers = torch.tensor(
            [tensor.data_ptr() for tensor in group_a],
            device=device,
            dtype=torch.int64,
        )
        self.c_pointers = torch.tensor(
            [tensor.data_ptr() for tensor in outputs],
            device=device,
            dtype=torch.int64,
        )
        workspace_storage = torch.empty(
            _CUBLAS_WORKSPACE_BYTES + _CUBLAS_WORKSPACE_ALIGNMENT - 1,
            device=device,
            dtype=torch.uint8,
        )
        workspace_offset = (-workspace_storage.data_ptr()) % _CUBLAS_WORKSPACE_ALIGNMENT
        self.workspace_storage = workspace_storage
        self.workspace = workspace_storage[
            workspace_offset : workspace_offset + _CUBLAS_WORKSPACE_BYTES
        ]
        if self.workspace.data_ptr() % _CUBLAS_WORKSPACE_ALIGNMENT:
            raise RuntimeError("failed to align cuBLAS workspace")
        self.library = library
        self.library_name = library_name
        self.version = version.value
        self.handle = handle
        self.handle_finalizer = weakref.finalize(
            self,
            library.cublasDestroy_v2,
            handle,
        )
        self.grouped = grouped
        self.group_count = group_count
        self.device = device
        self.dtype = common_dtype
        self.data_type = data_types[self.dtype]

    def __call__(self) -> object:
        import torch

        with torch.cuda.device(self.device):
            return self._call_on_current_device()

    def _call_on_current_device(self) -> object:
        import torch

        stream = torch.cuda.current_stream(self.device).cuda_stream
        _check_cublas(
            self.library.cublasSetStream_v2(
                self.handle,
                ctypes.c_void_p(stream),
            ),
            "cublasSetStream",
        )
        # Setting a cuBLAS stream resets its workspace, so bind the dedicated
        # Blackwell-sized workspace after the capture stream on every launch.
        _check_cublas(
            self.library.cublasSetWorkspace_v2(
                self.handle,
                ctypes.c_void_p(self.workspace.data_ptr()),
                _CUBLAS_WORKSPACE_BYTES,
            ),
            "cublasSetWorkspace",
        )
        status = self.grouped(
            self.handle,
            self.transa,
            self.transb,
            self.m,
            self.n,
            self.k,
            ctypes.cast(self.alpha, ctypes.c_void_p),
            ctypes.c_void_p(self.a_pointers.data_ptr()),
            self.data_type,
            self.lda,
            ctypes.c_void_p(self.b_pointers.data_ptr()),
            self.data_type,
            self.ldb,
            ctypes.cast(self.beta, ctypes.c_void_p),
            ctypes.c_void_p(self.c_pointers.data_ptr()),
            self.data_type,
            self.ldc,
            self.group_count,
            self.group_sizes,
            _CUBLAS_COMPUTE_32F,
        )
        _check_cublas(status, "cublasGemmGroupedBatchedEx")
        return None


def prepare_cublas(
    problems: tuple[tuple[int, int, int, int], ...],
    group_a: tuple[torch.Tensor, ...],
    group_b: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
) -> tuple[PreparedLaunch, dict[str, object]]:
    """Prepare CUDA-graph-compatible cuBLAS grouped GEMM device work."""
    import torch

    if not group_a:
        raise ValueError("cuBLAS comparison requires at least one group")
    with torch.cuda.device(group_a[0].device):
        launch = _CublasGroupedGemm(problems, group_a, group_b, outputs)
    torch.cuda.synchronize(group_a[0].device)
    owners: tuple[object, ...] = (launch, *group_a, *group_b, *outputs)
    return PreparedLaunch(launch, owners), {
        "api": "cublasGemmGroupedBatchedEx",
        "library": launch.library_name,
        "version": launch.version,
        "dtype": str(launch.dtype),
        "pointer_mode": "host",
        "workspace_bytes": _CUBLAS_WORKSPACE_BYTES,
        "device_pointer_tables_preloaded": True,
        "timing_scope": "full captured public-API device work",
    }
