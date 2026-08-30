from __future__ import annotations

import ctypes
from dataclasses import asdict
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import TYPE_CHECKING
import weakref

from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute.grouped_gemm_workloads import BENCHMARK_B_LAYOUT
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation


CUBLASLT_DISTRIBUTION = common.CUDA_CUBLAS_DISTRIBUTION
CUBLASLT_DISTRIBUTION_VERSION = common.CUDA_CUBLAS_VERSION
CUBLASLT_LIBRARY_RELATIVE_PATH = common.CUDA_CUBLASLT_LIBRARY_RELATIVE_PATH
# These ABI values are from the CUDA 13.6 cuBLASLt API shipped by the pinned
# nvidia-cublas distribution below. The grouped extension is version-gated by
# cublasLtGetVersion before any descriptor is created.
_CUBLAS_STATUS_SUCCESS = 0
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUDA_R_32F = 0
_CUDA_R_16BF = 14
_CUBLAS_COMPUTE_32F = 68
_CUBLASLT_POINTER_MODE_DEVICE = 1
_CUBLASLT_MATMUL_DESC_POINTER_MODE = 2
_CUBLASLT_MATMUL_DESC_TRANSA = 3
_CUBLASLT_MATMUL_DESC_TRANSB = 4
_CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5
_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6
_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7
_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8
_CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM = 13
_CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS = 14
_CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS = 15
_CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT = 23
_CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH = 12
_CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH = 13
_CUBLASLT_INTEGER_WIDTH_32 = 0
_DEFAULT_WORKSPACE_BYTES = 64 * 1024 * 1024
_WORKSPACE_ALIGNMENT = 256
CUBLASLT_HEURISTIC_QUERY_CAPACITY = 1
CUBLASLT_LIBRARY_VERSION = 130601


class _CublasLtMatmulAlgo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _CublasLtHeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _CublasLtMatmulAlgo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


@dataclass(frozen=True, slots=True)
class _CublasLtAlgorithm:
    serialized_hex: str
    workspace_bytes: int
    waves_count: float
    heuristic_rank: int


def _check_cublas(status: int, operation: str) -> None:
    if status != _CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"{operation} failed with cuBLAS status {status}")


def _pointer_alignment(pointer: int) -> int:
    return min(pointer & -pointer, 256)


def _configure_library(library: ctypes.CDLL) -> None:
    v, i, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    pv, pi, ps = ctypes.POINTER(v), ctypes.POINTER(i), ctypes.POINTER(s)
    pa = ctypes.POINTER(_CublasLtMatmulAlgo)
    ph = ctypes.POINTER(_CublasLtHeuristicResult)
    signatures = (
        ("cublasLtCreate", (pv,), i),
        ("cublasLtDestroy", (v,), i),
        ("cublasLtGetVersion", (), s),
        ("cublasLtMatmulDescCreate", (pv, i, i), i),
        ("cublasLtMatmulDescDestroy", (v,), i),
        ("cublasLtMatmulDescSetAttribute", (v, i, v, s), i),
        ("cublasLtGroupedMatrixLayoutCreate", (pv, i, i, v, v, v), i),
        ("cublasLtMatrixLayoutSetAttribute", (v, i, v, s), i),
        ("cublasLtMatrixLayoutDestroy", (v,), i),
        ("cublasLtMatmulPreferenceCreate", (pv,), i),
        ("cublasLtMatmulPreferenceDestroy", (v,), i),
        ("cublasLtMatmulPreferenceSetAttribute", (v, i, v, s), i),
        ("cublasLtMatmulAlgoGetHeuristicForStream", (v,) * 7 + (i, ph, pi, v), i),
        ("cublasLtMatmulAlgoCapGetAttribute", (pa, i, v, s, ps), i),
        ("cublasLtMatmul", (v,) * 12 + (pa, v, s, v), i),
    )
    for name, argtypes, restype in signatures:
        function = getattr(library, name)
        function.argtypes = list(argtypes)
        function.restype = restype


@functools.cache
def _validated_cublaslt_library() -> tuple[ctypes.CDLL, dict[str, object]]:
    distribution = common.require_pinned_distribution(
        CUBLASLT_DISTRIBUTION,
        CUBLASLT_DISTRIBUTION_VERSION,
    )
    path = common.distribution_file(
        CUBLASLT_DISTRIBUTION,
        distribution,
        CUBLASLT_LIBRARY_RELATIVE_PATH,
    )
    library = ctypes.CDLL(str(path))
    loaded_path = Path(str(library._name)).resolve(strict=True)
    if loaded_path != path:
        raise RuntimeError(
            f"cuBLASLt loaded {loaded_path}, expected distribution library {path}"
        )
    mapped = common.mapped_library_paths("libcublasLt.so")
    if mapped != (path,):
        raise RuntimeError(
            f"mapped cuBLASLt libraries are {tuple(map(str, mapped))}, "
            f"expected {(str(path),)}"
        )
    _configure_library(library)
    library_version = int(library.cublasLtGetVersion())
    if library_version != CUBLASLT_LIBRARY_VERSION:
        raise RuntimeError(
            f"cuBLASLt library is {library_version}, "
            f"expected {CUBLASLT_LIBRARY_VERSION}"
        )
    return library, {
        "distribution": CUBLASLT_DISTRIBUTION,
        "package_version": distribution.version,
        "library_version": library_version,
    }


def _destroy_resources(
    library: ctypes.CDLL,
    matrix_layouts: tuple[ctypes.c_void_p, ...],
    operation: ctypes.c_void_p,
    handle: ctypes.c_void_p,
) -> None:
    for layout in reversed(matrix_layouts):
        library.cublasLtMatrixLayoutDestroy(layout)
    library.cublasLtMatmulDescDestroy(operation)
    library.cublasLtDestroy(handle)


def _set_attribute(
    function: Callable[..., int],
    descriptor: ctypes.c_void_p,
    attribute: int,
    value: ctypes.c_int | ctypes.c_uint32 | ctypes.c_uint64,
    operation: str,
) -> None:
    _check_cublas(
        function(descriptor, attribute, ctypes.byref(value), ctypes.sizeof(value)),
        operation,
    )


def _create_grouped_matrix_layout(
    library: ctypes.CDLL,
    group_count: int,
    rows: torch.Tensor,
    columns: torch.Tensor,
    leading_dimensions: torch.Tensor,
) -> ctypes.c_void_p:
    layout = ctypes.c_void_p()
    _check_cublas(
        library.cublasLtGroupedMatrixLayoutCreate(
            ctypes.byref(layout),
            _CUDA_R_16BF,
            group_count,
            ctypes.c_void_p(rows.data_ptr()),
            ctypes.c_void_p(columns.data_ptr()),
            ctypes.c_void_p(leading_dimensions.data_ptr()),
        ),
        "cublasLtGroupedMatrixLayoutCreate",
    )
    try:
        for attribute, label in (
            (
                _CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
                "rows/columns",
            ),
            (_CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH, "LD"),
        ):
            _set_attribute(
                library.cublasLtMatrixLayoutSetAttribute,
                layout,
                attribute,
                ctypes.c_int(_CUBLASLT_INTEGER_WIDTH_32),
                f"set grouped matrix {label} integer width",
            )
    except Exception:
        library.cublasLtMatrixLayoutDestroy(layout)
        raise
    return layout


def _create_grouped_matrix_layouts(
    library: ctypes.CDLL,
    group_count: int,
    dimension_arrays: dict[str, torch.Tensor],
) -> tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]:
    layouts: list[ctypes.c_void_p] = []
    try:
        for prefix in ("a", "b", "output"):
            layouts.append(
                _create_grouped_matrix_layout(
                    library,
                    group_count,
                    dimension_arrays[f"{prefix}_rows"],
                    dimension_arrays[f"{prefix}_columns"],
                    dimension_arrays[f"{prefix}_leading_dimensions"],
                )
            )
    except Exception:
        for layout in reversed(layouts):
            library.cublasLtMatrixLayoutDestroy(layout)
        raise
    return layouts[0], layouts[1], layouts[2]


def _grouped_algorithm_supported(
    library: ctypes.CDLL,
    algorithm: _CublasLtMatmulAlgo,
) -> bool:
    value = ctypes.c_int32()
    written = ctypes.c_size_t()
    status = library.cublasLtMatmulAlgoCapGetAttribute(
        ctypes.byref(algorithm),
        _CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT,
        ctypes.byref(value),
        ctypes.sizeof(value),
        ctypes.byref(written),
    )
    return (
        status == _CUBLAS_STATUS_SUCCESS
        and written.value == ctypes.sizeof(value)
        and value.value != 0
    )


def cublaslt_layout_values(
    problems: tuple[tuple[int, int, int, int], ...],
) -> tuple[int, dict[str, list[int]]]:
    # cuBLASLt is column-major. Swap logical A/B so row-major A @ B.T becomes
    # column-major B @ A.T, producing the same row-major output allocation.
    m_values = [m for m, _n, _k, _batch in problems]
    n_values = [n for _m, n, _k, _batch in problems]
    k_values = [k for _m, _n, k, _batch in problems]
    return _CUBLAS_OP_T, {
        "a_rows": k_values,
        "a_columns": n_values,
        "a_leading_dimensions": k_values,
        "b_rows": k_values,
        "b_columns": m_values,
        "b_leading_dimensions": k_values,
        "output_rows": n_values,
        "output_columns": m_values,
        "output_leading_dimensions": n_values,
    }


def cublaslt_grouped_preference_values(
    problems: tuple[tuple[int, int, int, int], ...],
) -> dict[str, int]:
    """Return floor-mean uint32 shape hints for grouped heuristic selection."""

    if not problems:
        raise ValueError("cuBLASLt grouped preferences require at least one problem")
    group_count = len(problems)
    return {
        "average_reduction_dim": sum(k for _m, _n, k, _batch in problems)
        // group_count,
        # The column-major descriptor D is N x M after the row-major mapping in
        # ``cublaslt_layout_values``.
        "average_output_rows": sum(n for _m, n, _k, _batch in problems) // group_count,
        "average_output_columns": sum(m for m, _n, _k, _batch in problems)
        // group_count,
    }


def _set_grouped_preference_attributes(
    library: ctypes.CDLL,
    preference: ctypes.c_void_p,
    values: dict[str, int],
) -> None:
    for attribute, name in (
        (
            _CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM,
            "average_reduction_dim",
        ),
        (
            _CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS,
            "average_output_rows",
        ),
        (
            _CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS,
            "average_output_columns",
        ),
    ):
        _set_attribute(
            library.cublasLtMatmulPreferenceSetAttribute,
            preference,
            attribute,
            ctypes.c_uint32(values[name]),
            f"set grouped {name}",
        )


class _CublasLtGroupedGemm:
    def __init__(
        self,
        inputs: GroupedGemmInputs,
    ) -> None:
        if any(actual_m <= 0 for actual_m in inputs.case.actual_ms):
            raise ValueError("cuBLASLt requires every group to contain a row")
        library, library_identity = _validated_cublaslt_library()
        problems = tuple(
            (actual_m, inputs.case.n, inputs.case.k, 1)
            for actual_m in inputs.case.actual_ms
        )
        group_a = inputs.compact_a_slices()
        group_b = inputs.b.unbind()
        outputs = tuple(
            torch.empty(
                (actual_m, inputs.case.n),
                device=inputs.compact_a.device,
                dtype=torch.bfloat16,
            )
            for actual_m in inputs.case.actual_ms
        )
        device = inputs.compact_a.device

        workspace_storage = torch.empty(
            _DEFAULT_WORKSPACE_BYTES + _WORKSPACE_ALIGNMENT - 1,
            device=device,
            dtype=torch.uint8,
        )
        offset = (-workspace_storage.data_ptr()) % _WORKSPACE_ALIGNMENT
        workspace = workspace_storage[offset : offset + _DEFAULT_WORKSPACE_BYTES]

        group_count = len(problems)
        transa, dimension_values = cublaslt_layout_values(problems)
        grouped_preferences = cublaslt_grouped_preference_values(problems)
        with torch.cuda.device(device):
            query_stream = torch.cuda.current_stream(device)
            dimension_arrays = {
                name: torch.tensor(values, device=device, dtype=torch.int32)
                for name, values in dimension_values.items()
            }
            a_pointers = torch.tensor(
                [tensor.data_ptr() for tensor in group_b],
                device=device,
                dtype=torch.int64,
            )
            b_pointers = torch.tensor(
                [tensor.data_ptr() for tensor in group_a],
                device=device,
                dtype=torch.int64,
            )
            output_pointers = torch.tensor(
                [tensor.data_ptr() for tensor in outputs],
                device=device,
                dtype=torch.int64,
            )
            alpha = torch.tensor(1.0, device=device, dtype=torch.float32)
            beta = torch.tensor(0.0, device=device, dtype=torch.float32)
            query_stream.synchronize()

        handle = ctypes.c_void_p()
        operation = ctypes.c_void_p()
        preference = ctypes.c_void_p()
        layouts: list[ctypes.c_void_p] = []
        initialized = False
        try:
            _check_cublas(library.cublasLtCreate(ctypes.byref(handle)), "create")
            _check_cublas(
                library.cublasLtMatmulDescCreate(
                    ctypes.byref(operation), _CUBLAS_COMPUTE_32F, _CUDA_R_32F
                ),
                "create matmul descriptor",
            )
            for attribute, value, label in (
                (
                    _CUBLASLT_MATMUL_DESC_POINTER_MODE,
                    ctypes.c_int(_CUBLASLT_POINTER_MODE_DEVICE),
                    "pointer mode",
                ),
                (_CUBLASLT_MATMUL_DESC_TRANSA, ctypes.c_int(transa), "TRANSA"),
                (_CUBLASLT_MATMUL_DESC_TRANSB, ctypes.c_int(_CUBLAS_OP_N), "TRANSB"),
            ):
                _set_attribute(
                    library.cublasLtMatmulDescSetAttribute,
                    operation,
                    attribute,
                    value,
                    f"set {label}",
                )
            _check_cublas(
                library.cublasLtMatmulPreferenceCreate(ctypes.byref(preference)),
                "create preference",
            )
            _set_attribute(
                library.cublasLtMatmulPreferenceSetAttribute,
                preference,
                _CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.c_uint64(_DEFAULT_WORKSPACE_BYTES),
                "set workspace",
            )
            output_alignment = min(
                _pointer_alignment(tensor.data_ptr()) for tensor in outputs
            )
            alignments = {
                "A": min(_pointer_alignment(tensor.data_ptr()) for tensor in group_b),
                "B": min(_pointer_alignment(tensor.data_ptr()) for tensor in group_a),
                "C": output_alignment,
                "D": output_alignment,
            }
            for attribute, operand in (
                (_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, "A"),
                (_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, "B"),
                (_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, "C"),
                (_CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, "D"),
            ):
                _set_attribute(
                    library.cublasLtMatmulPreferenceSetAttribute,
                    preference,
                    attribute,
                    ctypes.c_uint32(alignments[operand]),
                    f"set {operand} alignment",
                )
            _set_grouped_preference_attributes(
                library,
                preference,
                grouped_preferences,
            )
            created_layouts = _create_grouped_matrix_layouts(
                library, group_count, dimension_arrays
            )
            layouts.extend(created_layouts)
            a_layout, b_matrix_layout, output_layout = created_layouts
            result = _CublasLtHeuristicResult()
            returned_count = ctypes.c_int()
            _check_cublas(
                library.cublasLtMatmulAlgoGetHeuristicForStream(
                    handle,
                    operation,
                    a_layout,
                    b_matrix_layout,
                    output_layout,
                    output_layout,
                    preference,
                    CUBLASLT_HEURISTIC_QUERY_CAPACITY,
                    ctypes.byref(result),
                    ctypes.byref(returned_count),
                    ctypes.c_void_p(query_stream.cuda_stream),
                ),
                "query grouped algorithms",
            )
            if returned_count.value != 1:
                raise RuntimeError(
                    "cuBLASLt did not return exactly one default heuristic result"
                )
            serialized = bytes(result.algo).hex()
            if (
                result.state != _CUBLAS_STATUS_SUCCESS
                or result.workspace_size > _DEFAULT_WORKSPACE_BYTES
                or not _grouped_algorithm_supported(library, result.algo)
            ):
                raise RuntimeError(
                    "cuBLASLt rank-zero heuristic result is not usable for grouped GEMM"
                )
            selected_algorithm = _CublasLtAlgorithm(
                serialized,
                int(result.workspace_size),
                float(result.waves_count),
                0,
            )
            initialized = True
        finally:
            if preference:
                library.cublasLtMatmulPreferenceDestroy(preference)
            if not initialized:
                for layout in reversed(layouts):
                    library.cublasLtMatrixLayoutDestroy(layout)
                if operation:
                    library.cublasLtMatmulDescDestroy(operation)
                if handle:
                    library.cublasLtDestroy(handle)

        self.library = library
        self.library_identity = library_identity
        self.handle = handle
        self.operation = operation
        self.a_layout = a_layout
        self.b_layout_descriptor = b_matrix_layout
        self.output_layout = output_layout
        self._finalizer = weakref.finalize(
            self,
            _destroy_resources,
            library,
            tuple(layouts),
            operation,
            handle,
        )
        self.outputs = outputs
        self.inputs = inputs
        self.device = device
        self.dimension_arrays = dimension_arrays
        self.a_pointers = a_pointers
        self.b_pointers = b_pointers
        self.output_pointers = output_pointers
        self.alpha = alpha
        self.beta = beta
        self.workspace_storage = workspace_storage
        self.workspace = workspace
        self.grouped_preferences = grouped_preferences
        self.selected_algorithm = selected_algorithm

    def prepared_implementation(
        self,
    ) -> PreparedImplementation:
        selected = self.selected_algorithm
        algorithm = _CublasLtMatmulAlgo.from_buffer_copy(
            bytes.fromhex(selected.serialized_hex)
        )

        def call() -> None:
            self(algorithm, selected)

        config = common.provider_config(
            "cublaslt",
            {
                "baseline": "cublaslt_native_heterogeneous_grouped_matmul",
                "api": "cublasLtGroupedMatrixLayoutCreate/cublasLtMatmul",
                "library": self.library_identity,
                "group_count": self.inputs.case.groups,
                "workspace_bytes": _DEFAULT_WORKSPACE_BYTES,
                "grouped_average_preferences": self.grouped_preferences,
                "grouped_average_preference_rounding": "integer_floor",
                "heuristic_query_capacity": CUBLASLT_HEURISTIC_QUERY_CAPACITY,
                "selected_algorithm": asdict(selected),
                "a_layout": {
                    "kind": "compact_group_views",
                    "shared_compact_allocation": True,
                    "logical_values_bitwise_equal": True,
                },
                "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
            },
        )
        return common.PreparedImplementation(
            name=f"cublaslt-{BENCHMARK_B_LAYOUT}-rank-{selected.heuristic_rank}",
            call=call,
            output_tensors=lambda _result: self.outputs,
            logical_outputs=lambda _result: self.outputs,
            config=config,
        )

    def __call__(
        self,
        algorithm: _CublasLtMatmulAlgo,
        metadata: _CublasLtAlgorithm,
    ) -> None:
        with torch.cuda.device(self.device):
            workspace_pointer = ctypes.c_void_p(self.workspace.data_ptr())
            _check_cublas(
                self.library.cublasLtMatmul(
                    self.handle,
                    self.operation,
                    ctypes.c_void_p(self.alpha.data_ptr()),
                    ctypes.c_void_p(self.a_pointers.data_ptr()),
                    self.a_layout,
                    ctypes.c_void_p(self.b_pointers.data_ptr()),
                    self.b_layout_descriptor,
                    ctypes.c_void_p(self.beta.data_ptr()),
                    ctypes.c_void_p(self.output_pointers.data_ptr()),
                    self.output_layout,
                    ctypes.c_void_p(self.output_pointers.data_ptr()),
                    self.output_layout,
                    ctypes.byref(algorithm),
                    workspace_pointer,
                    _DEFAULT_WORKSPACE_BYTES,
                    ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream),
                ),
                f"cublasLtMatmul grouped algorithm rank {metadata.heuristic_rank}",
            )


def prepare_cublaslt_default(
    inputs: GroupedGemmInputs,
) -> PreparedImplementation:
    """Use only cuBLASLt's rank-zero grouped-GEMM heuristic result."""

    with torch.cuda.device(inputs.compact_a.device):
        core = _CublasLtGroupedGemm(inputs)
    if core.selected_algorithm.heuristic_rank != 0:
        raise RuntimeError("cuBLASLt default result was not heuristic rank zero")
    return core.prepared_implementation()
