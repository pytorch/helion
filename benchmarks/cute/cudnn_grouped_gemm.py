from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import weakref

if TYPE_CHECKING:
    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation
    import torch

CUDNN_GROUPED_BASELINE = "cudnn_moe_grouped_matmul"
CUDNN_B_LAYOUTS = ("k_major", "n_major")
CUDNN_FRONTEND_VERSION = "1.27.0"
CUDNN_BACKEND_VERSION = 92400
CUDNN_REFERENCE_FRONTEND_COMMIT = "f77fbc3d21be3f24cd0286b9b368105f7c518b8a"
CUDNN_CUDART_ENVIRONMENT_VARIABLE = "CUDNN_FRONTEND_CUDART_LIB_NAME"
CUDA_RUNTIME_DISTRIBUTION = "nvidia-cuda-runtime"
CUDA_RUNTIME_VERSION = "13.3.29"
CUDNN_CUDART_RELATIVE_PATH = Path("nvidia/cu13/lib/libcudart.so.13")

# cuDNN graph tensor UIDs are arbitrary, stable identifiers within this graph.
_TOKEN_UID = 20
_WEIGHT_UID = 21
_FIRST_TOKEN_OFFSET_UID = 22
_OUTPUT_UID = 100


def configure_cudnn_cudart_library() -> Path:
    """Select the CUDA runtime used by the frontend shim before import."""

    distribution = importlib.metadata.distribution(CUDA_RUNTIME_DISTRIBUTION)
    if distribution.version != CUDA_RUNTIME_VERSION:
        raise RuntimeError(
            f"{CUDA_RUNTIME_DISTRIBUTION} is {distribution.version}, "
            f"expected {CUDA_RUNTIME_VERSION}"
        )
    default = Path(str(distribution.locate_file(CUDNN_CUDART_RELATIVE_PATH))).resolve(
        strict=True
    )
    path = Path(os.environ.get(CUDNN_CUDART_ENVIRONMENT_VARIABLE, default)).resolve()
    if path != default:
        raise RuntimeError(
            f"{CUDNN_CUDART_ENVIRONMENT_VARIABLE} must resolve to {default}"
        )
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"cuDNN CUDA runtime does not exist: {path}") from error
    if not resolved.is_file():
        raise RuntimeError(f"cuDNN CUDA runtime is not a file: {resolved}")
    os.environ[CUDNN_CUDART_ENVIRONMENT_VARIABLE] = str(resolved)
    return resolved


def _import_cudnn() -> object:
    try:
        return importlib.import_module("cudnn")
    except (ImportError, OSError) as error:
        raise RuntimeError("cuDNN frontend is unavailable") from error


def _backend_version_string(version: int) -> str:
    major, remainder = divmod(version, 10_000)
    minor, patch = divmod(remainder, 100)
    return f"{major}.{minor}.{patch}"


def _validated_cudnn_runtime() -> tuple[Path, Any, str, int]:
    cudart_path = configure_cudnn_cudart_library()
    cudnn = cast("Any", _import_cudnn())
    frontend_version = str(cudnn.__version__)
    backend_version = int(cudnn.backend_version())
    if frontend_version != CUDNN_FRONTEND_VERSION:
        raise RuntimeError(
            f"cuDNN frontend is {frontend_version}, expected {CUDNN_FRONTEND_VERSION}"
        )
    if backend_version != CUDNN_BACKEND_VERSION:
        raise RuntimeError(
            "cuDNN backend is "
            f"{_backend_version_string(backend_version)}, expected "
            f"{_backend_version_string(CUDNN_BACKEND_VERSION)}"
        )
    return cudart_path, cudnn, frontend_version, backend_version


class _CudnnLaunch:
    def __init__(
        self,
        inputs: GroupedGemmInputs,
        b_layout: str,
    ) -> None:
        import torch

        if b_layout not in CUDNN_B_LAYOUTS:
            raise ValueError(f"unsupported cuDNN B layout {b_layout!r}")
        cudart_path, cudnn, frontend_version, backend_version = (
            _validated_cudnn_runtime()
        )
        self.inputs = inputs
        self.b_layout = b_layout
        a = inputs.compact_a
        b = inputs.b_for_layout(b_layout)
        self.output = torch.empty(
            (inputs.case.total_m, inputs.case.n),
            device=a.device,
            dtype=torch.bfloat16,
        )
        self.device = a.device
        self._cudnn = cudnn

        with torch.cuda.device(self.device):
            self.token = a.unsqueeze(0)
            # The frontend ABI is [G,K,N]; this zero-copy transpose preserves
            # whichever physical logical-B layout the common input selected.
            self.weight = b.transpose(1, 2)
            # The MOE descriptor consumes one start per group, not an indptr
            # sentinel. Group extents come from the token tensor and next start.
            self.first_token_offsets = inputs.offsets[:-1].view(
                inputs.case.groups, 1, 1
            )
            self.output_3d = self.output.unsqueeze(0)
            self.handle = cudnn.create_handle()
            self._handle_finalizer = weakref.finalize(
                self, cudnn.destroy_handle, self.handle
            )
            cudnn.set_stream(
                handle=self.handle,
                stream=torch.cuda.current_stream(self.device).cuda_stream,
            )
            graph = cudnn.pygraph(
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
                handle=self.handle,
            )
            token_desc = graph.tensor(
                name="token",
                dim=list(self.token.shape),
                stride=list(self.token.stride()),
                data_type=cudnn.data_type.BFLOAT16,
                uid=_TOKEN_UID,
            )
            weight_desc = graph.tensor(
                name="weight",
                dim=list(self.weight.shape),
                stride=list(self.weight.stride()),
                data_type=cudnn.data_type.BFLOAT16,
                uid=_WEIGHT_UID,
            )
            offsets_desc = graph.tensor(
                name="first_token_offset",
                dim=list(self.first_token_offsets.shape),
                stride=list(self.first_token_offsets.stride()),
                data_type=cudnn.data_type.INT32,
                uid=_FIRST_TOKEN_OFFSET_UID,
            )
            accum = graph.moe_grouped_matmul(
                name="cudnn_moe_grouped_gemm",
                token=token_desc,
                weight=weight_desc,
                first_token_offset=offsets_desc,
                mode=cudnn.moe_grouped_matmul_mode.NONE,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            accum.set_uid(_OUTPUT_UID).set_output(True).set_data_type(
                cudnn.data_type.BFLOAT16
            )
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph.check_support()
            graph.build_plans()
            self.workspace_bytes = int(graph.get_workspace_size())
            self.graph = graph
            self.workspace = torch.empty(
                self.workspace_bytes,
                device=self.device,
                dtype=torch.uint8,
            )
            self._variant_pack = {
                _TOKEN_UID: self.token,
                _WEIGHT_UID: self.weight,
                _FIRST_TOKEN_OFFSET_UID: self.first_token_offsets,
                _OUTPUT_UID: self.output_3d,
            }
        self.base_provenance: dict[str, object] = {
            "baseline": CUDNN_GROUPED_BASELINE,
            "frontend_version": frontend_version,
            "backend_version": backend_version,
            "backend_version_string": _backend_version_string(backend_version),
            "frontend_release_commit": CUDNN_REFERENCE_FRONTEND_COMMIT,
            "cudart_path": str(cudart_path),
            "b_layout": b_layout,
            "selection_mode": "public_default_a_fallback",
            "preprocessing_timed": False,
            "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
        }

    def run(self) -> torch.Tensor:
        import torch

        with torch.cuda.device(self.device):
            # A captured graph may replay on a different current stream than
            # graph construction, so bind the frontend handle on every call.
            self._cudnn.set_stream(
                handle=self.handle,
                stream=torch.cuda.current_stream(self.device).cuda_stream,
            )
            self.graph.execute(
                self._variant_pack,
                self.workspace,
                handle=self.handle,
            )
        return self.output

    def prepared_implementation(self) -> PreparedImplementation:
        from benchmarks.cute import grouped_gemm_benchmark as common

        return common.PreparedImplementation(
            name=f"cudnn-{self.b_layout}-graph-default",
            call=self.run,
            output_tensors=lambda _result: (self.output,),
            logical_outputs=lambda _result: self.inputs.compact_output_slices(
                self.output
            ),
            config={
                "provider": "cudnn",
                **self.base_provenance,
                "plan": {
                    "selection": "graph_build_default",
                    "heuristic_modes": ["A", "FALLBACK"],
                    "workspace_bytes": self.workspace_bytes,
                },
                "a_layout": common.compact_contiguous_a_layout(),
            },
            owners=(self,),
        )


def prepare_cudnn_default(
    inputs: GroupedGemmInputs,
    *,
    b_layout: str,
) -> PreparedImplementation:
    """Build and execute cuDNN's first buildable A/FALLBACK plan."""

    return _CudnnLaunch(inputs, b_layout).prepared_implementation()
