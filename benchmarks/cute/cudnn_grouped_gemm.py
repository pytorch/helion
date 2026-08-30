from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import weakref

from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute.grouped_gemm_workloads import BENCHMARK_B_LAYOUT
import torch

if TYPE_CHECKING:
    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation


CUDNN_GROUPED_BASELINE = "cudnn_moe_grouped_matmul"
CUDNN_FRONTEND_DISTRIBUTION = "nvidia-cudnn-frontend"
CUDNN_FRONTEND_VERSION = "1.27.0"
CUDNN_BACKEND_DISTRIBUTION = "nvidia-cudnn-cu13"
CUDNN_BACKEND_DISTRIBUTION_VERSION = "9.24.0.43"
CUDNN_BACKEND_VERSION = 92400
CUDNN_CUDART_ENVIRONMENT_VARIABLE = "CUDNN_FRONTEND_CUDART_LIB_NAME"
CUDNN_LIBRARY_RELATIVE_PATH = Path("nvidia/cudnn/lib/libcudnn.so.9")

# cuDNN graph tensor UIDs are arbitrary, stable identifiers within this graph.
_TOKEN_UID = 20
_WEIGHT_UID = 21
_FIRST_TOKEN_OFFSET_UID = 22
_OUTPUT_UID = 100


def configure_cudnn_cudart_library() -> dict[str, str]:
    """Select the CUDA runtime used by the frontend shim before import."""

    distribution = common.require_pinned_distribution(
        common.CUDA_RUNTIME_DISTRIBUTION,
        common.CUDA_RUNTIME_VERSION,
    )
    default = common.distribution_file(
        common.CUDA_RUNTIME_DISTRIBUTION,
        distribution,
        common.CUDA_RUNTIME_LIBRARY_RELATIVE_PATH,
    )
    path = Path(os.environ.get(CUDNN_CUDART_ENVIRONMENT_VARIABLE, default)).resolve()
    if path != default:
        raise RuntimeError(
            f"{CUDNN_CUDART_ENVIRONMENT_VARIABLE} must resolve to {default}"
        )
    os.environ[CUDNN_CUDART_ENVIRONMENT_VARIABLE] = str(path)
    return {
        "distribution": common.CUDA_RUNTIME_DISTRIBUTION,
        "package_version": distribution.version,
        "path": str(path),
    }


def _import_cudnn() -> object:
    try:
        return importlib.import_module("cudnn")
    except (ImportError, OSError) as error:
        raise RuntimeError("cuDNN frontend is unavailable") from error


def _backend_version_string(version: int) -> str:
    major, remainder = divmod(version, 10_000)
    minor, patch = divmod(remainder, 100)
    return f"{major}.{minor}.{patch}"


def _distribution_identity(identity: dict[str, str]) -> dict[str, str]:
    return {key: identity[key] for key in ("distribution", "package_version")}


def _backend_library_identities() -> dict[str, object]:
    distribution = common.require_pinned_distribution(
        CUDNN_BACKEND_DISTRIBUTION,
        CUDNN_BACKEND_DISTRIBUTION_VERSION,
    )
    expected = {
        Path(str(distribution.locate_file(path))).resolve(strict=True)
        for path in distribution.files or ()
        if Path(str(path)).name.startswith("libcudnn")
        and ".so.9" in Path(str(path)).name
    }
    loaded = set(common.mapped_library_paths("libcudnn"))
    main_library = common.distribution_file(
        CUDNN_BACKEND_DISTRIBUTION,
        distribution,
        CUDNN_LIBRARY_RELATIVE_PATH,
    )
    if main_library not in loaded or not loaded.issubset(expected):
        raise RuntimeError(
            "loaded cuDNN libraries are not all from the pinned distribution: "
            f"{sorted(map(str, loaded))}"
        )
    return {
        "distribution": CUDNN_BACKEND_DISTRIBUTION,
        "package_version": distribution.version,
    }


def _loaded_cuda_runtime_identity() -> dict[str, str]:
    expected_identity = configure_cudnn_cudart_library()
    expected = Path(expected_identity["path"])
    loaded = common.mapped_library_paths("libcudart.so")
    if loaded != (expected,):
        raise RuntimeError(
            "loaded CUDA runtimes are "
            f"{tuple(map(str, loaded))}, expected {(str(expected),)}"
        )
    return _distribution_identity(expected_identity)


def _frontend_identity(cudnn: object) -> dict[str, object]:
    cudnn = cast("Any", cudnn)
    distribution = common.require_pinned_distribution(
        CUDNN_FRONTEND_DISTRIBUTION,
        CUDNN_FRONTEND_VERSION,
    )
    expected_module = common.distribution_file(
        CUDNN_FRONTEND_DISTRIBUTION,
        distribution,
        Path("cudnn/__init__.py"),
    )
    module = Path(str(cudnn.__file__)).resolve(strict=True)
    if module != expected_module:
        raise RuntimeError(
            f"cuDNN frontend imported from {module}, expected {expected_module}"
        )
    extension = Path(str(cudnn._pybind_module.__file__)).resolve(strict=True)
    if not extension.is_relative_to(expected_module.parent):
        raise RuntimeError(
            "cuDNN frontend extension was imported outside its distribution: "
            f"{extension}"
        )
    return {
        "distribution": CUDNN_FRONTEND_DISTRIBUTION,
        "package_version": distribution.version,
    }


def _validated_cudnn_runtime() -> tuple[Any, str, int, dict[str, object]]:
    cudart = configure_cudnn_cudart_library()
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
    return (
        cudnn,
        frontend_version,
        backend_version,
        {
            "frontend": _frontend_identity(cudnn),
            "requested_cuda_runtime": _distribution_identity(cudart),
        },
    )


def _selected_plan_identity(graph: object) -> dict[str, object]:
    graph = cast("Any", graph)
    selected_index = int(graph._plan_index)
    engine_id, knobs = graph.get_engine_and_knobs_at_index(selected_index)
    return {
        "candidate_count": int(graph.get_execution_plan_count()),
        "selected_index": selected_index,
        "name": str(graph.get_plan_name_at_index(selected_index)),
        "engine_id": int(engine_id),
        "knobs": [
            {"type": str(knob), "value": int(value)}
            for knob, value in sorted(knobs.items(), key=lambda pair: str(pair[0]))
        ],
    }


class _CudnnLaunch:
    def __init__(
        self,
        inputs: GroupedGemmInputs,
    ) -> None:
        cudnn, frontend_version, backend_version, runtime_identity = (
            _validated_cudnn_runtime()
        )
        self.inputs = inputs
        a = inputs.compact_a
        b = inputs.b
        self.output = torch.empty(
            (inputs.case.total_m, inputs.case.n),
            device=a.device,
            dtype=torch.bfloat16,
        )
        self.device = a.device
        self._cudnn = cudnn
        # PreparedImplementation.config intentionally retains this dict by
        # reference so the first execution can append loaded-library provenance.
        self._runtime_identity = runtime_identity
        self._loaded_runtime_validated = False

        with torch.cuda.device(self.device):
            self.token = a.unsqueeze(0)
            # The frontend ABI is [G,K,N], so use a zero-copy transpose of the
            # benchmark's canonical K-major [G,N,K] storage.
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
            self.selected_plan = _selected_plan_identity(graph)
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
            "runtime": runtime_identity,
            "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
        }

    def run(self) -> torch.Tensor:
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
        if not self._loaded_runtime_validated:
            self._runtime_identity.update(
                {
                    "backend_libraries": _backend_library_identities(),
                    "loaded_cuda_runtime": _loaded_cuda_runtime_identity(),
                }
            )
            self._loaded_runtime_validated = True
        return self.output

    def prepared_implementation(self) -> PreparedImplementation:
        return common.PreparedImplementation(
            name=f"cudnn-{BENCHMARK_B_LAYOUT}-graph-default",
            call=self.run,
            output_tensors=lambda _result: (self.output,),
            logical_outputs=lambda _result: self.inputs.compact_output_slices(
                self.output
            ),
            config=common.provider_config(
                "cudnn",
                {
                    **self.base_provenance,
                    "plan": {
                        "selection": "graph_build_default",
                        "heuristic_modes": ["A", "FALLBACK"],
                        **self.selected_plan,
                        "workspace_bytes": self.workspace_bytes,
                    },
                    "a_layout": common.compact_contiguous_a_layout(),
                },
            ),
        )


def prepare_cudnn_default(
    inputs: GroupedGemmInputs,
) -> PreparedImplementation:
    """Build and execute cuDNN's first buildable A/FALLBACK plan."""

    return _CudnnLaunch(inputs).prepared_implementation()
