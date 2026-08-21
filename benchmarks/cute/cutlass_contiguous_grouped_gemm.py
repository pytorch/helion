from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import importlib
import importlib.metadata
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation
    import torch

CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
CUTLASS_TAG = "4.7.0"
CUTLASS_COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
CUTLASS_DSL_VERSION = "4.7.0"
CUTLASS_OPERATOR_API_VERSION = "0.2.0"
CUTLASS_TARGET_SM = "100a"
CUTLASS_OPERATOR_BASELINE = "cutlass_operator_contiguous_offset_bf16"
_OPERATOR_MODULE = (
    "cutlass.operators.providers.cutedsl.gemm.sm100_contiguous_offset_2d3d_dense_gemm"
)
_PINNED_DEPENDENCIES = {
    "nvidia-cutlass-dsl": CUTLASS_DSL_VERSION,
    "nvidia-cutlass-dsl-libs-base": CUTLASS_DSL_VERSION,
    "nvidia-cutlass-dsl-libs-core": CUTLASS_DSL_VERSION,
    "nvidia-cutlass-dsl-libs-cu13": CUTLASS_DSL_VERSION,
    "apache-tvm-ffi": "0.1.13.post3",
    "torch-c-dlpack-ext": "0.1.5",
}


@dataclass(frozen=True, slots=True)
class _CutlassOperatorConfig:
    use_2cta_mma: bool
    tile_shape: tuple[int, int, int]
    cluster_shape: tuple[int, int, int]


def _int3(values: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise RuntimeError(f"CUTLASS {name} must contain three positive integers")
    first, second, third = result
    return first, second, third


def require_cutlass_dependencies() -> None:
    """Require the exact package set used for publication."""

    problems: list[str] = []
    for name, expected in _PINNED_DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed")
            continue
        if actual != expected:
            problems.append(f"{name} is {actual}, expected {expected}")
    if problems:
        raise RuntimeError(
            "CUTLASS Operator adapter dependencies are unavailable: "
            + "; ".join(problems)
        )


def verify_cutlass_checkout(cutlass_root: Path) -> dict[str, object]:
    """Require the clean pinned CUTLASS source used by the public campaign."""

    from benchmarks.cute import grouped_gemm_benchmark as common

    checkout = common.clean_checkout(cutlass_root, CUTLASS_COMMIT, "CUTLASS")
    root = Path(str(checkout["path"]))
    if not (root / "operators" / "cutlass").is_dir():
        raise RuntimeError(f"CUTLASS Operator API package is missing below {root}")
    return {
        "repository": CUTLASS_REPOSITORY,
        "tag": CUTLASS_TAG,
        **checkout,
    }


def _load_operator_api(cutlass_root: Path) -> tuple[Any, type[object]]:
    package_root = (cutlass_root / "operators" / "cutlass").resolve()
    cutlass = importlib.import_module("cutlass")
    package_path = cutlass.__path__
    value = str(package_root)
    # Providers run in fresh subprocesses, so extending the installed CUTLASS
    # namespace with the pinned source checkout is intentionally process-local.
    cutlass.__path__ = [
        value,
        *(str(path) for path in package_path if str(Path(path).resolve()) != value),
    ]
    importlib.invalidate_caches()
    try:
        ops = cast("Any", importlib.import_module("cutlass.operators"))
        module = importlib.import_module(_OPERATOR_MODULE)
        operator_class = cast(
            "type[object]", module.ContiguousOffset2D3DGemmDenseOperator
        )
    except (AttributeError, ImportError, OSError) as error:
        raise RuntimeError("failed to import CUTLASS grouped operators") from error
    return ops, operator_class


def _operator_config(metadata: object) -> _CutlassOperatorConfig:
    metadata = cast("Any", metadata)
    return _CutlassOperatorConfig(
        bool(metadata.design.use_2cta_mma),
        _int3(metadata.design.tile_shape, "operator tile_shape"),
        _int3(metadata.design.cluster_shape, "operator cluster_shape"),
    )


def prepare_cutlass_default(
    inputs: GroupedGemmInputs,
    *,
    cutlass_root: Path,
) -> PreparedImplementation:
    """Compile only the first operator returned by CUTLASS's public registry."""

    from benchmarks.cute import grouped_gemm_benchmark as common
    import torch

    checkout = verify_cutlass_checkout(cutlass_root)
    require_cutlass_dependencies()
    device = inputs.compact_a.device
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("CUTLASS grouped adapter targets B200/SM100")
    if inputs.case.k % 8 or inputs.case.n % 8:
        raise ValueError("CUTLASS BF16 K and N must be multiples of 8")

    a = inputs.compact_a
    b_kn = inputs.b.transpose(1, 2)
    offsets = inputs.offsets[1:]
    output = torch.empty(
        (inputs.case.total_m, inputs.case.n),
        device=device,
        dtype=torch.bfloat16,
    )
    ops, operator_class = _load_operator_api(cutlass_root)
    operator_api_version = str(ops.__version__)
    if operator_api_version != CUTLASS_OPERATOR_API_VERSION:
        raise RuntimeError(
            "CUTLASS Operator API version is "
            f"{operator_api_version}, expected {CUTLASS_OPERATOR_API_VERSION}"
        )
    global_options = ops.GlobalOptions()
    global_options.use_tvm_ffi = True
    if ops.GlobalOptions().use_tvm_ffi is not True:
        raise RuntimeError("CUTLASS global use_tvm_ffi option did not persist")
    with torch.cuda.device(device):
        args = ops.GroupedGemmArguments(
            a,
            b_kn,
            output,
            accumulator_type=torch.float32,
            offsets=offsets,
        )
        discovered = ops.get_operators(
            args,
            metadata_filter=lambda metadata: metadata.operator_class is operator_class,
            target_sm=CUTLASS_TARGET_SM,
            providers=[ops.CuTeDSLProvider],
        )
    if not discovered:
        raise RuntimeError(
            f"CUTLASS returned no {operator_class.__name__} operator for the workload"
        )
    operator = discovered[0]
    config = _operator_config(operator.metadata)
    with torch.cuda.device(device):
        artifact = operator.compile(args, target_sm=CUTLASS_TARGET_SM)

    def call() -> torch.Tensor:
        with torch.cuda.device(device):
            operator.run(
                args,
                compiled_artifact=artifact,
                stream=torch.cuda.current_stream(device),
                assume_supported_args=True,
            )
        return output

    return common.PreparedImplementation(
        name=f"{CUTLASS_OPERATOR_BASELINE}_registry_first",
        call=call,
        output_tensors=lambda _result: (output,),
        logical_outputs=lambda _result: inputs.compact_output_slices(output),
        config={
            "provider": "cutlass",
            "selection_mode": "public_registry_first",
            "baseline": CUTLASS_OPERATOR_BASELINE,
            "repository": CUTLASS_REPOSITORY,
            "tag": CUTLASS_TAG,
            "checkout": checkout,
            "operator_api_version": operator_api_version,
            "global_options": {"use_tvm_ffi": True},
            "target_sm": CUTLASS_TARGET_SM,
            "b_layout": "k_major",
            "selected_config": asdict(config),
            "operator_name": str(operator.metadata.operator_name),
            "compiled_for": str(artifact.compiled_for),
            "a_layout": common.compact_contiguous_a_layout(),
            "preprocessing_timed": False,
            "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
        },
        owners=(
            inputs,
            a,
            b_kn,
            offsets,
            output,
            global_options,
            args,
            operator,
            artifact,
        ),
    )
