from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import importlib
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from benchmarks.cute import grouped_gemm_benchmark as common
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation


CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
# Match Helion's repository-wide, validated CuTe DSL pin.
CUTLASS_RELEASE_TAG = "v4.7.0"
CUTLASS_COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
CUTLASS_DSL_VERSION = "4.7.0"
CUTLASS_OPERATOR_API_VERSION = "0.2.0"
CUTLASS_SELECTION_MIN_REPETITIONS = 32
CUTLASS_SELECTION_CAPTURE_WARMUPS = 2
CUTLASS_TARGET_SMS = {
    (10, 0): "100a",
    (10, 3): "103a",
}
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


@dataclass(frozen=True)
class _CutlassCandidate:
    registry_index: int
    operator_name: str
    compiled_for: str
    prepared: PreparedImplementation


def _int3(values: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise RuntimeError(f"CUTLASS {name} must contain three positive integers")
    first, second, third = result
    return first, second, third


def cutlass_target_sm(capability: tuple[int, int]) -> str:
    """Return the CUTLASS architecture target for a validated device."""

    try:
        return CUTLASS_TARGET_SMS[capability]
    except KeyError as error:
        raise RuntimeError(
            "CUTLASS grouped adapter targets B200/SM100 or GB300/SM103"
        ) from error


def require_cutlass_dependencies() -> None:
    """Require the exact package set used for publication."""

    common.require_pinned_distributions(
        _PINNED_DEPENDENCIES,
        "CUTLASS Operator adapter dependencies are unavailable",
    )


def verify_cutlass_checkout(cutlass_root: Path) -> tuple[Path, str]:
    """Require the clean pinned CUTLASS source used by the public campaign."""

    root = cutlass_root.expanduser().resolve(strict=True)
    commit = common.clean_checkout(root, CUTLASS_COMMIT, "CUTLASS")
    if not (root / "operators" / "cutlass").is_dir():
        raise RuntimeError(f"CUTLASS Operator API package is missing below {root}")
    return root, commit


def _load_operator_api(
    cutlass_root: Path,
) -> tuple[Any, type[object]]:
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
    if operator_class.__module__ != _OPERATOR_MODULE:
        raise RuntimeError(
            "CUTLASS grouped operator class came from "
            f"{operator_class.__module__!r}, expected {_OPERATOR_MODULE!r}"
        )
    common.validate_module_source(
        ops, cutlass_root, "CUTLASS module 'cutlass.operators'"
    )
    common.validate_module_source(
        module, cutlass_root, f"CUTLASS module {_OPERATOR_MODULE!r}"
    )
    return ops, operator_class


def _operator_config(metadata: object) -> dict[str, object]:
    metadata = cast("Any", metadata)
    return {
        "use_2cta_mma": bool(metadata.design.use_2cta_mma),
        "tile_shape": _int3(metadata.design.tile_shape, "operator tile_shape"),
        "cluster_shape": _int3(metadata.design.cluster_shape, "operator cluster_shape"),
    }


def _tune_candidates(
    candidates: Sequence[_CutlassCandidate],
    oracle: Sequence[torch.Tensor],
) -> dict[str, object]:
    """Validate and cold-L2 time every supported CUTLASS operator."""

    from pretuned_kernels import _bench

    if not candidates:
        raise ValueError("CUTLASS tuning requires at least one candidate")
    captures = []
    for candidate in candidates:
        captured = common.capture_implementation(
            candidate.prepared,
            warmups=CUTLASS_SELECTION_CAPTURE_WARMUPS,
        )
        evidence = common.validate_capture(captured, oracle)
        if not evidence["ok"]:
            raise RuntimeError(
                f"CUTLASS candidate {candidate.operator_name!r} failed correctness: "
                f"{evidence}"
            )
        captures.append(captured)

    cycle = 2 * len(candidates)
    repetitions = (CUTLASS_SELECTION_MIN_REPETITIONS + cycle - 1) // cycle * cycle
    selection_ms = _bench.bench_pre_captured_cudagraphs(
        [capture.replay for capture in captures],
        rep=repetitions,
    )
    if len(selection_ms) != len(candidates) or any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        for value in selection_ms
    ):
        raise RuntimeError("CUTLASS candidate timings must be finite and positive")

    selected_index = min(
        range(len(candidates)),
        key=lambda index: (selection_ms[index], candidates[index].operator_name),
    )
    evidence: dict[str, object] = {
        "method": "all_supported_operators_cold_l2",
        "timer": "bench_pre_captured_cudagraphs",
        "capture_warmups": CUTLASS_SELECTION_CAPTURE_WARMUPS,
        "repetitions": repetitions,
        "selected_candidate_index": selected_index,
        "candidates": [
            {
                "registry_index": candidate.registry_index,
                "operator_name": candidate.operator_name,
                "config": candidate.prepared.config,
                "compiled_for": candidate.compiled_for,
                "selection_median_ms": float(elapsed_ms),
                "correctness_checked": True,
            }
            for candidate, elapsed_ms in zip(
                candidates,
                selection_ms,
                strict=True,
            )
        ],
    }
    return evidence


def _prepared_candidate(
    inputs: GroupedGemmInputs,
    args: object,
    output: torch.Tensor,
    operator: object,
    artifact: object,
    *,
    name: str,
    config: dict[str, Any],
) -> PreparedImplementation:
    device = inputs.compact_a.device
    operator_api = cast("Any", operator)

    def call() -> torch.Tensor:
        with torch.cuda.device(device):
            operator_api.run(
                args,
                compiled_artifact=artifact,
                stream=torch.cuda.current_stream(device),
                assume_supported_args=True,
            )
        return output

    return common.PreparedImplementation(
        name=name,
        call=call,
        output_tensors=lambda _result: (output,),
        logical_outputs=lambda _result: inputs.compact_output_slices(output),
        config=config,
    )


def prepare_cutlass_default(
    inputs: GroupedGemmInputs,
    *,
    cutlass_root: Path,
) -> PreparedImplementation:
    """Tune every supported public-registry operator outside final timing."""

    cutlass_root, commit = verify_cutlass_checkout(cutlass_root)
    require_cutlass_dependencies()
    device = inputs.compact_a.device
    target_sm = cutlass_target_sm(torch.cuda.get_device_capability(device))
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
            target_sm=target_sm,
            providers=[ops.CuTeDSLProvider],
        )
    if not discovered:
        raise RuntimeError(
            f"CUTLASS returned no {operator_class.__name__} operator for the workload"
        )
    indexed_operators = sorted(
        enumerate(discovered),
        key=lambda item: str(item[1].metadata.operator_name),
    )
    operator_names = [
        str(operator.metadata.operator_name) for _, operator in indexed_operators
    ]
    if len(operator_names) != len(set(operator_names)):
        raise RuntimeError("CUTLASS returned duplicate supported operator names")

    candidates = []
    for candidate_index, (registry_index, operator) in enumerate(indexed_operators):
        operator_name = str(operator.metadata.operator_name)
        config = _operator_config(operator.metadata)
        with torch.cuda.device(device):
            artifact = operator.compile(args, target_sm=target_sm)
        compiled_for = str(artifact.compiled_for)
        candidates.append(
            _CutlassCandidate(
                registry_index=registry_index,
                operator_name=operator_name,
                compiled_for=compiled_for,
                prepared=_prepared_candidate(
                    inputs,
                    args,
                    output,
                    operator,
                    artifact,
                    name=f"{CUTLASS_OPERATOR_BASELINE}-candidate-{candidate_index}",
                    config=config,
                ),
            )
        )

    selection = _tune_candidates(candidates, inputs.oracle)
    selected_index = selection.get("selected_candidate_index")
    if type(selected_index) is not int or not 0 <= selected_index < len(candidates):
        raise RuntimeError("CUTLASS tuning returned an invalid selected candidate")
    selected = candidates[selected_index]
    return replace(
        selected.prepared,
        name=f"{CUTLASS_OPERATOR_BASELINE}_cold_l2_tuned",
        config=common.provider_config(
            "cutlass",
            {
                "baseline": CUTLASS_OPERATOR_BASELINE,
                "benchmark_label": "CUTLASS supported-operator cold-L2 tuned",
                "repository": CUTLASS_REPOSITORY,
                "release_tag": CUTLASS_RELEASE_TAG,
                "commit": commit,
                "operator_api_version": operator_api_version,
                "global_options": {"use_tvm_ffi": True},
                "target_sm": target_sm,
                "registry_tuning": selection,
                "a_layout": common.compact_contiguous_a_layout(),
                "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
            },
        ),
    )
