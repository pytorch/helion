"""Compare Helion grouped GEMM with reproducible public provider baselines.

The public command launches one fresh worker process per provider/replicate.
Each worker validates and captures both implementations, performs a 10-second
thermal warmup, and records 102 cold-L2 paired samples with balanced ordering.
Compilation, packing, provider selection, and Helion selection stay outside
the timed region.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
import importlib.metadata
import json
import math
from operator import itemgetter
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from benchmarks.cute import cublaslt_grouped_gemm as cublaslt
from benchmarks.cute import cudnn_grouped_gemm as cudnn
from benchmarks.cute import cutlass_contiguous_grouped_gemm as cutlass
from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute import quack_grouped_gemm as quack
from benchmarks.cute.grouped_gemm_workloads import BENCHMARK_B_LAYOUT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_CUTLASS_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_FMT_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_VERSION
from benchmarks.cute.grouped_gemm_workloads import PROVIDER_CLI_MODES
from benchmarks.cute.grouped_gemm_workloads import PROVIDER_DEFAULTS_MODE
from benchmarks.cute.grouped_gemm_workloads import PROVIDER_DEFAULTS_PLAN_MODE
from benchmarks.cute.grouped_gemm_workloads import PROVIDER_DEFAULTS_WORKER_MODE
from benchmarks.cute.grouped_gemm_workloads import PROVIDER_SELECTION_MODES
import torch

hl: Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmCase
    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation

    from helion.runtime.kernel import Kernel


REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_SCHEMA = "helion-grouped-gemm-provider-defaults-v10"
SUMMARY_SCHEMA = "helion-grouped-gemm-provider-summary-v15"
BENCHMARK_REPETITIONS = 102
PUBLICATION_REPLICATES = 3
THERMAL_WARMUP_MS = 10_000
CAPTURE_WARMUPS = 2
TELEMETRY_INTERVAL_SECONDS = 5
COMPUTE_APPLICATION_INTERVAL_SECONDS = 1
POST_WORKER_IDLE_GRACE_SECONDS = 5
NVCC_RELEASE_PATTERN = re.compile(r"\brelease\s+(\d+\.\d+),\s+V([0-9.]+)")
PROVIDERS = ("deepgemm", "quack", "cudnn", "cublaslt", "cutlass")
SOURCE_BACKED_PROVIDERS = ("deepgemm", "quack", "cutlass")
QUACK_SELECTION_FIELDS = frozenset(
    {
        "selected_config",
        "resolved_dynamic_scheduler",
        "resolved_split_k",
        "dispatch_plan",
    }
)
CUTLASS_SELECTION_MEASUREMENT_FIELDS = frozenset({"selection_median_ms"})
_MISSING = object()


def _provider_requirements(provider: str) -> dict[str, object]:
    if provider == "deepgemm":
        return {
            "api": {
                "function": "m_grouped_bf16_gemm_nt_contiguous",
                "compiled_dims": "nk",
                "use_psum_layout": False,
                "ensure_zero_padding": False,
            },
            "provenance.git_head": DEEPGEMM_COMMIT,
            "provenance.cutlass_head": DEEPGEMM_CUTLASS_COMMIT,
            "provenance.fmt_head": DEEPGEMM_FMT_COMMIT,
            "provenance.version": DEEPGEMM_VERSION,
            "provenance.native_extension_sha256": str,
        }
    if provider == "quack":
        return {
            "benchmark_label": quack.QUACK_BENCHMARK_LABEL,
            "selection_api": "gemm(default tuned=True)",
            "requested_config": None,
            "requested_dynamic_scheduler": False,
            "tuned": True,
            "resolved_split_k": 1,
            "selected_config.b_layout": BENCHMARK_B_LAYOUT,
            "dispatch_plan.type": "quack.gemm._GemmPlan",
            "package.source_provenance": {
                "kind": "upstream_main_snapshot",
                "repository": quack.QUACK_REPOSITORY,
                "commit": quack.QUACK_COMMIT,
                "base_release_tag": quack.QUACK_BASE_RELEASE_TAG,
                "is_formal_release": False,
                "benchmark_label": quack.QUACK_BENCHMARK_LABEL,
            },
            "package.upstream_commit": quack.QUACK_COMMIT,
            "package.distribution_version": quack.QUACK_PACKAGE_METADATA_VERSION,
            "package.dependency_versions": quack.QUACK_DEPENDENCY_VERSIONS,
            "package.module_version": quack.QUACK_PACKAGE_METADATA_VERSION,
            "package.installation": str,
        }
    if provider == "cudnn":
        return {
            "baseline": cudnn.CUDNN_GROUPED_BASELINE,
            "frontend_version": cudnn.CUDNN_FRONTEND_VERSION,
            "backend_version": cudnn.CUDNN_BACKEND_VERSION,
            "runtime.frontend": {
                "distribution": cudnn.CUDNN_FRONTEND_DISTRIBUTION,
                "package_version": cudnn.CUDNN_FRONTEND_VERSION,
            },
            "runtime.requested_cuda_runtime": {
                "distribution": common.CUDA_RUNTIME_DISTRIBUTION,
                "package_version": common.CUDA_RUNTIME_VERSION,
            },
            "plan.selection": "graph_build_default",
        }
    if provider == "cublaslt":
        return {
            "library": {
                "distribution": cublaslt.CUBLASLT_DISTRIBUTION,
                "package_version": cublaslt.CUBLASLT_DISTRIBUTION_VERSION,
                "library_version": cublaslt.CUBLASLT_LIBRARY_VERSION,
            },
            "heuristic_query_capacity": cublaslt.CUBLASLT_HEURISTIC_QUERY_CAPACITY,
            "group_count": int,
            "grouped_average_preferences": dict,
            "selected_algorithm.serialized_hex": str,
            "selected_algorithm.heuristic_rank": 0,
        }
    if provider == "cutlass":
        return {
            "repository": cutlass.CUTLASS_REPOSITORY,
            "release_tag": cutlass.CUTLASS_RELEASE_TAG,
            "commit": cutlass.CUTLASS_COMMIT,
            "operator_api_version": cutlass.CUTLASS_OPERATOR_API_VERSION,
            "target_sm": str,
            "registry_tuning": dict,
        }
    return {}


def publication_runs() -> tuple[tuple[str, int], ...]:
    """Return the fixed execution order shared by the plan and controller."""

    return tuple(
        (provider, replicate)
        for replicate in range(PUBLICATION_REPLICATES)
        for provider in PROVIDERS
    )


def _quack_provenance(config: dict[str, Any]) -> dict[str, object]:
    """Return provider identity excluding QuACK's fresh autotune selection."""

    missing = QUACK_SELECTION_FIELDS - config.keys()
    if missing:
        raise RuntimeError(
            f"QuACK configuration is missing selection fields: {sorted(missing)}"
        )
    return {
        key: value for key, value in config.items() if key not in QUACK_SELECTION_FIELDS
    }


def _cutlass_provenance(config: dict[str, Any]) -> dict[str, object]:
    """Return CUTLASS selection identity without replicate measurements."""

    tuning = config.get("registry_tuning")
    if not isinstance(tuning, dict) or not isinstance(tuning.get("candidates"), list):
        raise RuntimeError("CUTLASS configuration is missing registry tuning evidence")
    candidates = []
    for candidate in tuning["candidates"]:
        if not isinstance(candidate, dict):
            raise RuntimeError("CUTLASS registry candidate evidence is invalid")
        candidates.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in CUTLASS_SELECTION_MEASUREMENT_FIELDS
            }
        )
    stable_tuning = {**tuning, "candidates": candidates}
    return {**config, "registry_tuning": stable_tuning}


def _all_equal(values: Sequence[object]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def _nested_value(value: dict[str, Any], path: str) -> object:
    current: object = value
    for field in path.split("."):
        if not isinstance(current, dict) or field not in current:
            return _MISSING
        current = current[field]
    return current


def _matches_requirement(value: object, required: object) -> bool:
    if isinstance(required, type):
        return type(value) is required and (required is not str or bool(value))
    if isinstance(required, dict):
        return (
            isinstance(value, dict)
            and value.keys() == required.keys()
            and all(
                _matches_requirement(value[key], item) for key, item in required.items()
            )
        )
    return type(value) is type(required) and value == required


def _valid_provider_contract(
    value: object,
    provider: str,
    *,
    expected_case: dict[str, Any] | None = None,
    device: dict[str, Any] | None = None,
    complete: bool = False,
) -> bool:
    requirements = _provider_requirements(provider)
    if not isinstance(value, dict) or not requirements:
        return False
    if not all(
        type(actual := value.get(key)) is type(required) and actual == required
        for key, required in common.provider_config(provider, {}).items()
    ) or not all(
        _matches_requirement(_nested_value(value, path), required)
        for path, required in requirements.items()
    ):
        return False
    if provider == "deepgemm":
        return (
            re.fullmatch(
                r"[0-9a-f]{64}", value["provenance"]["native_extension_sha256"]
            )
            is not None
        )
    if provider == "quack":
        return value["package"]["installation"] in {
            "editable",
            "installed_distribution_with_source_override",
        }
    if provider == "cudnn":
        runtime = value["runtime"]
        if not complete:
            return True
        return (
            runtime.get("backend_libraries")
            == {
                "distribution": cudnn.CUDNN_BACKEND_DISTRIBUTION,
                "package_version": cudnn.CUDNN_BACKEND_DISTRIBUTION_VERSION,
            }
            and runtime.get("loaded_cuda_runtime") == runtime["requested_cuda_runtime"]
        )
    if provider == "cublaslt":
        valid = value["group_count"] > 0
        if not valid or expected_case is None:
            return valid
        problems = tuple(
            (m, expected_case["n"], expected_case["k"], 1)
            for m in expected_case["actual_ms"]
        )
        return value["group_count"] == expected_case["groups"] and value[
            "grouped_average_preferences"
        ] == cublaslt.cublaslt_grouped_preference_values(problems)
    target_sm = value["target_sm"]
    return (
        isinstance(target_sm, str)
        and target_sm in cutlass.CUTLASS_TARGET_SMS.values()
        and (
            device is None
            or target_sm == cutlass.CUTLASS_TARGET_SMS.get(tuple(device["capability"]))
        )
        and _valid_cutlass_tuning_evidence(value)
    )


@dataclass(frozen=True)
class _ComputeApplication:
    pid: int
    process_name: str


HELION_SELECTION = "compiler_heuristic"
GROUPED_WORKLIST_HEURISTIC = "cute_tcgen05_grouped_worklist"
SOURCE_M_TILE = 256
BLOCK_M = 256
BLOCK_N = 128
BLOCK_K_CHOICES = (64, 128)
BENCHMARK_LAYOUT_POLICY = f"canonical_{BENCHMARK_B_LAYOUT}_for_all_implementations"
_COMMON_PROTOCOL = {
    "capture_warmups": CAPTURE_WARMUPS,
    "thermal_warmup_ms": THERMAL_WARMUP_MS,
    "paired_cold_l2_samples": BENCHMARK_REPETITIONS,
    "fresh_process_and_caches_per_provider_replicate": True,
    "balanced_rotated_reversed_order": True,
    "correctness_rtol": common.CORRECTNESS_RTOL,
    "correctness_atol": common.CORRECTNESS_ATOL,
    "max_normalized_diff": common.CORRECTNESS_MAX_NORMALIZED_DIFF,
    "poison_replay_rewrite_check": True,
    "exact_repeat_replay_check": True,
    "helion_padding_zero_check": True,
}
HELION_SELECTION_ENVIRONMENT_VARIABLES = (
    "HELION_AUTOTUNE_EFFORT",
    "HELION_BACKEND",
    "HELION_CUTE_MMA_IMPL",
)
HELION_SELECTION_STARTUP_ENVIRONMENT = {
    name: os.environ.get(name) for name in HELION_SELECTION_ENVIRONMENT_VARIABLES
}
WORKER_CACHE_NAMES = {
    "CUDA_CACHE_PATH": "cuda",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "DG_JIT_CACHE_DIR": "deepgemm_jit",
    "HELION_CACHE_DIR": "helion",
    "QUACK_CACHE_DIR": "quack",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton",
    "XDG_CACHE_HOME": "xdg",
}
WORKER_CONTROL_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "CUTE_DSL_",
    "CUTLASS_",
    "DG_",
    "HELION_",
    "LD_",
    "NVIDIA_",
    "QUACK_",
    "PYTORCH_",
    "TORCH_",
    "TORCHINDUCTOR_",
    "TRITON_",
)
COMPILER_AND_LOADER_CONTROLS = frozenset(
    {
        "CC",
        "CFLAGS",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_C_COMPILER",
        "CMAKE_INCLUDE_PATH",
        "CMAKE_LIBRARY_PATH",
        "CMAKE_PREFIX_PATH",
        "CMAKE_TOOLCHAIN_FILE",
        "COMPILER_PATH",
        "CPATH",
        "CPPFLAGS",
        "CPLUS_INCLUDE_PATH",
        "CUDACXX",
        "CUDNN_FRONTEND_CUDART_LIB_NAME",
        "CUDAFLAGS",
        "CUDAHOSTCXX",
        "CXX",
        "CXXFLAGS",
        "C_INCLUDE_PATH",
        "GCC_EXEC_PREFIX",
        "GLIBC_TUNABLES",
        "LDFLAGS",
        "LIBRARY_PATH",
        "NVCC",
        "NVCC_APPEND_FLAGS",
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
        "OBJC_INCLUDE_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_SYSROOT_DIR",
        "SDKROOT",
        "TORCH_CUDA_ARCH_LIST",
    }
)
TELEMETRY_FIELDS = (
    "uuid",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "power.limit",
    "clocks_event_reasons.active",
)
WORKER_TERMINATION_GRACE_SECONDS = 5
NVIDIA_SMI_TIMEOUT_SECONDS = 10
GPU_IDLE_CLOCK_EVENT_REASON = 0x1
SW_POWER_CAP_CLOCK_EVENT_REASON = 0x4
ALLOWED_CLOCK_EVENT_REASONS = (
    GPU_IDLE_CLOCK_EVENT_REASON | SW_POWER_CAP_CLOCK_EVENT_REASON
)


class _CampaignInterrupted(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@dataclass(frozen=True)
class _TelemetrySample:
    pstate: str
    sm_clock_mhz: float
    memory_clock_mhz: float
    power_draw_watts: float
    power_limit_watts: float
    active_clock_event_reasons: int


def _make_helion_kernel() -> Kernel[torch.Tensor]:
    """Create the benchmark kernel through the ordinary compiler path."""

    global hl

    import helion
    import helion.language as hl

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def grouped_gemm(
        a_packed: torch.Tensor,
        b_grouped: torch.Tensor,
        worklist: torch.Tensor,
    ) -> torch.Tensor:
        m_total, k = a_packed.shape
        _groups, n, k2 = b_grouped.shape
        assert k == k2
        assert worklist.size(1) == 4
        block_m = hl.register_block_size(BLOCK_M)
        block_n = hl.register_block_size(BLOCK_N)
        block_k = hl.register_block_size(*BLOCK_K_CHOICES)
        out = torch.empty(
            (m_total, n),
            dtype=a_packed.dtype,
            device=a_packed.device,
        )
        for work_tile, tile_m, tile_n in hl.tile(
            [worklist.size(0), BLOCK_M, n],
            block_size=[1, block_m, block_n],
        ):
            work_id = work_tile.begin
            group_id = worklist[work_id, 0]
            global_m_start = worklist[work_id, 1]
            valid_m = worklist[work_id, 2]
            store_m = worklist[work_id, 3]
            local_m = tile_m.index
            row_index = global_m_start + local_m
            valid_rows = local_m < valid_m
            store_rows = local_m < store_m
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k, block_size=block_k):
                a_block = hl.load(
                    a_packed,
                    [row_index, tile_k],
                    extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
                )
                acc = torch.addmm(
                    acc,
                    a_block,
                    b_grouped[group_id, tile_n, tile_k].T,
                )
            hl.store(
                out,
                [row_index, tile_n],
                acc.to(out.dtype),
                extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
            )
        return out

    return grouped_gemm


def prepare_helion(
    inputs: GroupedGemmInputs,
) -> PreparedImplementation:
    """Bind Helion and select its ordinary compiler default."""

    packed = common.pack_compact_rows(inputs, SOURCE_M_TILE)
    b = inputs.b
    kernel_args = (
        packed.a,
        b,
        packed.worklist,
    )
    kernel = _make_helion_kernel()
    bound = kernel.bind(kernel_args)
    config_spec = bound.config_spec
    compiler_default = config_spec.compiler_default_config
    if compiler_default is None:
        raise RuntimeError(
            "grouped GEMM compiler heuristics produced no primary config"
        )
    compiler_default_config = compiler_default.config
    if (
        compiler_default_config.get("tcgen05_grouped_mode") != "worklist_nm"
        or compiler_default_config.get("tcgen05_grouped_worklist_source_m_tile")
        != SOURCE_M_TILE
    ):
        raise RuntimeError("compiler primary config is not grouped-worklist NM")
    if config_spec.autotuner_heuristics.count(GROUPED_WORKLIST_HEURISTIC) != 1:
        raise RuntimeError(
            "grouped-worklist compiler heuristic did not fire exactly once"
        )
    selected_config = config_spec.default_config()
    selected_effective = dict(selected_config.config)
    if any(
        key not in selected_effective or selected_effective[key] != value
        for key, value in compiler_default_config.items()
    ):
        raise RuntimeError("effective config is not the grouped-worklist default")
    bound.set_config(selected_config)
    selection_evidence = {
        "selection_mode": HELION_SELECTION,
        "autotuned": False,
        "b_layout": BENCHMARK_B_LAYOUT,
        "selection_api": "BoundKernel.set_config(ConfigSpec.default_config())",
        "config": selected_effective,
        "a_layout": {
            "kind": "aligned_contiguous_worklist",
            "alignment": SOURCE_M_TILE,
            "logical_values_bitwise_equal": True,
        },
    }
    torch.cuda.synchronize()

    def output_tensors(result: object) -> tuple[torch.Tensor, ...]:
        if not isinstance(result, torch.Tensor):
            raise TypeError("Helion grouped GEMM returned a non-tensor")
        return (result,)

    def logical_outputs(result: object) -> tuple[torch.Tensor, ...]:
        if not isinstance(result, torch.Tensor):
            raise TypeError("Helion grouped GEMM returned a non-tensor")
        if not packed.output_padding_is_zero(result):
            raise RuntimeError("Helion did not zero aligned output padding")
        return packed.output_slices(result)

    def call() -> torch.Tensor:
        return bound(*kernel_args)

    return common.PreparedImplementation(
        name=f"helion-compiler-heuristic-{inputs.case.id}",
        call=call,
        output_tensors=output_tensors,
        logical_outputs=logical_outputs,
        config=selection_evidence,
        track_cute_graph=True,
    )


def prepare_provider_default(
    provider: str,
    inputs: GroupedGemmInputs,
    *,
    cutlass_root: Path | None,
    deepgemm_root: Path | None,
    quack_root: Path | None = None,
) -> PreparedImplementation:
    """Prepare one documented provider comparison implementation."""

    if provider == "deepgemm":
        from benchmarks.cute.grouped_gemm_deepgemm_support import (
            prepare_deepgemm_default,
        )

        if deepgemm_root is None:
            raise ValueError(
                "--provider-deepgemm-root is required for the DeepGEMM provider"
            )
        prepared = prepare_deepgemm_default(
            inputs,
            deepgemm_root=deepgemm_root,
        )
    elif provider == "cutlass":
        from benchmarks.cute.cutlass_contiguous_grouped_gemm import (
            prepare_cutlass_default,
        )

        if cutlass_root is None:
            raise ValueError(
                "--provider-cutlass-root is required for the CUTLASS provider"
            )
        prepared = prepare_cutlass_default(inputs, cutlass_root=cutlass_root)
    elif provider == "quack":
        from benchmarks.cute.quack_grouped_gemm import prepare_quack_default

        prepared = prepare_quack_default(
            inputs,
            quack_root=quack_root,
        )
    elif provider == "cudnn":
        from benchmarks.cute.cudnn_grouped_gemm import prepare_cudnn_default

        prepared = prepare_cudnn_default(inputs)
    elif provider == "cublaslt":
        from benchmarks.cute.cublaslt_grouped_gemm import prepare_cublaslt_default

        prepared = prepare_cublaslt_default(inputs)
    else:
        raise ValueError(f"unsupported provider {provider!r}")
    if not _valid_provider_contract(prepared.config, provider):
        raise RuntimeError(f"{provider} produced an invalid selection contract")
    return prepared


def _validated_capture(
    prepared: PreparedImplementation,
    oracle: Sequence[torch.Tensor],
) -> tuple[common.CapturedImplementation, dict[str, Any]]:
    captured = common.capture_implementation(prepared, warmups=CAPTURE_WARMUPS)
    correctness = common.validate_capture(captured, oracle)
    if not correctness["ok"]:
        raise RuntimeError(f"{prepared.name} failed correctness: {correctness}")
    return captured, correctness


def run_case(
    provider: str,
    case: GroupedGemmCase,
    device: torch.device,
    *,
    cutlass_root: Path | None,
    deepgemm_root: Path | None,
    quack_root: Path | None = None,
) -> dict[str, object]:
    """Validate and time one selected Helion/provider-default pair."""

    from pretuned_kernels import _bench

    inputs = common.make_inputs(case, device, seed=case.row_index)
    # Provider imports, compilation, graph capture, and timing must not perturb
    # the deterministic per-row input streams.
    with torch.random.fork_rng(devices=[device.index]):
        helion = prepare_helion(inputs)
        provider_impl = prepare_provider_default(
            provider,
            inputs,
            cutlass_root=cutlass_root,
            deepgemm_root=deepgemm_root,
            quack_root=quack_root,
        )
        helion_capture, helion_correctness = _validated_capture(helion, inputs.oracle)
        provider_capture, provider_correctness = _validated_capture(
            provider_impl,
            inputs.oracle,
        )
    _bench.thermal_warmup(THERMAL_WARMUP_MS)
    with torch.random.fork_rng(devices=[device.index]):
        helion_ms, provider_ms = _bench.bench_pre_captured_cudagraphs(
            [helion_capture.replay, provider_capture.replay],
            rep=BENCHMARK_REPETITIONS,
        )
    post_timing_checks = {
        "helion": common.check_correctness(helion_capture, inputs.oracle),
        "provider": common.check_correctness(provider_capture, inputs.oracle),
    }
    for implementation, check in post_timing_checks.items():
        if not check["ok"]:
            raise RuntimeError(
                f"{implementation} output changed during benchmark timing: {check}"
            )
    helion_correctness["post_timing"] = post_timing_checks["helion"]
    provider_correctness["post_timing"] = post_timing_checks["provider"]
    if not all(
        math.isfinite(value) and value > 0.0 for value in (helion_ms, provider_ms)
    ):
        raise RuntimeError("benchmark timings must be finite and positive")
    return {
        "case": case.as_dict(),
        "configs": {
            "helion": helion.config,
            "provider": provider_impl.config,
        },
        "correctness": {
            "helion": helion_correctness,
            "provider": provider_correctness,
        },
        "timings": {
            "helion_ms": helion_ms,
            "provider_ms": provider_ms,
        },
    }


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(REPO_ROOT), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _source_identity() -> str:
    if _git_value("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("benchmark requires a clean Helion checkout")
    return _git_value("rev-parse", "HEAD")


def _device_info(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if not common.is_supported_grouped_gemm_device(
        device.type, properties.name, capability
    ):
        raise RuntimeError(
            "grouped-GEMM defaults benchmark requires NVIDIA B200/SM100 or "
            "NVIDIA GB300/SM103, got "
            f"{properties.name!r} capability {capability}"
        )
    uuid = str(properties.uuid)
    return {
        "visible": common.require_single_visible_device(),
        "name": properties.name,
        "capability": list(capability),
        "uuid": uuid if uuid.startswith("GPU-") else f"GPU-{uuid}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory": properties.total_memory,
    }


def _worker_cache_directories(run_dir: Path) -> dict[str, str]:
    caches = {}
    for variable, relative in WORKER_CACHE_NAMES.items():
        path = run_dir / "cache" / relative
        path.mkdir(parents=True)
        caches[variable] = str(path)
    return caches


def _selection_environment() -> dict[str, str]:
    return {
        "HELION_AUTOTUNE_EFFORT": "none",
        "HELION_BACKEND": "cute",
        "HELION_CUTE_MMA_IMPL": "tcgen05",
    }


def _restore_provider_import_path(provider: str, quack_root: Path | None) -> None:
    """Restore a validated source override removed by direct-script startup."""

    if provider != "quack" or quack_root is None:
        return
    root = str(quack_root.resolve(strict=True))
    if root not in sys.path:
        sys.path.insert(1, root)


def _run_worker(args: argparse.Namespace) -> int:
    _restore_provider_import_path(args.provider, args.quack_root)
    selection_environment = _selection_environment()
    expected_selection_environment = {
        key: selection_environment.get(key)
        for key in HELION_SELECTION_ENVIRONMENT_VARIABLES
    }
    if expected_selection_environment != HELION_SELECTION_STARTUP_ENVIRONMENT:
        raise RuntimeError("Helion selection environment was not set before startup")
    if os.environ.get("HELION_HEURISTIC_DIR"):
        raise RuntimeError(
            "--provider-defaults-worker does not permit HELION_HEURISTIC_DIR"
        )
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or (run_dir / "result.json").exists():
        raise RuntimeError(f"worker requires a fresh run directory: {run_dir}")

    common.require_single_visible_device()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    device_info = _device_info(device)
    cuda_stack = _cuda_toolchain_identity()
    common.configure_oracle_precision()

    cases = common.official_cases()
    if len(cases) != 8:
        raise RuntimeError("benchmark requires exactly eight official cases")
    rows = [
        run_case(
            args.provider,
            case,
            device,
            cutlass_root=args.cutlass_root,
            deepgemm_root=args.deepgemm_root,
            quack_root=args.quack_root,
        )
        for case in cases
    ]
    _validate_mapped_cuda_libraries(_installed_cuda_stack()[1])
    result = {
        "schema": RESULT_SCHEMA,
        "provider": args.provider,
        "replicate": args.replicate,
        "device": device_info,
        "source": _source_identity(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "triton": importlib.metadata.version("triton"),
            "cuda_driver": _cuda_driver_version(cast("str", device_info["uuid"])),
            "cuda_stack": cuda_stack,
        },
        "rows": rows,
    }
    common.write_result(run_dir / "result.json", result)
    return 0


def _provider_roots(args: argparse.Namespace) -> dict[str, Path]:
    roots = {
        provider: cast("Path", vars(args)[f"{provider}_root"]).expanduser().resolve()
        for provider in SOURCE_BACKED_PROVIDERS
    }
    missing = [
        f"{provider}: {root}" for provider, root in roots.items() if not root.is_dir()
    ]
    if missing:
        raise ValueError("provider roots must be directories: " + "; ".join(missing))
    return roots


def _worker_environment(
    run_dir: Path,
    *,
    cuda_visible_devices: str,
    quack_root: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith((*WORKER_CONTROL_PREFIXES, "PYTHON")) or (
            name in COMPILER_AND_LOADER_CONTROLS
        ):
            environment.pop(name)
    cuda_home, artifacts = _installed_cuda_stack()
    cudart = artifacts["cudart"]
    environment.update(_selection_environment())
    environment.update(_worker_cache_directories(run_dir))
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["CUDA_HOME"] = str(cuda_home)
    environment["CUDA_PATH"] = str(cuda_home)
    environment["CUDNN_FRONTEND_CUDART_LIB_NAME"] = str(cudart)
    environment["LD_PRELOAD"] = os.pathsep.join(
        str(artifacts[name]) for name in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES
    )
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PATH"] = os.pathsep.join(
        (
            str(cuda_home / "bin"),
            str(Path(sys.executable).resolve().parent),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        )
    )
    python_paths = [str(REPO_ROOT)]
    if quack_root is not None:
        python_paths.append(str(quack_root.resolve(strict=True)))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def _cuda_toolkit_identity(cuda_home: Path) -> dict[str, str]:
    nvcc = cuda_home / "bin" / "nvcc"
    completed = subprocess.run(
        (str(nvcc), "--version"),
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    match = NVCC_RELEASE_PATTERN.search(output)
    if completed.returncode or match is None:
        raise RuntimeError(f"could not identify CUDA toolkit nvcc at {nvcc}")
    return {
        "release": match.group(1),
        "compiler_version": match.group(2),
    }


def _installed_cuda_stack() -> tuple[Path, dict[str, Path]]:
    distributions = common.require_pinned_distributions(
        common.CUDA_STACK_DISTRIBUTION_VERSIONS,
        "CUDA stack is not pinned",
    )
    artifacts = {
        label: common.distribution_file(distribution, distributions[distribution], path)
        for label, (distribution, path) in common.CUDA_STACK_REQUIRED_ARTIFACTS.items()
    }
    roots = {
        artifact.parents[len(common.CUDA_STACK_REQUIRED_ARTIFACTS[label][1].parts) - 3]
        for label, artifact in artifacts.items()
    }
    if len(roots) != 1:
        raise RuntimeError(
            f"CUDA stack distributions use different roots: {sorted(map(str, roots))}"
        )
    cuda_home = roots.pop()
    return cuda_home, artifacts


def _validate_mapped_cuda_libraries(artifacts: dict[str, Path]) -> None:
    for label, prefix in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES.items():
        expected = artifacts[label]
        loaded = common.mapped_library_paths(prefix)
        if loaded != (expected,):
            raise RuntimeError(
                f"worker loaded {label} libraries {tuple(map(str, loaded))}, "
                f"expected {(str(expected),)}"
            )


def _cuda_toolchain_identity() -> dict[str, object]:
    expected_cuda_home, artifacts = _installed_cuda_stack()
    expected_cudart = artifacts["cudart"]
    cuda_home_text = os.environ.get("CUDA_HOME")
    cudart_text = os.environ.get("CUDNN_FRONTEND_CUDART_LIB_NAME")
    if cuda_home_text is None or cudart_text is None:
        raise RuntimeError("worker CUDA toolkit environment is incomplete")
    cuda_home = Path(cuda_home_text).resolve(strict=True)
    cudart = Path(cudart_text).resolve(strict=True)
    if cuda_home != expected_cuda_home or cudart != expected_cudart:
        raise RuntimeError("worker CUDA toolkit environment changed after validation")
    _validate_mapped_cuda_libraries(artifacts)
    toolkit = _cuda_toolkit_identity(cuda_home)
    if (
        toolkit["release"] != common.CUDA_TOOLKIT_RELEASE
        or toolkit["compiler_version"] != common.CUDA_COMPILER_VERSION
    ):
        raise RuntimeError(
            "nvcc reports release "
            f"{toolkit['release']} V{toolkit['compiler_version']}, expected release "
            f"{common.CUDA_TOOLKIT_RELEASE} V{common.CUDA_COMPILER_VERSION}"
        )
    return {
        "release": toolkit["release"],
        "compiler_version": toolkit["compiler_version"],
        "distribution_versions": dict(common.CUDA_STACK_DISTRIBUTION_VERSIONS),
    }


def _worker_command(
    provider: str,
    replicate: int,
    run_dir: Path,
    roots: dict[str, Path],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-X",
        f"pycache_prefix={run_dir / 'cache' / 'pycache'}",
        str(Path(__file__).with_name("compare_grouped_gemm_backends.py").resolve()),
        "--provider-defaults-worker",
        "--provider",
        provider,
        "--provider-replicate",
        str(replicate),
        "--provider-run-dir",
        str(run_dir),
    ]
    if provider in roots:
        command.extend((f"--provider-{provider}-root", str(roots[provider])))
    return command


def _nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ("nvidia-smi", *arguments),
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed ({completed.returncode}): {detail}")
    return completed.stdout


def _cuda_driver_version(device_uuid: str) -> str:
    output = _nvidia_smi(
        "-i",
        device_uuid,
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"could not identify CUDA driver for {device_uuid}")
    return rows[0]


def _resolve_target_gpu(cuda_visible_devices: str) -> str:
    selector = common.require_single_visible_device(cuda_visible_devices)
    output = _nvidia_smi(
        "-i",
        selector,
        "--query-gpu=uuid",
        "--format=csv,noheader,nounits",
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or not rows[0].startswith("GPU-"):
        raise RuntimeError("target GPU UUID query did not return one physical GPU")
    return rows[0]


def _query_compute_applications(
    target_gpu_uuid: str,
) -> tuple[_ComputeApplication, ...]:
    output = _nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    )
    applications: list[_ComputeApplication] = []
    for row in (line.strip() for line in output.splitlines() if line.strip()):
        fields = tuple(field.strip() for field in row.split(",", maxsplit=2))
        if len(fields) != 3:
            raise RuntimeError("compute-application query returned malformed CSV")
        if fields[0] == target_gpu_uuid:
            try:
                pid = int(fields[1])
            except ValueError as error:
                raise RuntimeError(
                    "compute-application query returned an invalid PID"
                ) from error
            if pid <= 0:
                raise RuntimeError("compute-application query returned an invalid PID")
            applications.append(
                _ComputeApplication(
                    pid=pid,
                    process_name=fields[2],
                )
            )
    return tuple(applications)


def _require_target_gpu_idle(target_gpu_uuid: str) -> None:
    applications = _query_compute_applications(target_gpu_uuid)
    if applications:
        application = applications[0]
        raise RuntimeError(
            "target GPU is not idle: "
            f"{target_gpu_uuid}, {application.pid}, {application.process_name}"
        )


def _check_compute_applications(
    *,
    target_gpu_uuid: str,
    worker_process_group: int,
    known_worker_pids: set[int],
) -> None:
    foreign: list[tuple[_ComputeApplication, int | None]] = []
    for application in _query_compute_applications(target_gpu_uuid):
        try:
            process_group: int | None = os.getpgid(application.pid)
        except ProcessLookupError:
            process_group = None
        if process_group == worker_process_group:
            known_worker_pids.add(application.pid)
        elif not (process_group is None and application.pid in known_worker_pids):
            foreign.append((application, process_group))
    if foreign:
        details = "; ".join(
            f"pid={application.pid} pgid={process_group} "
            f"name={application.process_name!r}"
            for application, process_group in foreign
        )
        raise RuntimeError(
            f"target GPU {target_gpu_uuid} has a compute application outside "
            f"worker process group {worker_process_group}: {details}"
        )


def _query_telemetry(target_gpu_uuid: str) -> _TelemetrySample:
    fields = ",".join(TELEMETRY_FIELDS)
    output = _nvidia_smi(
        "-i",
        target_gpu_uuid,
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi telemetry query did not return one row")
    values = tuple(field.strip() for field in rows[0].split(","))
    if len(values) != len(TELEMETRY_FIELDS) or values[0] != target_gpu_uuid:
        raise RuntimeError("telemetry query did not return the target GPU UUID")
    try:
        return _TelemetrySample(
            pstate=values[1],
            sm_clock_mhz=float(values[2]),
            memory_clock_mhz=float(values[3]),
            power_draw_watts=float(values[4]),
            power_limit_watts=float(values[5]),
            active_clock_event_reasons=int(values[6], 0),
        )
    except ValueError as error:
        raise RuntimeError(
            "telemetry query returned an invalid numeric value"
        ) from error


def _summarize_telemetry(
    samples: Sequence[_TelemetrySample],
) -> dict[str, object]:
    if not samples:
        raise RuntimeError("GPU telemetry is empty")
    if any(not sample.pstate or sample.pstate.startswith("[") for sample in samples):
        raise RuntimeError("GPU telemetry returned an invalid pstate")
    if any(
        any(
            not math.isfinite(value) or value < 0
            for value in (
                sample.power_draw_watts,
                sample.sm_clock_mhz,
                sample.memory_clock_mhz,
            )
        )
        for sample in samples
    ):
        raise RuntimeError("GPU telemetry contains an invalid power or clock value")
    power_limits = {sample.power_limit_watts for sample in samples}
    if any(not math.isfinite(value) or value <= 0 for value in power_limits):
        raise RuntimeError("GPU telemetry contains an invalid power limit")
    if len(power_limits) != 1:
        raise RuntimeError("GPU power limit changed")
    active_reasons = Counter(
        f"0x{sample.active_clock_event_reasons:x}"
        for sample in samples
        if sample.active_clock_event_reasons
    )
    disallowed = Counter(
        f"0x{reason:x}"
        for sample in samples
        if (
            reason := (sample.active_clock_event_reasons & ~ALLOWED_CLOCK_EVENT_REASONS)
        )
    )
    if disallowed:
        raise RuntimeError(
            "GPU telemetry reported disallowed clock-event reasons: "
            f"{dict(sorted(disallowed.items()))}"
        )

    def metric(values: Sequence[float]) -> dict[str, float]:
        return {
            "minimum": min(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
        }

    return {
        "sample_count": len(samples),
        "power_draw_watts": metric([s.power_draw_watts for s in samples]),
        "sm_clock_mhz": metric([s.sm_clock_mhz for s in samples]),
        "memory_clock_mhz": metric([s.memory_clock_mhz for s in samples]),
        "power_limit_watts": next(iter(power_limits)),
        "pstates": dict(sorted(Counter(sample.pstate for sample in samples).items())),
        "sample_scope": "whole_worker_including_setup_and_post_exit",
        "active_clock_event_reason_sample_count": sum(active_reasons.values()),
        "active_clock_event_reasons": dict(sorted(active_reasons.items())),
    }


def _wait_for_target_gpu_idle(target_gpu_uuid: str) -> None:
    deadline = time.monotonic() + POST_WORKER_IDLE_GRACE_SECONDS
    while True:
        try:
            _require_target_gpu_idle(target_gpu_uuid)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
        else:
            return


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    previous_interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
    previous_terminate = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        if not _process_group_exists(process.pid):
            process.wait()
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + WORKER_TERMINATION_GRACE_SECONDS
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.1)
        if _process_group_exists(process.pid):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_interrupt)
        signal.signal(signal.SIGTERM, previous_terminate)


def _run_monitored_worker(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    target_gpu_uuid: str,
) -> tuple[int, list[_TelemetrySample], int]:
    telemetry: list[_TelemetrySample] = []
    compute_application_samples = 0
    worker_error: BaseException | None = None
    _require_target_gpu_idle(target_gpu_uuid)
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            known_worker_pids = {process.pid}
            try:
                next_telemetry_sample = time.monotonic()
                next_compute_application_sample = next_telemetry_sample
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= next_compute_application_sample:
                        _check_compute_applications(
                            target_gpu_uuid=target_gpu_uuid,
                            worker_process_group=process.pid,
                            known_worker_pids=known_worker_pids,
                        )
                        compute_application_samples += 1
                        next_compute_application_sample = (
                            now + COMPUTE_APPLICATION_INTERVAL_SECONDS
                        )
                    if now >= next_telemetry_sample:
                        telemetry.append(_query_telemetry(target_gpu_uuid))
                        next_telemetry_sample = now + TELEMETRY_INTERVAL_SECONDS
                    next_sample = min(
                        next_compute_application_sample, next_telemetry_sample
                    )
                    time.sleep(min(0.25, max(0.0, next_sample - time.monotonic())))
                _check_compute_applications(
                    target_gpu_uuid=target_gpu_uuid,
                    worker_process_group=process.pid,
                    known_worker_pids=known_worker_pids,
                )
                compute_application_samples += 1
                telemetry.append(_query_telemetry(target_gpu_uuid))
                returncode = process.wait()
                return (
                    returncode,
                    telemetry,
                    compute_application_samples,
                )
            finally:
                _terminate_process(process)
    except BaseException as error:
        worker_error = error
        raise
    finally:
        try:
            _wait_for_target_gpu_idle(target_gpu_uuid)
        except RuntimeError:
            if worker_error is None:
                raise


def _geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geomean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _valid_accuracy_check(value: object, group_count: int) -> bool:
    if not isinstance(value, dict) or value.get("ok") is not True:
        return False
    if (
        type(value.get("group_count")) is not int
        or value["group_count"] != group_count
        or value.get("rtol") != common.CORRECTNESS_RTOL
        or value.get("atol") != common.CORRECTNESS_ATOL
        or type(value.get("mismatch_count")) is not int
        or value["mismatch_count"] != 0
    ):
        return False
    return (
        all(
            isinstance(metric, int | float)
            and not isinstance(metric, bool)
            and math.isfinite(metric)
            and metric >= 0
            for metric in (value.get("max_abs"), value.get("max_normalized_diff"))
        )
        and value["max_normalized_diff"] <= common.CORRECTNESS_MAX_NORMALIZED_DIFF
    )


def _valid_capture_evidence(value: object, group_count: int) -> bool:
    return (
        isinstance(value, dict)
        and _valid_accuracy_check(value, group_count)
        and value.get("poisoned_replay_rewrote_output") is True
        and value.get("repeat_replay_exact") is True
        and _valid_accuracy_check(value.get("post_timing"), group_count)
    )


def _valid_cutlass_tuning_evidence(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    tuning = value.get("registry_tuning")
    if not isinstance(tuning, dict) or tuning.keys() != {
        "method",
        "timer",
        "capture_warmups",
        "repetitions",
        "selected_candidate_index",
        "candidates",
    }:
        return False
    candidates = tuning.get("candidates")
    selected_index = tuning.get("selected_candidate_index")
    repetitions = tuning.get("repetitions")
    if (
        tuning.get("method") != "all_supported_operators_cold_l2"
        or tuning.get("timer") != "bench_pre_captured_cudagraphs"
        or tuning.get("capture_warmups") != cutlass.CUTLASS_SELECTION_CAPTURE_WARMUPS
        or not isinstance(candidates, list)
        or not candidates
        or type(selected_index) is not int
        or not 0 <= selected_index < len(candidates)
        or type(repetitions) is not int
        or repetitions < cutlass.CUTLASS_SELECTION_MIN_REPETITIONS
        or repetitions % (2 * len(candidates))
    ):
        return False
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or candidate.keys()
            != {
                "registry_index",
                "operator_name",
                "config",
                "compiled_for",
                "selection_median_ms",
                "correctness_checked",
            }
            or type(candidate.get("registry_index")) is not int
            or candidate["registry_index"] < 0
            or not isinstance(candidate.get("operator_name"), str)
            or not candidate["operator_name"]
            or not isinstance(candidate.get("config"), dict)
            or candidate["config"].keys()
            != {"use_2cta_mma", "tile_shape", "cluster_shape"}
            or not isinstance(candidate.get("compiled_for"), str)
            or not candidate["compiled_for"]
            or not isinstance(candidate.get("selection_median_ms"), int | float)
            or isinstance(candidate["selection_median_ms"], bool)
            or not math.isfinite(candidate["selection_median_ms"])
            or candidate["selection_median_ms"] <= 0
            or candidate.get("correctness_checked") is not True
        ):
            return False
    operator_names = [
        cast("str", candidate["operator_name"]) for candidate in candidates
    ]
    if operator_names != sorted(operator_names) or len(operator_names) != len(
        set(operator_names)
    ):
        return False
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            cast("float", candidates[index]["selection_median_ms"]),
            operator_names[index],
        ),
    )
    return selected_index == best_index


def _validate_worker_result(
    result: object,
    provider: str,
    replicate: int,
    *,
    expected_source: str | None = None,
    expected_device: dict[str, Any] | None = None,
    expected_versions: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(result, dict):
        raise RuntimeError(f"{provider} result identity is inconsistent")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("provider") != provider
        or result.get("replicate") != replicate
    ):
        raise RuntimeError(f"{provider} result identity is inconsistent")
    source = result.get("source")
    if not isinstance(source, str) or not source:
        raise RuntimeError("campaign worker is missing Helion source identity")
    if expected_source is not None and source != expected_source:
        raise RuntimeError("Helion source identity changed across campaign workers")
    device = result.get("device")
    if (
        not isinstance(device, dict)
        or not isinstance(device.get("uuid"), str)
        or not device["uuid"]
        or not isinstance(device.get("name"), str)
        or not device["name"]
        or not isinstance(device.get("capability"), list)
        or len(device["capability"]) != 2
        or any(type(value) is not int for value in device["capability"])
        or not common.is_supported_grouped_gemm_device(
            "cuda", device["name"], tuple(device["capability"])
        )
    ):
        raise RuntimeError("campaign worker is missing GPU identity")
    if expected_device is not None and device != expected_device:
        raise RuntimeError("GPU identity changed across campaign workers")
    versions = result.get("versions")
    version_names = ("python", "torch", "torch_cuda", "cutlass_dsl", "triton")
    if (
        not isinstance(versions, dict)
        or any(
            not isinstance(versions.get(name), str) or not versions[name]
            for name in version_names
        )
        or not isinstance(versions.get("cuda_driver"), str)
        or not versions["cuda_driver"]
        or not isinstance(versions.get("cuda_stack"), dict)
        or versions["cuda_stack"].get("release") != common.CUDA_TOOLKIT_RELEASE
        or versions["cuda_stack"].get("compiler_version")
        != common.CUDA_COMPILER_VERSION
        or versions["cuda_stack"].get("distribution_versions")
        != common.CUDA_STACK_DISTRIBUTION_VERSIONS
    ):
        raise RuntimeError("campaign worker is missing software versions")
    if expected_versions is not None and versions != expected_versions:
        raise RuntimeError("software versions changed across campaign workers")
    expected_cases = tuple(case.as_dict() for case in common.official_cases())
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_cases):
        raise RuntimeError(
            f"{provider} result does not contain {len(expected_cases)} rows"
        )
    for row_index, (row, expected_case) in enumerate(
        zip(rows, expected_cases, strict=True)
    ):
        if not isinstance(row, dict) or row.get("case") != expected_case:
            raise RuntimeError(f"{provider} row {row_index} changed the workload")
        configs = row.get("configs")
        correctness = row.get("correctness")
        timings = row.get("timings")
        if (
            not isinstance(configs, dict)
            or not isinstance(configs.get("helion"), dict)
            or not isinstance(configs["helion"].get("config"), dict)
            or not isinstance(configs.get("provider"), dict)
        ):
            raise RuntimeError(f"{provider} row {row_index} is missing configurations")
        helion_config = configs["helion"]
        if (
            helion_config.get("selection_mode") != HELION_SELECTION
            or helion_config.get("autotuned") is not False
            or helion_config.get("b_layout") != BENCHMARK_B_LAYOUT
            or helion_config.get("selection_api")
            != "BoundKernel.set_config(ConfigSpec.default_config())"
            or helion_config["config"].get("tcgen05_grouped_mode") != "worklist_nm"
            or helion_config["config"].get("tcgen05_grouped_worklist_source_m_tile")
            != SOURCE_M_TILE
        ):
            raise RuntimeError(f"{provider} row {row_index} changed Helion selection")
        if not _valid_provider_contract(
            configs["provider"],
            provider,
            expected_case=expected_case,
            device=device,
            complete=True,
        ):
            raise RuntimeError(f"{provider} row {row_index} changed provider selection")
        if not isinstance(correctness, dict) or any(
            not _valid_capture_evidence(
                correctness.get(implementation), cast("int", expected_case["groups"])
            )
            for implementation in ("helion", "provider")
        ):
            raise RuntimeError(
                f"{provider} row {row_index} is missing correctness evidence"
            )
        if not isinstance(timings, dict) or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in (timings.get("helion_ms"), timings.get("provider_ms"))
        ):
            raise RuntimeError(f"{provider} result contains invalid timings")
    return source, device, versions


def summarize_results(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected_runs = publication_runs()
    expected_cases = tuple(
        cast("dict[str, Any]", case.as_dict()) for case in common.official_cases()
    )
    row_count = len(expected_cases)
    if row_count != 8:
        raise RuntimeError("benchmark requires exactly eight official cases")
    if len(results) != len(expected_runs):
        raise RuntimeError("campaign produced an unexpected worker result count")
    results_by_provider: dict[str, list[dict[str, Any]]] = {
        provider: [] for provider in PROVIDERS
    }
    identity: tuple[str, dict[str, Any], dict[str, Any]] | None = None
    for result, (provider, replicate) in zip(results, expected_runs, strict=True):
        expected_source, expected_device, expected_versions = (
            identity if identity is not None else (None, None, None)
        )
        worker_identity = _validate_worker_result(
            result,
            provider,
            replicate,
            expected_source=expected_source,
            expected_device=expected_device,
            expected_versions=expected_versions,
        )
        if identity is None:
            identity = worker_identity
        results_by_provider[provider].append(result)
    if identity is None:
        raise RuntimeError("campaign produced no worker results")
    source, device, software_stack = identity
    provider_summaries = {}
    helion_configs: list[list[object]] = [[] for _ in range(row_count)]
    for provider in PROVIDERS:
        provider_results = results_by_provider[provider]
        row_speedups: list[list[float]] = [[] for _ in range(row_count)]
        row_helion_ms: list[list[float]] = [[] for _ in range(row_count)]
        row_provider_ms: list[list[float]] = [[] for _ in range(row_count)]
        provider_configs: list[list[object]] = [[] for _ in range(row_count)]
        for result in provider_results:
            rows = cast("list[dict[str, Any]]", result["rows"])
            timing_rows = [cast("dict[str, float]", row["timings"]) for row in rows]
            helion_times = [float(timings["helion_ms"]) for timings in timing_rows]
            provider_times = [float(timings["provider_ms"]) for timings in timing_rows]
            speedups = [
                provider_ms / helion_ms
                for helion_ms, provider_ms in zip(
                    helion_times, provider_times, strict=True
                )
            ]
            for row_index, (row, helion_ms, provider_ms, speedup) in enumerate(
                zip(rows, helion_times, provider_times, speedups, strict=True)
            ):
                row_speedups[row_index].append(speedup)
                row_helion_ms[row_index].append(helion_ms)
                row_provider_ms[row_index].append(provider_ms)
                helion_configs[row_index].append(row["configs"]["helion"]["config"])
                provider_config = row["configs"]["provider"]
                provider_configs[row_index].append(provider_config)
        row_summaries = [
            {
                "row_index": row_index,
                "geomean_speedup": _geomean(speedups),
                "helion_ms_by_replicate": row_helion_ms[row_index],
                "provider_ms_by_replicate": row_provider_ms[row_index],
            }
            for row_index, speedups in enumerate(row_speedups)
        ]
        all_speedups = [speedup for row in row_speedups for speedup in row]
        worst_row = min(row_summaries, key=itemgetter("geomean_speedup"))
        comparable_configs = provider_configs
        if provider == "quack":
            if not all(
                _all_equal(
                    [
                        _quack_provenance(cast("dict[str, Any]", config))
                        for config in configs
                    ]
                )
                for configs in provider_configs
            ):
                raise RuntimeError(f"{provider} provenance changed across replicates")
            selection_stability = "fresh_autotuned"
            varying_config_rows = [
                row_index
                for row_index, configs in enumerate(provider_configs)
                if not _all_equal(configs)
            ]
        elif provider == "cutlass":
            comparable_configs = [
                [
                    _cutlass_provenance(cast("dict[str, Any]", config))
                    for config in configs
                ]
                for configs in provider_configs
            ]
            varying_config_rows = [
                row_index
                for row_index, configs in enumerate(comparable_configs)
                if not _all_equal(configs)
            ]
            if varying_config_rows:
                raise RuntimeError(
                    "cutlass selected config changed across fresh tuning replicates"
                )
            selection_stability = "fresh_tuned_reproducible"
        else:
            varying_config_rows = [
                row_index
                for row_index, configs in enumerate(comparable_configs)
                if not _all_equal(configs)
            ]
            if varying_config_rows:
                raise RuntimeError(
                    f"{provider} selected config changed across replicates"
                )
            selection_stability = "fixed"
        provider_summary = {
            "selection": PROVIDER_SELECTION_MODES[provider],
            "selection_stability": selection_stability,
            "cross_replicate_geomean": _geomean(all_speedups),
            "row_wins": sum(row["geomean_speedup"] > 1.0 for row in row_summaries),
            "worst_row": worst_row,
            "rows": row_summaries,
            "varying_config_rows": varying_config_rows,
        }
        benchmark_labels = [
            cast("dict[str, Any]", config).get("benchmark_label")
            for configs in provider_configs
            for config in configs
        ]
        if not _all_equal(benchmark_labels):
            raise RuntimeError(
                f"{provider} benchmark label changed across rows or replicates"
            )
        benchmark_label = benchmark_labels[0]
        if benchmark_label is not None and (
            not isinstance(benchmark_label, str) or not benchmark_label
        ):
            raise RuntimeError(f"{provider} benchmark label is invalid")
        if benchmark_label:
            provider_summary["benchmark_label"] = benchmark_label
        provider_summaries[provider] = provider_summary
    if not all(_all_equal(configs) for configs in helion_configs):
        raise RuntimeError("fixed Helion config changed across fresh workers")
    return {
        "schema": SUMMARY_SCHEMA,
        "providers": list(PROVIDERS),
        "replicates": PUBLICATION_REPLICATES,
        "helion_selection": HELION_SELECTION,
        "cases": list(expected_cases),
        "device": device,
        "software_stack": software_stack,
        "speedup_definition": "provider_ms / helion_ms; higher favors Helion",
        "protocol": {
            **_COMMON_PROTOCOL,
            "rows_per_replicate": row_count,
            "row_timing_statistic": "median_ms",
            "raw_paired_samples_retained": False,
            "post_timing_correctness_check": True,
            "protocol_evidence": "clean_worker_source_and_fixed_constants",
            "oracle_float32_matmul_precision": common.ORACLE_FLOAT32_MATMUL_PRECISION,
            "layout_policy": BENCHMARK_LAYOUT_POLICY,
        },
        "provider_results": provider_summaries,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    row_count = summary["protocol"]["rows_per_replicate"]
    print("\nprovider   geomean  wins  worst")
    for provider in summary["providers"]:
        result = summary["provider_results"][provider]
        worst = result["worst_row"]
        print(
            f"{provider:<10} {result['cross_replicate_geomean']:>7.3f}x "
            f"{result['row_wins']:>2}/{row_count}  row {worst['row_index']}: "
            f"{worst['geomean_speedup']:.3f}x"
        )
        if result["varying_config_rows"]:
            print(
                f"  {result['selection_stability']} config variation on rows "
                f"{result['varying_config_rows']}"
            )
        if benchmark_label := result.get("benchmark_label"):
            print(f"  benchmark: {benchmark_label}")
    monitoring = summary["monitoring"]
    print(
        "\nGPU telemetry (whole worker, including setup): "
        f"power {monitoring['power_draw_watts']['mean']:.1f} W, "
        f"SM clock {monitoring['sm_clock_mhz']['mean']:.0f} MHz, "
        f"memory clock {monitoring['memory_clock_mhz']['mean']:.0f} MHz"
    )
    if reasons := monitoring["active_clock_event_reasons"]:
        print(f"  allowed active GPU clock-event reasons: {reasons}")


def _run_campaign(args: argparse.Namespace) -> int:
    roots = _provider_roots(args)
    target_gpu_uuid = _resolve_target_gpu(args.cuda_visible_devices)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(REPO_ROOT):
        raise ValueError("output directory must be outside the Helion checkout")
    source = _source_identity()
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    compute_application_sample_count = 0
    campaign_telemetry: list[_TelemetrySample] = []
    per_run_telemetry: dict[str, dict[str, object]] = {}
    campaign_device: dict[str, Any] | None = None
    campaign_versions: dict[str, Any] | None = None
    for provider, replicate in publication_runs():
        run_id = f"{provider}-r{replicate}"
        run_dir = output_dir / run_id
        run_dir.mkdir()
        command = _worker_command(provider, replicate, run_dir, roots)
        log_path = run_dir / "worker.log"
        print(f"=== {run_id} ===", flush=True)
        returncode, run_telemetry, compute_application_samples = _run_monitored_worker(
            command,
            environment=_worker_environment(
                run_dir,
                cuda_visible_devices=target_gpu_uuid,
                quack_root=(roots.get("quack") if provider == "quack" else None),
            ),
            log_path=log_path,
            target_gpu_uuid=target_gpu_uuid,
        )
        if returncode:
            tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-30:])
            raise RuntimeError(f"{run_id} failed with {returncode}:\n{tail}")
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"{run_id} did not produce result.json")
        result = json.loads(result_path.read_text())
        _, worker_device, worker_versions = _validate_worker_result(
            result,
            provider,
            replicate,
            expected_source=source,
            expected_device=campaign_device,
            expected_versions=campaign_versions,
        )
        if campaign_device is None:
            campaign_device = worker_device
            campaign_versions = worker_versions
        if (
            worker_device.get("uuid") != target_gpu_uuid
            or worker_device.get("visible") != target_gpu_uuid
        ):
            raise RuntimeError(f"{run_id} worker used the wrong GPU: {worker_device}")
        # Validate each worker immediately so a bad run aborts before the
        # remaining fresh-process campaign is launched.
        run_monitoring = _summarize_telemetry(run_telemetry)
        run_monitoring["compute_application_sample_count"] = compute_application_samples
        per_run_telemetry[run_id] = run_monitoring
        campaign_telemetry.extend(run_telemetry)
        compute_application_sample_count += compute_application_samples
        results.append(result)
        if _source_identity() != source:
            raise RuntimeError("Helion checkout changed during the campaign")
    summary = summarize_results(results)
    summary["source"] = source
    monitoring = _summarize_telemetry(campaign_telemetry)
    monitoring.update(
        {
            "target_gpu_idle_before_after_each_run": True,
            "compute_application_sample_interval_seconds": (
                COMPUTE_APPLICATION_INTERVAL_SECONDS
            ),
            "telemetry_sample_interval_seconds": TELEMETRY_INTERVAL_SECONDS,
            "compute_application_sample_count": compute_application_sample_count,
            "allowed_clock_event_reason_mask": f"0x{ALLOWED_CLOCK_EVENT_REASONS:x}",
            "by_provider_replicate": per_run_telemetry,
        }
    )
    summary["monitoring"] = monitoring
    common.write_result(output_dir / "summary.json", summary)
    if _source_identity() != source:
        raise RuntimeError("Helion checkout changed while writing the summary")
    _print_summary(summary)
    print(f"\nWrote {output_dir / 'summary.json'}")
    return 0


def campaign_plan() -> dict[str, object]:
    """Return the CPU-only, immutable publication protocol plan."""

    from benchmarks.cute.cublaslt_grouped_gemm import CUBLASLT_DISTRIBUTION_VERSION
    from benchmarks.cute.cublaslt_grouped_gemm import CUBLASLT_LIBRARY_VERSION
    from benchmarks.cute.cudnn_grouped_gemm import CUDNN_BACKEND_DISTRIBUTION_VERSION
    from benchmarks.cute.cudnn_grouped_gemm import CUDNN_FRONTEND_VERSION
    from benchmarks.cute.cutlass_contiguous_grouped_gemm import CUTLASS_COMMIT
    from benchmarks.cute.cutlass_contiguous_grouped_gemm import CUTLASS_RELEASE_TAG
    from benchmarks.cute.grouped_gemm_deepgemm_support import DEEPGEMM_COMMIT
    from benchmarks.cute.grouped_gemm_deepgemm_support import DEEPGEMM_VERSION
    from benchmarks.cute.quack_grouped_gemm import QUACK_BASE_RELEASE_TAG
    from benchmarks.cute.quack_grouped_gemm import QUACK_COMMIT

    cases = common.official_cases()
    return {
        "schema": "helion-grouped-gemm-provider-plan-v3",
        "providers": list(PROVIDERS),
        "replicates": PUBLICATION_REPLICATES,
        "worker_count": len(PROVIDERS) * PUBLICATION_REPLICATES,
        "run_order": [
            {"provider": provider, "replicate": replicate}
            for provider, replicate in publication_runs()
        ],
        "cases": [case.as_dict() for case in cases],
        "helion": {
            "selection": HELION_SELECTION,
            "kernel_api": "helion.kernel",
            "required_heuristic": GROUPED_WORKLIST_HEURISTIC,
            "source_m_tile": SOURCE_M_TILE,
            "b_layout": BENCHMARK_B_LAYOUT,
        },
        "provider_pins": {
            "deepgemm": {"version": DEEPGEMM_VERSION, "commit": DEEPGEMM_COMMIT},
            "quack": {
                "base_release": QUACK_BASE_RELEASE_TAG,
                "commit": QUACK_COMMIT,
            },
            "cudnn": {
                "frontend": CUDNN_FRONTEND_VERSION,
                "backend": CUDNN_BACKEND_DISTRIBUTION_VERSION,
            },
            "cublaslt": {
                "distribution": CUBLASLT_DISTRIBUTION_VERSION,
                "library": CUBLASLT_LIBRARY_VERSION,
            },
            "cutlass": {"release": CUTLASS_RELEASE_TAG, "commit": CUTLASS_COMMIT},
        },
        "provider_selections": dict(PROVIDER_SELECTION_MODES),
        "provenance": {
            "clean_helion_commit": True,
            "cuda_toolkit_release": common.CUDA_TOOLKIT_RELEASE,
            "cuda_compiler_version": common.CUDA_COMPILER_VERSION,
            "cuda_distributions": common.CUDA_STACK_DISTRIBUTION_VERSIONS,
            "cuda_loaded_libraries_validated": True,
            "provider_versions_commits_and_configs_recorded": True,
        },
        "protocol": {
            **_COMMON_PROTOCOL,
            "fail_closed_source_gpu_stack_and_telemetry": True,
        },
    }


def build_arg_parser(mode: str) -> argparse.ArgumentParser:
    """Build the strict parser for one provider campaign mode."""

    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} {mode}",
        description=__doc__,
        allow_abbrev=False,
    )
    if mode == PROVIDER_DEFAULTS_PLAN_MODE:
        return parser
    roots_required = mode == PROVIDER_DEFAULTS_MODE
    for provider in SOURCE_BACKED_PROVIDERS:
        parser.add_argument(
            f"--provider-{provider}-root",
            dest=f"{provider}_root",
            type=Path,
            required=roots_required,
        )
    if mode == PROVIDER_DEFAULTS_MODE:
        parser.add_argument(
            "--provider-output-dir", dest="output_dir", type=Path, required=True
        )
        parser.add_argument(
            "--cuda-visible-devices",
            default=os.environ.get("CUDA_VISIBLE_DEVICES"),
            help="one CUDA_VISIBLE_DEVICES entry (GPU UUID or index)",
        )
        return parser
    parser.add_argument(
        "--provider", choices=PROVIDERS, required=True, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--provider-replicate",
        dest="replicate",
        type=_nonnegative_int,
        required=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--provider-run-dir",
        dest="run_dir",
        type=Path,
        required=True,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str]) -> int:
    """Parse and execute a provider mode selected by the public entrypoint."""

    arguments = list(argv)
    if not arguments or arguments[0] not in PROVIDER_CLI_MODES:
        raise ValueError("provider campaign mode is required")
    mode = arguments.pop(0)
    args = build_arg_parser(mode).parse_args(arguments)
    if mode == PROVIDER_DEFAULTS_PLAN_MODE:
        print(json.dumps(campaign_plan(), indent=2, sort_keys=True))
        return 0
    if mode == PROVIDER_DEFAULTS_WORKER_MODE:
        return _run_worker(args)
    if args.cuda_visible_devices is None:
        raise ValueError("set CUDA_VISIBLE_DEVICES or --cuda-visible-devices")
    common.require_single_visible_device(args.cuda_visible_devices)

    def handle_signal(signum: int, _frame: object) -> None:
        raise _CampaignInterrupted(signum)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return _run_campaign(args)
    except _CampaignInterrupted as error:
        return 128 + error.signum
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
