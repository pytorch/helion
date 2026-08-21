"""Compare Helion grouped GEMM with selectable provider public defaults.

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
import importlib.metadata
from itertools import accumulate
import json
import math
from operator import itemgetter
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmCase
    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation
    from pretuned_kernels.grouped_gemm_deepgemm.reviewed_profiles import OfficialShape
    from pretuned_kernels.grouped_gemm_deepgemm.reviewed_profiles import (
        ReviewedWorklistProfile,
    )
    import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    configured_pythonpath = os.environ.get("PYTHONPATH")
    if configured_pythonpath is not None:
        startup_directory = Path.cwd()
        configured_paths = {
            (startup_directory / configured_path).resolve()
            for configured_path in configured_pythonpath.split(os.pathsep)
        }
        sys.path[:] = [
            entry
            for entry in sys.path
            if (startup_directory / entry).resolve() not in configured_paths
        ]
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cute import grouped_gemm_benchmark as common  # noqa: E402

RESULT_SCHEMA = "helion-grouped-gemm-provider-defaults-v2"
SUMMARY_SCHEMA = "helion-grouped-gemm-provider-summary-v2"
BENCHMARK_REPETITIONS = 102
THERMAL_WARMUP_MS = 10_000
CAPTURE_WARMUPS = 2
TELEMETRY_INTERVAL_SECONDS = 5
POST_WORKER_IDLE_GRACE_SECONDS = 5
PROVIDERS = ("deepgemm", "quack", "cudnn", "cublaslt", "cutlass")
PROVIDER_SELECTIONS = {
    "deepgemm": "public_kmajor_nk_no_psum_zero_padding_off",
    "quack": "public_tuned_false_config_none",
    "cudnn": "public_a_fallback_build_execute",
    "cublaslt": "heuristic_result_zero_requested_one",
    "cutlass": "registry_first_no_timing_search",
}
HELION_SELECTIONS = (
    "compiler_heuristic",
    "live_autotune",
    "final_reviewed_aot",
)
GROUPED_WORKLIST_HEURISTIC = "cute_tcgen05_grouped_worklist"
HELION_SELECTION_ENVIRONMENT_VARIABLES = (
    "HELION_AOT_MODE",
    "HELION_AUTOTUNER",
    "HELION_AUTOTUNE_EFFORT",
    "HELION_AUTOTUNE_RANDOM_SEED",
    "HELION_BACKEND",
    "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS",
    "HELION_CUTE_MMA_IMPL",
    "HELION_SKIP_CACHE",
)
HELION_SELECTION_STARTUP_ENVIRONMENT = {
    name: os.environ.get(name) for name in HELION_SELECTION_ENVIRONMENT_VARIABLES
}
LIVE_AUTOTUNE_CACHE = "LocalAutotuneCache"
LIVE_AUTOTUNE_SEARCH_POLICY = {
    "algorithm": "LFBOTreeSearch",
    "autotune_effort": "full",
    "force": True,
    "random_seed": 0,
    "autotune_cache": LIVE_AUTOTUNE_CACHE,
    "skip_cache_read_write": True,
    "fresh_worker_cache": True,
    "budget_seconds": None,
    "max_generations_override": None,
    "effective_max_generations": 20,
    "best_of_k": 1,
    "accuracy_check": True,
    "benchmark_subprocess": True,
    "benchmark_timeout_seconds": 30,
    "configured_compile_timeout_seconds": 60,
    "effective_compile_timeout_seconds": None,
    "precompile_mode": None,
    "adaptive_timeout": True,
    "initial_population_strategy": "from_random",
    "initial_population": 100,
    "copies": 5,
    "best_available_pad_random": True,
    "finishing_rounds": 0,
    "num_neighbors_cap": -1,
    "config_overrides": {},
    "advanced_controls_files": [],
    "compiler_seed_candidates_enabled": True,
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
    "CUDA_",
    "CUTE_DSL_",
    "DG_",
    "HELION_",
    "LD_",
    "QUACK_",
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
        "NVIDIA_TF32_OVERRIDE",
        "NVCC",
        "NVCC_APPEND_FLAGS",
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
        "OBJC_INCLUDE_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_SYSROOT_DIR",
        "SDKROOT",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "TORCH_CUDA_ARCH_LIST",
        "CUBLAS_WORKSPACE_CONFIG",
    }
)
TELEMETRY_FIELDS = (
    "timestamp",
    "uuid",
    "pstate",
    "clocks.sm",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "clocks_event_reasons.active",
)
WORKER_TERMINATION_GRACE_SECONDS = 5
NVIDIA_SMI_TIMEOUT_SECONDS = 10


class _CampaignInterrupted(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _require_live_autotuner_algorithm() -> str:
    expected = cast("str", LIVE_AUTOTUNE_SEARCH_POLICY["algorithm"])
    observed = os.environ.get("HELION_AUTOTUNER")
    if observed != expected:
        raise RuntimeError(
            f"live autotune requires HELION_AUTOTUNER={expected!r}, got {observed!r}"
        )
    return expected


def _layout_name(b_major: str) -> str:
    if b_major == "k":
        return "k_major"
    if b_major == "n":
        return "n_major"
    raise ValueError(f"unsupported reviewed B major {b_major!r}")


def make_exact_common_inputs(
    case: GroupedGemmCase,
    shape: OfficialShape,
    device: torch.device,
) -> tuple[GroupedGemmInputs, ReviewedWorklistProfile]:
    """Generate the final benchmark's exact logical values for one row."""

    from pretuned_kernels.grouped_gemm_deepgemm import (
        _deepgemm_public_api as public_api,
    )
    from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles
    import torch

    expected_shape = (
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    case_shape = (
        case.groups,
        case.expected_m_per_group,
        case.n,
        case.k,
    )
    if case_shape != expected_shape:
        raise RuntimeError(f"workload shape mismatch: {case_shape} != {expected_shape}")
    profile = reviewed_profiles.exact_reviewed_worklist_profile(*expected_shape)
    packed_a, logical_b, _helion_b, _layout, reference, _worklist = (
        public_api._make_reviewed_case(
            shape,
            case.actual_ms,
            device,
        )
    )
    stored_ms = tuple(
        common.align(actual_m, profile.source_m_tile) for actual_m in case.actual_ms
    )
    starts = tuple(accumulate((0, *stored_ms)))[:-1]
    compact_a = torch.cat(
        tuple(
            packed_a[start : start + actual_m]
            for start, actual_m in zip(starts, case.actual_ms, strict=True)
        )
    )
    oracle = tuple(
        reference[start : start + actual_m]
        for start, actual_m in zip(starts, case.actual_ms, strict=True)
    )
    b_n_major = logical_b.transpose(1, 2).contiguous().transpose(1, 2)
    if not torch.equal(logical_b, b_n_major):
        raise RuntimeError("grouped B layout conversion changed logical values")
    offsets = torch.tensor(
        (0, *accumulate(case.actual_ms)),
        device=device,
        dtype=torch.int32,
    )
    inputs = common.GroupedGemmInputs(
        case=case,
        compact_a=compact_a,
        b=logical_b,
        b_n_major=b_n_major,
        offsets=offsets,
        oracle=oracle,
    )
    return inputs, profile


def prepare_helion(
    inputs: GroupedGemmInputs,
    profile: ReviewedWorklistProfile,
    selection_mode: str,
) -> PreparedImplementation:
    """Bind Helion with compiler, live-autotune, or reviewed-AOT selection."""

    from pretuned_kernels.grouped_gemm_deepgemm import (
        _deepgemm_public_api as public_api,
    )
    from pretuned_kernels.grouped_gemm_deepgemm import grouped_gemm_deepgemm
    from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles
    import torch

    packed = common.pack_compact_rows(inputs, profile.source_m_tile)
    b_layout = _layout_name(profile.b_major)
    b = inputs.b_for_layout(b_layout)
    kernel_args = (
        packed.a,
        b,
        packed.worklist,
        inputs.case.expected_m_per_group,
        profile.source_m_tile,
    )
    kernel = grouped_gemm_deepgemm.create_grouped_gemm_deepgemm_kernel()
    if selection_mode == "live_autotune":
        kernel.settings.autotune_cache = LIVE_AUTOTUNE_CACHE
    bound = kernel.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    if selection_mode == "final_reviewed_aot":
        initial_output = bound(*kernel_args)
        config = public_api._effective_reviewed_config(bound, profile)
        selection_evidence: dict[str, object] = {
            "selection_mode": selection_mode,
            "reviewed_profile": {
                "name": profile.config_name,
                "manifest_sha256": (reviewed_profiles.REVIEWED_PROFILE_MANIFEST_SHA256),
                "role": "selected_config_and_packing",
                "b_layout": b_layout,
                "source_m_tile": profile.source_m_tile,
            },
            "config": config,
            "config_sha256": common.config_sha256(config["effective"]),
            "a_layout": {
                "kind": "aligned_contiguous_worklist",
                "alignment": profile.source_m_tile,
                "logical_values_bitwise_equal": True,
            },
        }
    else:
        config_spec = bound.config_spec
        compiler_primary = config_spec.compiler_default_config
        if compiler_primary is None:
            raise RuntimeError(
                "grouped GEMM compiler heuristics produced no primary config"
            )
        compiler_promoted = dict(compiler_primary.config)
        fired_heuristics = list(config_spec.autotuner_heuristics)
        ordered_seeds = [
            dict(seed.config) for seed in config_spec.compiler_seed_configs
        ]
        promoted_indexes = [
            index
            for index, seed in enumerate(ordered_seeds)
            if seed == compiler_promoted
        ]
        grouped_worklist_seed_indexes = [
            index
            for index, seed in enumerate(ordered_seeds)
            if seed.get("tcgen05_grouped_mode") == "worklist_nm"
            and seed.get("tcgen05_grouped_worklist_source_m_tile")
            == profile.source_m_tile
        ]
        if (
            fired_heuristics.count(GROUPED_WORKLIST_HEURISTIC) != 1
            or not grouped_worklist_seed_indexes
            or compiler_promoted.get("tcgen05_grouped_mode") != "worklist_nm"
            or compiler_promoted.get("tcgen05_grouped_worklist_source_m_tile")
            != profile.source_m_tile
            or promoted_indexes != [grouped_worklist_seed_indexes[0]]
        ):
            raise RuntimeError(
                "compiler primary is not the named grouped-worklist rank-zero promotion"
            )
        compiler_effective_config = config_spec.default_config()
        compiler_effective = dict(compiler_effective_config.config)
        compiler_evidence = {
            "promoting_heuristic": GROUPED_WORKLIST_HEURISTIC,
            "fired_heuristics": fired_heuristics,
            "promoted": compiler_promoted,
            "promoted_sha256": common.config_sha256(compiler_promoted),
            "effective": compiler_effective,
            "effective_sha256": common.config_sha256(compiler_effective),
            "ordered_seeds": ordered_seeds,
            "ordered_seed_sha256": [
                common.config_sha256(seed) for seed in ordered_seeds
            ],
            "promoted_seed_index": promoted_indexes[0],
            "grouped_worklist_seed_indexes": grouped_worklist_seed_indexes,
            "promotion_policy": "named_heuristic_rank_zero_to_compiler_default",
            "effective_config_source": "ConfigSpec.default_config",
        }
        if selection_mode == "compiler_heuristic":
            selected_config = compiler_effective_config
            bound.set_config(selected_config)
            selection_api = "BoundKernel.set_config(ConfigSpec.default_config())"
            live_search = None
        elif selection_mode == "live_autotune":
            from helion.autotuner.effort_profile import get_effort_profile

            # CuTe compiles each candidate inline; make the backend's effective
            # no-precompile policy explicit before recording and running search.
            bound.settings.autotune_precompile = None
            autotuner_algorithm = _require_live_autotuner_algorithm()
            profile_settings = get_effort_profile(bound.settings.autotune_effort)
            lfbo = profile_settings.lfbo_pattern_search
            if lfbo is None:
                raise RuntimeError("live autotune did not resolve a full LFBO profile")
            live_search = {
                "algorithm": autotuner_algorithm,
                "autotune_effort": bound.settings.autotune_effort,
                "force": True,
                "random_seed": bound.settings.autotune_random_seed,
                "autotune_cache": bound.settings.autotune_cache,
                "skip_cache_read_write": os.environ.get("HELION_SKIP_CACHE") == "1",
                "fresh_worker_cache": True,
                "budget_seconds": bound.settings.autotune_budget_seconds,
                "max_generations_override": bound.settings.autotune_max_generations,
                "effective_max_generations": (
                    bound.settings.autotune_max_generations or lfbo.max_generations
                ),
                "best_of_k": bound.settings.autotune_best_of_k,
                "accuracy_check": bound.settings.autotune_accuracy_check,
                "benchmark_subprocess": bound.settings.autotune_benchmark_subprocess,
                "benchmark_timeout_seconds": (
                    bound.settings.autotune_benchmark_timeout
                ),
                "configured_compile_timeout_seconds": (
                    bound.settings.autotune_compile_timeout
                ),
                "effective_compile_timeout_seconds": None,
                "precompile_mode": bound.settings.autotune_precompile,
                "adaptive_timeout": bound.settings.autotune_adaptive_timeout,
                "initial_population_strategy": (
                    bound.settings.autotune_initial_population_strategy
                    or lfbo.initial_population_strategy
                ),
                "initial_population": lfbo.initial_population,
                "copies": lfbo.copies,
                "best_available_pad_random": lfbo.best_available_pad_random,
                "finishing_rounds": profile_settings.finishing_rounds,
                "num_neighbors_cap": int(
                    os.environ.get("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", "-1")
                ),
                "config_overrides": bound.settings.autotune_config_overrides,
                "advanced_controls_files": bound.settings.autotune_search_acf,
                "compiler_seed_candidates_enabled": bool(ordered_seeds)
                and not bound.settings.disable_autotuner_heuristics,
            }
            if live_search != LIVE_AUTOTUNE_SEARCH_POLICY:
                raise RuntimeError(
                    "live autotune search policy differs from the publication contract"
                )
            selected_config = bound.autotune(kernel_args, force=True)
            selection_api = "BoundKernel.autotune(force=True)"
        else:
            raise ValueError(f"unsupported Helion selection {selection_mode!r}")
        initial_output = bound(*kernel_args)
        selected_requested = dict(selected_config.config)
        selected_effective = dict(
            bound.config_spec.normalized_config(selected_config).config
        )
        if (
            selected_effective.get("tcgen05_grouped_worklist_source_m_tile")
            != profile.source_m_tile
        ):
            raise RuntimeError(
                "Helion selection changed the fixed physical-A source tile"
            )
        selection_evidence = {
            "selection_mode": selection_mode,
            "autotuned": selection_mode == "live_autotune",
            "reviewed_profile": {
                "name": profile.config_name,
                "manifest_sha256": (reviewed_profiles.REVIEWED_PROFILE_MANIFEST_SHA256),
                "role": "packing_constraints_only",
                "b_layout": b_layout,
                "source_m_tile": profile.source_m_tile,
            },
            "compiler_heuristic": compiler_evidence,
            "selection_api": selection_api,
            "config": {
                "requested": selected_requested,
                "effective": selected_effective,
            },
            "config_sha256": common.config_sha256(selected_effective),
            "live_search": live_search,
            "a_layout": {
                "kind": "aligned_contiguous_worklist",
                "alignment": profile.source_m_tile,
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
        name=f"helion-{selection_mode.replace('_', '-')}-{inputs.case.id}",
        call=call,
        output_tensors=output_tensors,
        logical_outputs=logical_outputs,
        config=selection_evidence,
        owners=(
            inputs,
            packed,
            b,
            kernel,
            bound,
            initial_output,
            *kernel_args,
        ),
        track_cute_graph=True,
    )


def prepare_provider_default(
    provider: str,
    inputs: GroupedGemmInputs,
    profile: ReviewedWorklistProfile,
    *,
    cutlass_root: Path | None,
    deepgemm_root: Path | None,
) -> PreparedImplementation:
    """Prepare exactly one provider-selected default implementation."""

    reviewed_layout = _layout_name(profile.b_major)
    if provider == "deepgemm":
        from benchmarks.cute import grouped_gemm_deepgemm_support as support
        from pretuned_kernels.grouped_gemm_deepgemm import (
            _deepgemm_public_api as public_api,
        )
        import torch

        if deepgemm_root is None:
            raise ValueError("--deepgemm-root is required for the DeepGEMM provider")
        deep_gemm, provenance = support.import_deepgemm(
            deepgemm_root,
            support.M_ALIGNMENT,
        )
        packed = common.pack_compact_rows(inputs, support.M_ALIGNMENT)
        layout = torch.full(
            (packed.a.size(0),),
            -1,
            device=packed.a.device,
            dtype=torch.int32,
        )
        for group, (start, actual_m) in enumerate(
            zip(packed.starts, packed.actual_ms, strict=True)
        ):
            layout[start : start + actual_m] = group
        output = torch.empty(
            (packed.a.size(0), inputs.case.n),
            device=packed.a.device,
            dtype=packed.a.dtype,
        )

        def call() -> torch.Tensor:
            return public_api._launch_deepgemm(
                deep_gemm,
                packed.a,
                inputs.b,
                output,
                layout,
            )

        prepared = common.PreparedImplementation(
            name="deepgemm-public-default",
            call=call,
            output_tensors=lambda result: (cast("torch.Tensor", result),),
            logical_outputs=lambda result: packed.output_slices(
                cast("torch.Tensor", result)
            ),
            config={
                "provider": "deepgemm",
                "selection_mode": "public_api_fixed_no_tune",
                "b_layout": "k_major",
                "api": {
                    "function": "m_grouped_bf16_gemm_nt_contiguous",
                    "compiled_dims": "nk",
                    "use_psum_layout": False,
                    "ensure_zero_padding": False,
                },
                "a_layout": {
                    "kind": "aligned_contiguous_layout",
                    "alignment": support.M_ALIGNMENT,
                    "logical_values_bitwise_equal": True,
                },
                "preprocessing_timed": False,
                "provenance": provenance,
            },
            owners=(deep_gemm, inputs, packed, layout, output),
        )
    elif provider == "cutlass":
        from benchmarks.cute.cutlass_contiguous_grouped_gemm import (
            prepare_cutlass_default,
        )

        if cutlass_root is None:
            raise ValueError("--cutlass-root is required for the CUTLASS provider")
        prepared = prepare_cutlass_default(inputs, cutlass_root=cutlass_root)
    elif provider == "quack":
        from benchmarks.cute.quack_grouped_gemm import prepare_quack_default

        prepared = prepare_quack_default(inputs, b_layout=reviewed_layout)
    elif provider == "cudnn":
        from benchmarks.cute.cudnn_grouped_gemm import prepare_cudnn_default

        prepared = prepare_cudnn_default(inputs, b_layout=reviewed_layout)
    elif provider == "cublaslt":
        from benchmarks.cute.cublaslt_grouped_gemm import prepare_cublaslt_default

        prepared = prepare_cublaslt_default(inputs, b_layout=reviewed_layout)
    else:
        raise ValueError(f"unsupported provider {provider!r}")
    provider_layout = cast("str", prepared.config["b_layout"])
    prepared.config.update(
        {
            "reviewed_helion_b_layout": reviewed_layout,
            "physical_b_layout_matches_reviewed": provider_layout == reviewed_layout,
            "logical_b_values_bitwise_equal": True,
        }
    )
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
    shape: OfficialShape,
    device: torch.device,
    *,
    helion_selection: str,
    cutlass_root: Path | None,
    deepgemm_root: Path | None,
) -> dict[str, object]:
    """Validate and time one selected Helion/provider-default pair."""

    from pretuned_kernels import _bench
    import torch

    inputs, profile = make_exact_common_inputs(case, shape, device)
    # Provider imports, compilation, graph capture, and paired timing must not
    # perturb later rows. Thermal warmup intentionally advances the continuous
    # seed-0 public input stream once between the two isolated scopes.
    with torch.random.fork_rng(devices=[device.index]):
        helion = prepare_helion(inputs, profile, helion_selection)
        provider_impl = prepare_provider_default(
            provider,
            inputs,
            profile,
            cutlass_root=cutlass_root,
            deepgemm_root=deepgemm_root,
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
    if not all(
        math.isfinite(value) and value > 0.0 for value in (helion_ms, provider_ms)
    ):
        raise RuntimeError("benchmark timings must be finite and positive")
    return {
        "case": case.as_dict(),
        "row_index": shape.row_index,
        "configs": {
            "helion": helion.config,
            "provider": provider_impl.config,
        },
        "correctness": {
            "passed": True,
            "helion": helion_correctness,
            "provider": provider_correctness,
        },
        "timings": {
            "helion_ms": helion_ms,
            "provider_ms": provider_ms,
            "helion_us": helion_ms * 1000.0,
            "provider_us": provider_ms * 1000.0,
            "helion_speedup": provider_ms / helion_ms,
        },
    }


def parse_providers(text: str) -> tuple[str, ...]:
    providers = tuple(item.strip().lower() for item in text.split(",") if item.strip())
    if not providers:
        raise argparse.ArgumentTypeError("select at least one provider")
    unknown = tuple(provider for provider in providers if provider not in PROVIDERS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown providers: {', '.join(unknown)}")
    if len(set(providers)) != len(providers):
        raise argparse.ArgumentTypeError("provider list contains duplicates")
    return providers


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _existing_directory(text: str) -> Path:
    path = Path(text).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


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


def _source_identity() -> dict[str, str]:
    status = _git_value("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("benchmark requires a clean Helion checkout")
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "tree": _git_value("rev-parse", "HEAD^{tree}"),
    }


def _device_info(device: torch.device) -> dict[str, object]:
    import torch

    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if not common.is_canonical_b200(device.type, properties.name, capability):
        raise RuntimeError(
            "grouped-GEMM defaults benchmark requires NVIDIA B200/SM100, got "
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


def _selection_environment(selection: str) -> dict[str, str]:
    environment = {
        "HELION_AOT_MODE": (
            "evaluate" if selection == "final_reviewed_aot" else "disabled"
        ),
        "HELION_AUTOTUNER": "LFBOTreeSearch",
        "HELION_AUTOTUNE_EFFORT": "full",
        "HELION_AUTOTUNE_RANDOM_SEED": "0",
        "HELION_BACKEND": "cute",
        "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "-1",
        "HELION_CUTE_MMA_IMPL": "tcgen05",
    }
    if selection == "live_autotune":
        environment["HELION_SKIP_CACHE"] = "1"
    return environment


def _run_worker(args: argparse.Namespace) -> int:
    selection_environment = _selection_environment(args.helion_selection)
    expected_selection_environment = {
        key: selection_environment.get(key)
        for key in HELION_SELECTION_ENVIRONMENT_VARIABLES
    }
    if expected_selection_environment != HELION_SELECTION_STARTUP_ENVIRONMENT:
        raise RuntimeError("Helion selection environment was not set before startup")
    if os.environ.get("HELION_HEURISTIC_DIR"):
        raise RuntimeError("--worker does not permit HELION_HEURISTIC_DIR")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir() or (run_dir / "result.json").exists():
        raise RuntimeError(f"worker requires a fresh run directory: {run_dir}")

    common.require_single_visible_device()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    device_info = _device_info(device)
    oracle_precision = common.configure_oracle_precision()
    from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles

    cases = common.official_cases()
    if len(cases) != 8 or len(reviewed_profiles.OFFICIAL_SHAPES) != 8:
        raise RuntimeError("benchmark requires exactly eight reviewed cases")
    torch.manual_seed(0)
    rows = [
        run_case(
            args.provider,
            case,
            shape,
            device,
            helion_selection=args.helion_selection,
            cutlass_root=args.cutlass_root,
            deepgemm_root=args.deepgemm_root,
        )
        for case, shape in zip(
            cases,
            reviewed_profiles.OFFICIAL_SHAPES,
            strict=True,
        )
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "provider": args.provider,
        "replicate": args.replicate,
        "helion_selection": args.helion_selection,
        "device": device_info,
        "source": _source_identity(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "settings": {
            "input_seed": 0,
            "provider_selection": PROVIDER_SELECTIONS[args.provider],
            "helion_selection_timed": False,
            "provider_selection_timed": False,
            "capture_warmups": CAPTURE_WARMUPS,
            "repetitions": BENCHMARK_REPETITIONS,
            "thermal_warmup_ms": THERMAL_WARMUP_MS,
            "cold_l2": True,
            "balanced_rotated_reversed_order": True,
            "oracle_float32_matmul_precision": oracle_precision,
        },
        "rows": rows,
    }
    json.dumps(result, allow_nan=False)
    common.write_result(run_dir / "result.json", result)
    print(json.dumps({"result": str(run_dir / "result.json")}))
    return 0


def _provider_roots(args: argparse.Namespace) -> dict[str, Path]:
    roots = {
        provider: root
        for provider, root in (
            ("deepgemm", args.deepgemm_root),
            ("cutlass", args.cutlass_root),
        )
        if root is not None
    }
    for provider in ("deepgemm", "cutlass"):
        if provider in args.providers and provider not in roots:
            raise ValueError(f"--{provider}-root is required for {provider}")
        if provider not in args.providers and provider in roots:
            raise ValueError(f"--{provider}-root requires selecting {provider}")
    return roots


def _worker_environment(
    run_dir: Path,
    *,
    cuda_visible_devices: str,
    helion_selection: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith((*WORKER_CONTROL_PREFIXES, "PYTHON")) or (
            name in COMPILER_AND_LOADER_CONTROLS
        ):
            environment.pop(name)
    cuda_home, cudart = _installed_cuda_runtime()
    environment.update(_selection_environment(helion_selection))
    environment.update(_worker_cache_directories(run_dir))
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["CUDA_HOME"] = str(cuda_home)
    environment["CUDA_PATH"] = str(cuda_home)
    environment["CUDNN_FRONTEND_CUDART_LIB_NAME"] = str(cudart)
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
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return environment


def _installed_cuda_runtime() -> tuple[Path, Path]:
    distribution = importlib.metadata.distribution("nvidia-cuda-runtime")
    candidates = [
        Path(str(distribution.locate_file(path))).resolve(strict=True)
        for path in distribution.files or ()
        if Path(path).name == "libcudart.so.13"
    ]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise RuntimeError(
            "installed nvidia-cuda-runtime must contain one libcudart.so.13"
        )
    cudart = candidates[0]
    cuda_home = cudart.parents[1]
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"installed CUDA runtime has no nvcc: {cuda_home}")
    return cuda_home, cudart


def _worker_command(
    args: argparse.Namespace,
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
        str(Path(__file__).resolve()),
        "--worker",
        "--provider",
        provider,
        "--replicate",
        str(replicate),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(args.output_dir),
        "--helion-selection",
        args.helion_selection,
    ]
    if provider in roots:
        command.extend((f"--{provider}-root", str(roots[provider])))
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


def _require_target_gpu_idle(target_gpu_uuid: str) -> None:
    output = _nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    for row in (line.strip() for line in output.splitlines() if line.strip()):
        fields = tuple(field.strip() for field in row.split(",", maxsplit=3))
        if len(fields) != 4:
            raise RuntimeError("compute-application query returned malformed CSV")
        if fields[0] == target_gpu_uuid:
            raise RuntimeError(f"target GPU is not idle: {row}")


def _query_telemetry(target_gpu_uuid: str) -> str:
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
    if len(values) != len(TELEMETRY_FIELDS) or values[1] != target_gpu_uuid:
        raise RuntimeError("telemetry query did not return the target GPU UUID")
    return ",".join(values)


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


def _summarize_telemetry(path: Path, target_gpu_uuid: str) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or tuple(lines[0].split(",")) != TELEMETRY_FIELDS:
        raise RuntimeError(f"malformed telemetry header: {path}")
    active_reasons: Counter[str] = Counter()
    for line in lines[1:]:
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != len(TELEMETRY_FIELDS) or fields[1] != target_gpu_uuid:
            raise RuntimeError(f"telemetry is not bound to {target_gpu_uuid}: {path}")
        try:
            active = int(fields[-1], 0)
        except ValueError as error:
            raise RuntimeError(f"invalid active clock-event reason: {path}") from error
        if active:
            active_reasons[fields[-1]] += 1
    return {
        "sample_count": len(lines) - 1,
        "active_clock_event_reason_sample_count": sum(active_reasons.values()),
        "active_clock_event_reasons": dict(sorted(active_reasons.items())),
    }


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
    telemetry_path: Path,
    target_gpu_uuid: str,
) -> tuple[int, int]:
    samples = 0
    _require_target_gpu_idle(target_gpu_uuid)
    try:
        with log_path.open("xb") as log, telemetry_path.open("x") as telemetry:
            telemetry.write(",".join(TELEMETRY_FIELDS) + "\n")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                next_sample = time.monotonic()
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= next_sample:
                        telemetry.write(_query_telemetry(target_gpu_uuid) + "\n")
                        telemetry.flush()
                        samples += 1
                        next_sample = now + TELEMETRY_INTERVAL_SECONDS
                    time.sleep(min(0.25, max(0.0, next_sample - time.monotonic())))
                telemetry.write(_query_telemetry(target_gpu_uuid) + "\n")
                returncode = process.wait()
                return returncode, samples + 1
            finally:
                _terminate_process(process)
    finally:
        _wait_for_target_gpu_idle(target_gpu_uuid)


def _geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geomean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_results(
    results: Sequence[dict[str, Any]],
    *,
    providers: Sequence[str],
    replicates: int,
    helion_selection: str,
) -> dict[str, Any]:
    row_count = len(common.official_cases())
    provider_summaries = {}
    config_hashes: list[list[str]] = [[] for _ in range(row_count)]
    for provider in providers:
        provider_results = sorted(
            (result for result in results if result["provider"] == provider),
            key=itemgetter("replicate"),
        )
        if len(provider_results) != replicates:
            raise RuntimeError(f"{provider} produced an incomplete replicate set")
        if [result.get("replicate") for result in provider_results] != list(
            range(replicates)
        ) or any(
            result.get("schema") != RESULT_SCHEMA
            or result.get("provider") != provider
            or result.get("helion_selection") != helion_selection
            for result in provider_results
        ):
            raise RuntimeError(f"{provider} result identity is inconsistent")
        replicate_summaries = []
        row_speedups: list[list[float]] = [[] for _ in range(row_count)]
        for result in provider_results:
            rows = result["rows"]
            if len(rows) != row_count or [row.get("row_index") for row in rows] != list(
                range(row_count)
            ):
                raise RuntimeError(
                    f"{provider} result does not contain {row_count} rows"
                )
            speedups = [float(row["timings"]["helion_speedup"]) for row in rows]
            for row_index, (row, speedup) in enumerate(
                zip(rows, speedups, strict=True)
            ):
                row_speedups[row_index].append(speedup)
                config_hashes[row_index].append(
                    row["configs"]["helion"]["config_sha256"]
                )
            worst_index = min(range(row_count), key=speedups.__getitem__)
            replicate_summaries.append(
                {
                    "replicate": result["replicate"],
                    "geomean_speedup": _geomean(speedups),
                    "wins": sum(speedup > 1.0 for speedup in speedups),
                    "worst_row": worst_index,
                    "worst_speedup": speedups[worst_index],
                }
            )
        row_summaries = [
            {
                "row_index": row_index,
                "geomean_speedup": _geomean(speedups),
                "replicate_speedups": speedups,
            }
            for row_index, speedups in enumerate(row_speedups)
        ]
        all_speedups = [speedup for row in row_speedups for speedup in row]
        worst_row = min(row_summaries, key=itemgetter("geomean_speedup"))
        provider_summaries[provider] = {
            "selection": PROVIDER_SELECTIONS[provider],
            "cross_replicate_geomean": _geomean(all_speedups),
            "row_wins": sum(row["geomean_speedup"] > 1.0 for row in row_summaries),
            "row_count": row_count,
            "worst_row": worst_row,
            "replicates": replicate_summaries,
            "rows": row_summaries,
        }
    distributions = [
        {
            "row_index": row_index,
            "config_sha256_counts": dict(sorted(Counter(hashes).items())),
            "invariant": len(set(hashes)) == 1,
        }
        for row_index, hashes in enumerate(config_hashes)
    ]
    if helion_selection != "live_autotune" and not all(
        item["invariant"] for item in distributions
    ):
        raise RuntimeError("fixed Helion config changed across fresh workers")
    return {
        "schema": SUMMARY_SCHEMA,
        "providers": list(providers),
        "replicates": replicates,
        "helion_selection": helion_selection,
        "speedup_definition": "provider_ms / helion_ms; higher favors Helion",
        "protocol": {
            "fresh_process_and_caches_per_provider_replicate": True,
            "provider_worker_order": "rotated_then_reversed",
            "rows_per_replicate": row_count,
            "thermal_warmup_ms": THERMAL_WARMUP_MS,
            "paired_cold_l2_samples": BENCHMARK_REPETITIONS,
            "balanced_rotated_reversed_order": True,
        },
        "helion_config_distributions": distributions,
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
    monitoring = summary["monitoring"]
    if monitoring["active_clock_event_reason_sample_count"]:
        print(
            "active GPU clock-event reasons: "
            f"{monitoring['active_clock_event_reasons']}"
        )


def _provider_order(providers: Sequence[str], replicate: int) -> tuple[str, ...]:
    order = list(providers)
    if not order:
        return ()
    offset = replicate % len(order)
    order = order[offset:] + order[:offset]
    if (replicate // len(order)) % 2:
        order.reverse()
    return tuple(order)


def _run_campaign(args: argparse.Namespace) -> int:
    roots = _provider_roots(args)
    target_gpu_uuid = _resolve_target_gpu(args.cuda_visible_devices)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(REPO_ROOT):
        raise ValueError("output directory must be outside the Helion checkout")
    source = _source_identity()
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for replicate in range(args.replicates):
        for provider in _provider_order(args.providers, replicate):
            run_id = f"{provider}-r{replicate}"
            run_dir = output_dir / run_id
            run_dir.mkdir()
            command = _worker_command(args, provider, replicate, run_dir, roots)
            telemetry_path = run_dir / "telemetry.csv"
            log_path = run_dir / "worker.log"
            print(f"=== {run_id} ===", flush=True)
            returncode, samples = _run_monitored_worker(
                command,
                environment=_worker_environment(
                    run_dir,
                    cuda_visible_devices=target_gpu_uuid,
                    helion_selection=args.helion_selection,
                ),
                log_path=log_path,
                telemetry_path=telemetry_path,
                target_gpu_uuid=target_gpu_uuid,
            )
            if returncode:
                tail = "".join(
                    log_path.read_text(errors="replace").splitlines(True)[-30:]
                )
                raise RuntimeError(f"{run_id} failed with {returncode}:\n{tail}")
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                raise RuntimeError(f"{run_id} did not produce result.json")
            result = json.loads(result_path.read_text())
            if result.get("source") != source:
                raise RuntimeError(f"{run_id} worker source identity changed")
            device = result.get("device")
            if not isinstance(device, dict) or (
                device.get("uuid") != target_gpu_uuid
                or device.get("visible") != target_gpu_uuid
            ):
                raise RuntimeError(f"{run_id} worker used the wrong GPU: {device}")
            monitoring = _summarize_telemetry(telemetry_path, target_gpu_uuid)
            if monitoring["sample_count"] != samples:
                raise RuntimeError(f"{run_id} telemetry sample count changed")
            results.append(result)
            runs.append(
                {
                    "provider": provider,
                    "replicate": replicate,
                    "result": str(result_path.relative_to(output_dir)),
                    "log": str(log_path.relative_to(output_dir)),
                    "telemetry": str(telemetry_path.relative_to(output_dir)),
                    "telemetry_samples": monitoring["sample_count"],
                    "active_clock_event_reason_sample_count": monitoring[
                        "active_clock_event_reason_sample_count"
                    ],
                    "active_clock_event_reasons": monitoring[
                        "active_clock_event_reasons"
                    ],
                }
            )
            if _source_identity() != source:
                raise RuntimeError("Helion checkout changed during the campaign")
    summary = summarize_results(
        results,
        providers=args.providers,
        replicates=args.replicates,
        helion_selection=args.helion_selection,
    )
    summary["source"] = source
    active_reasons: Counter[str] = Counter()
    for run in runs:
        active_reasons.update(run["active_clock_event_reasons"])
    summary["monitoring"] = {
        "target_gpu_uuid": target_gpu_uuid,
        "target_gpu_idle_before_after_each_run": True,
        "telemetry_sample_count": sum(run["telemetry_samples"] for run in runs),
        "active_clock_event_reason_sample_count": sum(active_reasons.values()),
        "active_clock_event_reasons": dict(sorted(active_reasons.items())),
        "runs_with_active_clock_event_reasons": [
            f"{run['provider']}-r{run['replicate']}"
            for run in runs
            if run["active_clock_event_reason_sample_count"]
        ],
        "caveat": (
            "One or more telemetry samples reported active GPU clock-event reasons."
            if active_reasons
            else "No telemetry sample reported an active GPU clock-event reason."
        ),
    }
    summary["runs"] = runs
    common.write_result(output_dir / "summary.json", summary)
    if _source_identity() != source:
        raise RuntimeError("Helion checkout changed while writing the summary")
    _print_summary(summary)
    print(f"\nWrote {output_dir / 'summary.json'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--providers",
        type=parse_providers,
        default=PROVIDERS,
        help=f"comma-separated provider subset (default: {','.join(PROVIDERS)})",
    )
    parser.add_argument("--replicates", type=_positive_int, default=3)
    parser.add_argument(
        "--helion-selection",
        choices=HELION_SELECTIONS,
        default="final_reviewed_aot",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES"),
        help="one CUDA_VISIBLE_DEVICES entry (GPU UUID or index)",
    )
    parser.add_argument("--cutlass-root", type=_existing_directory)
    parser.add_argument("--deepgemm-root", type=_existing_directory)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    parser.add_argument("--replicate", type=_nonnegative_int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.worker:
        if args.provider is None or args.replicate is None or args.run_dir is None:
            raise ValueError(
                "internal --worker requires provider, replicate, and run-dir"
            )
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


if __name__ == "__main__":
    raise SystemExit(main())
