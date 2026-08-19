"""Eight-shape comparison with DeepGEMM's public contiguous grouped API."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.cute import grouped_gemm_deepgemm_support as _SUPPORT
from pretuned_kernels import _bench as _BENCH
from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles as _REVIEWED
import torch

import helion.runtime as helion_runtime

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from types import ModuleType

    from pretuned_kernels.grouped_gemm_deepgemm.reviewed_profiles import (
        ReviewedWorklistProfile,
    )

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

DEEPGEMM_ROOT_ENV = "HELION_DEEPGEMM_ROOT"
BASELINE_NAME = "deepgemm_public_kmajor_nk_no_psum"


@dataclass
class CapturedReplay:
    graph: torch.cuda.CUDAGraph
    owners: tuple[object, ...]

    def __call__(self) -> object:
        return self.graph.replay()


def _capture(
    call: Callable[[], torch.Tensor],
    owners: Sequence[object],
    *,
    track_cute: bool,
) -> tuple[CapturedReplay, torch.Tensor]:
    """Compile, capture, verify one replay, and retain pointer owners."""
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    if track_cute:
        with helion_runtime.cute_cuda_graph() as graph:
            output = call()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
    output.fill_(13.0)
    graph.replay()
    torch.cuda.synchronize()
    return CapturedReplay(graph, (*owners, output, graph)), output


def _deepgemm_root() -> Path:
    value = os.environ.get(DEEPGEMM_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"{DEEPGEMM_ROOT_ENV} must point to built DeepGEMM commit "
            f"{_SUPPORT.DEEPGEMM_COMMIT}"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{DEEPGEMM_ROOT_ENV} does not exist: {root}")
    return root


def _make_reviewed_case(
    shape: _REVIEWED.OfficialShape,
    actual_ms: Sequence[int],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    profile = _REVIEWED.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    a, logical_b, layout, reference, worklist = _SUPPORT.make_case(
        shape.groups,
        shape.n,
        shape.k,
        actual_ms,
        device,
        profile.source_m_tile,
    )
    helion_b = (
        logical_b.transpose(1, 2).contiguous().transpose(1, 2)
        if profile.b_major == "n"
        else logical_b
    )
    return a, logical_b, helion_b, layout, reference, worklist


def _check_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    layout: torch.Tensor,
    *,
    require_zero_padding: bool,
) -> None:
    result = _SUPPORT.correctness(
        output,
        reference,
        layout,
        max_diff=1e-5,
        require_zero_padding=require_zero_padding,
    )
    if not result["ok"]:
        raise RuntimeError(f"grouped GEMM correctness failed: {result}")


def _launch_deepgemm(
    deep_gemm: ModuleType,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    layout: torch.Tensor,
) -> torch.Tensor:
    """Call the public API while excluding undefined padding from the work."""
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        a,
        b,
        output,
        layout,
        compiled_dims="nk",
        use_psum_layout=False,
        ensure_zero_padding=False,
    )
    return output


def _effective_reviewed_config(
    bound: BoundKernel[torch.Tensor], profile: ReviewedWorklistProfile
) -> dict[str, dict[str, object]]:
    requested = _REVIEWED.reviewed_config_values(profile.config_name)
    actual = bound._config
    if actual is None or actual.config != requested:
        raise RuntimeError(
            "AOT evaluation did not select the exact reviewed config "
            f"{profile.config_name}"
        )
    expected_effective = bound.config_spec.normalized_config(requested)
    effective = bound.config_spec.normalized_config(actual)
    if effective.config != expected_effective.config:
        raise RuntimeError(
            f"reviewed config normalization changed for {profile.config_name}"
        )
    return {
        "requested": dict(actual.config),
        "effective": dict(effective.config),
    }


def _captured_calls(
    shape_and_ms: tuple[_REVIEWED.OfficialShape, tuple[int, ...]],
    deep_gemm: ModuleType,
    kernel_factory: Callable[[], Kernel[torch.Tensor]],
    selected_configs: dict[int, dict[str, object]],
) -> tuple[Callable[[], object], list[tuple[str, Callable[[], object]]], str]:
    shape, actual_ms = shape_and_ms
    device = torch.device("cuda", torch.cuda.current_device())
    a, b, helion_b, layout, reference, worklist = _make_reviewed_case(
        shape,
        actual_ms,
        device,
    )
    external_a, external_b, external_layout, external_reference, _worklist = (
        _SUPPORT.repack_case_alignment(
            a,
            b,
            reference,
            worklist,
            actual_ms,
            _SUPPORT.M_ALIGNMENT,
        )
    )
    profile = _REVIEWED.exact_reviewed_worklist_profile(
        shape.groups,
        shape.expected_m_per_group,
        shape.n,
        shape.k,
    )
    kernel_args = (
        a,
        helion_b,
        worklist,
        shape.expected_m_per_group,
        profile.source_m_tile,
    )
    bound = kernel_factory().bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    helion_replay, helion_output = _capture(
        lambda: bound(*kernel_args),
        (bound, *kernel_args),
        track_cute=True,
    )
    selected_configs[shape.row_index] = {
        "config_name": profile.config_name,
        "b_major": profile.b_major,
        "source_m_tile": profile.source_m_tile,
        "config": _effective_reviewed_config(bound, profile),
    }

    deep_output = torch.empty_like(external_reference, dtype=external_a.dtype)

    def deepgemm_call() -> torch.Tensor:
        return _launch_deepgemm(
            deep_gemm,
            external_a,
            external_b,
            deep_output,
            external_layout,
        )

    deepgemm_replay, deepgemm_output = _capture(
        deepgemm_call,
        (external_a, external_b, external_layout, deep_output),
        track_cute=False,
    )
    _check_output(helion_output, reference, layout, require_zero_padding=True)
    _check_output(
        deepgemm_output,
        external_reference,
        external_layout,
        require_zero_padding=False,
    )
    actual_label = ",".join(str(value) for value in actual_ms)
    label = (
        f"row{shape.row_index} G{shape.groups} M=[{actual_label}] "
        f"N={shape.n} K={shape.k}"
    )
    return helion_replay, [(BASELINE_NAME, deepgemm_replay)], f"{label:<68s}"


def _run_aot_training(
    kernel_factory: Callable[[], Kernel[torch.Tensor]], verbose: bool
) -> dict[str, object]:
    """Tune the official shapes without loading the external baseline."""
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    actual_ms_by_shape = _REVIEWED.official_actual_ms(seed=0)
    for shape, actual_ms in zip(
        _REVIEWED.OFFICIAL_SHAPES,
        actual_ms_by_shape,
        strict=True,
    ):
        a, _b, helion_b, layout, reference, worklist = _make_reviewed_case(
            shape,
            actual_ms,
            device,
        )
        profile = _REVIEWED.exact_reviewed_worklist_profile(
            shape.groups,
            shape.expected_m_per_group,
            shape.n,
            shape.k,
        )
        kernel_args = (
            a,
            helion_b,
            worklist,
            shape.expected_m_per_group,
            profile.source_m_tile,
        )
        bound = kernel_factory().bind(kernel_args)
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        output = bound(*kernel_args)
        torch.cuda.synchronize()
        _check_output(output, reference, layout, require_zero_padding=True)
        if verbose:
            print(
                f"AOT {os.environ['HELION_AOT_MODE']}: "
                f"DeepGEMM row {shape.row_index} passed"
            )
    return {
        "helion_wins": 0,
        "total": 0,
        "geomean": 0.0,
        "best_speedup": 0.0,
        "baselines": {},
    }


def main(
    kernel_factory: Callable[[], Kernel[torch.Tensor]], verbose: bool = True
) -> dict[str, object]:
    """Run the short comparison or the kernel's collect/measure phases."""
    env_name = "HELION_CUTE_MMA_IMPL"
    previous_mma_impl = os.environ.get(env_name)
    os.environ[env_name] = "tcgen05"
    try:
        aot_mode = os.environ.get("HELION_AOT_MODE", "evaluate").lower()
        if aot_mode in {
            "collect",
            "measure",
        }:
            return _run_aot_training(kernel_factory, verbose)
        if aot_mode != "evaluate":
            raise RuntimeError(
                "the public-API benchmark requires HELION_AOT_MODE=evaluate"
            )
        if os.environ.get("HELION_HEURISTIC_DIR"):
            raise RuntimeError(
                "the public-API benchmark does not permit HELION_HEURISTIC_DIR"
            )

        deep_gemm, provenance = _SUPPORT.import_deepgemm(
            _deepgemm_root(),
            _SUPPORT.M_ALIGNMENT,
        )
        torch.manual_seed(0)
        cases = tuple(
            zip(
                _REVIEWED.OFFICIAL_SHAPES,
                _REVIEWED.official_actual_ms(seed=0),
                strict=True,
            )
        )
        selected_configs: dict[int, dict[str, object]] = {}
        result = _BENCH.run_sweep(
            cases,
            lambda case: _captured_calls(
                case,
                deep_gemm,
                kernel_factory,
                selected_configs,
            ),
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            rep=102,
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=f"{'official shape and actual per-group M':<68s}",
        )
        expected_rows = {shape.row_index for shape in _REVIEWED.OFFICIAL_SHAPES}
        if selected_configs.keys() != expected_rows:
            raise RuntimeError(
                "benchmark did not validate every reviewed Helion config"
            )
        result["benchmark_metadata"] = {
            "reviewed_profile_manifest_sha256": (
                _REVIEWED.REVIEWED_PROFILE_MANIFEST_SHA256
            ),
            "deepgemm_provenance": provenance,
            "deepgemm_api": {
                "function": "m_grouped_bf16_gemm_nt_contiguous",
                "b_major": "k",
                "compiled_dims": "nk",
                "use_psum_layout": False,
                "ensure_zero_padding": False,
                "m_alignment": _SUPPORT.M_ALIGNMENT,
            },
        }
        result["benchmark_details"] = {
            "reviewed_helion_configs": [
                {"row_index": row_index, **selected_configs[row_index]}
                for row_index in sorted(selected_configs)
            ],
        }
        return result
    finally:
        if previous_mma_impl is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous_mma_impl
