"""Pretuned B200 BF16 grouped GEMM versus pinned DeepGEMM.

This is the dashboard-sized version of
``benchmarks/cute/deepgemm_selected_path.py``.  It uses that harness's eight
official ``(groups, expected M/group, N, K)`` shapes and its deterministic
seed-0 stream of actual per-group M sizes.  Both implementations consume the
same packed A tensor and grouped B tensor with logically equivalent group
metadata in their implementation-specific formats, write separate outputs,
pass correctness checks, and are timed by replaying pre-captured CUDA graphs
with L2 cleared before every replay.

Set ``HELION_DEEPGEMM_ROOT`` to a clean checkout of DeepGEMM commit
``559d79fb6994a58b8a15b4b93bf13ccc16edf247`` with its extension built in
place.  The shared harness verifies that commit, its CUTLASS submodule commit,
and the effective contiguous-layout alignment before the baseline is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from benchmarks.cute import cublas_grouped_gemm as _CUBLAS
from benchmarks.cute import deepgemm_selected_path as _HARNESS
from pretuned_kernels import _bench as _BENCH
import torch

import helion
from helion.autotuner import AOTAutotuneCache
from helion.autotuner import FiniteSearch
import helion.language as hl
import helion.runtime as helion_runtime

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from types import ModuleType

    from benchmarks.cute.deepgemm_selected_path import OfficialShape

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import CompiledConfig


OFFICIAL_SHAPES: tuple[OfficialShape, ...] = _HARNESS.OFFICIAL_SHAPES
ACTUAL_MS: tuple[tuple[int, ...], ...] = _HARNESS.official_actual_ms(seed=0)
M_ALIGNMENT: int = _HARNESS.M_ALIGNMENT
TILE_M: int = _HARNESS.DEEPGEMM_SELECTED_TILE_M
TILE_N: int = _HARNESS.DEEPGEMM_SELECTED_TILE_N
TILE_K: int = _HARNESS.DEEPGEMM_SELECTED_TILE_K
DEEPGEMM_COMMIT: str = _HARNESS.DEEPGEMM_COMMIT
DEEPGEMM_CUTLASS_COMMIT: str = _HARNESS.DEEPGEMM_CUTLASS_COMMIT
DEEPGEMM_ROOT_ENV = "HELION_DEEPGEMM_ROOT"
BASELINE_NAME = "deepgemm_bf16_grouped_nt"
CUBLAS_BASELINE_NAME: str = _CUBLAS.CUBLAS_GROUPED_BASELINE


def _selected_key(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> tuple[int, ...]:
    """Specialize on the packed/grouped shapes and compact worklist length."""
    return (*a_packed.shape, *b_grouped.shape, int(worklist.size(0)))


def _worklist_config(ab_stages: int) -> helion.Config:
    """Return one valid B200 worklist-NM schedule from the finite AOT space."""
    return helion.Config(
        block_sizes=[TILE_M, TILE_N, TILE_K],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=ab_stages,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=M_ALIGNMENT,
    )


_AOT_CONFIGS = tuple(_worklist_config(ab_stages) for ab_stages in range(4, 8))


def _reference(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> torch.Tensor:
    """Reference valid rows and the required zero-filled aligned padding."""
    output = torch.zeros(
        (a_packed.size(0), b_grouped.size(1)),
        device=a_packed.device,
        dtype=a_packed.dtype,
    )
    rows = cast("list[list[int]]", worklist.cpu().tolist())
    for group, start, valid_m, _store_m in rows:
        output[start : start + valid_m] = (
            a_packed[start : start + valid_m] @ b_grouped[group].T
        )
    return output


class _ColdCudagraphBenchmarkProvider(_BENCH.CapturedCudagraphBenchmarkProvider):
    """Validate and score worklist candidates with the dashboard protocol."""

    def _capture_validated_replay(
        self, config: helion.Config, fn: CompiledConfig
    ) -> Callable[[], object] | None:
        output = fn(*self.args)
        torch.cuda.synchronize()
        if (
            self.settings.autotune_accuracy_check
            and not self._validate_against_baseline(config, output, self.args)
        ):
            self._record_accuracy_failure(config)
            return None

        with helion_runtime.cute_cuda_graph() as graph:
            captured_output = fn(*self.args)
        torch.cuda.synchronize()
        if self.settings.autotune_accuracy_check:
            captured_output.fill_(13.0)
            graph.replay()
            torch.cuda.synchronize()
            if not self._validate_against_baseline(config, captured_output, self.args):
                self._record_accuracy_failure(config)
                return None
        return graph.replay


def _deepgemm_aot_autotuner(
    bound_kernel: BoundKernel,
    args: Sequence[object],
    **kwargs: object,
) -> AOTAutotuneCache:
    """Search only the validated worklist-NM schedule family."""
    bound_kernel.env.config_spec.cute_tcgen05_search_enabled = True
    provider = partial(_ColdCudagraphBenchmarkProvider, repetitions=50)

    def make_search() -> FiniteSearch:
        return FiniteSearch(
            bound_kernel,
            args,
            configs=_AOT_CONFIGS,
            benchmark_provider_cls=provider,
        )

    return AOTAutotuneCache(make_search(), autotuner_factory=make_search)


@helion.aot_kernel(
    backend="cute",
    key=_selected_key,
    static_shapes=False,
    autotuner_fn=_deepgemm_aot_autotuner,
    autotune_precompile=None,
    autotune_baseline_fn=_reference,
    autotune_baseline_atol=3e-2,
    autotune_baseline_rtol=3e-2,
)
def grouped_gemm_deepgemm(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> torch.Tensor:
    """BF16 ``A[sum(align(Mg)),K] @ B[G,N,K].T`` using an N,M worklist."""
    m_total_aligned, k = a_packed.shape
    _groups, n, k2 = b_grouped.shape
    assert k == k2, "K dimension mismatch between A and B"
    assert worklist.size(1) == 4

    block_m = hl.register_block_size(TILE_M)
    block_n = hl.register_block_size(TILE_N)
    block_k = hl.register_block_size(TILE_K)
    out = torch.empty(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )

    for work_tile, tile_m, tile_n in hl.tile(
        [worklist.size(0), TILE_M, n],
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
    """Compile/warm up a call, capture it, and retain all pointer owners."""
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
    raw_path = os.environ.get(DEEPGEMM_ROOT_ENV)
    if not raw_path:
        raise RuntimeError(
            f"{DEEPGEMM_ROOT_ENV} must point to built DeepGEMM commit {DEEPGEMM_COMMIT}"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{DEEPGEMM_ROOT_ENV} does not exist: {path}")
    return path


def _check_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    layout: torch.Tensor,
    *,
    require_zero_padding: bool,
) -> None:
    result: dict[str, Any] = _HARNESS.correctness(
        output,
        reference,
        layout,
        max_diff=1e-3,
        padding_atol=0.0,
        require_zero_padding=require_zero_padding,
    )
    if not result["ok"]:
        raise RuntimeError(f"grouped GEMM correctness failed: {result}")


def _captured_calls(
    shape_and_ms: tuple[OfficialShape, tuple[int, ...]],
    deep_gemm: ModuleType,
) -> tuple[Callable[[], object], list[tuple[str, Callable[[], object]]], str]:
    shape, actual_ms = shape_and_ms
    device = torch.device("cuda", torch.cuda.current_device())
    a, b, layout, reference, worklist = _HARNESS.make_case(
        shape,
        actual_ms,
        device,
        M_ALIGNMENT,
    )
    kernel_args = (a, b, worklist)
    bound = grouped_gemm_deepgemm.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True

    def helion_call() -> torch.Tensor:
        return bound(*kernel_args)

    helion_replay, helion_output = _capture(
        helion_call,
        (bound, *kernel_args),
        track_cute=True,
    )

    deep_output = torch.empty_like(reference)

    def deepgemm_call() -> torch.Tensor:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, deep_output, layout)
        return deep_output

    deepgemm_replay, captured_deep_output = _capture(
        deepgemm_call,
        (a, b, layout, deep_output),
        track_cute=False,
    )

    cublas_output = torch.empty_like(reference)
    problems = tuple((m, shape.n, shape.k, 1) for m in actual_ms)
    group_a: list[torch.Tensor] = []
    group_b: list[torch.Tensor] = []
    group_c: list[torch.Tensor] = []
    start = 0
    for group, actual_m in enumerate(actual_ms):
        end = start + actual_m
        group_a.append(a[start:end])
        group_b.append(b[group])
        group_c.append(cublas_output[start:end])
        start += _HARNESS.align(actual_m, M_ALIGNMENT)
    cublas_launch, _cublas_provenance = _CUBLAS.prepare_cublas(
        problems,
        tuple(group_a),
        tuple(group_b),
        tuple(group_c),
    )

    def cublas_call() -> torch.Tensor:
        cublas_launch()
        return cublas_output

    cublas_replay, captured_cublas_output = _capture(
        cublas_call,
        (cublas_launch, cublas_output, *group_a, *group_b, *group_c),
        track_cute=False,
    )
    _check_output(
        helion_output,
        reference,
        layout,
        require_zero_padding=True,
    )
    _check_output(
        captured_deep_output,
        reference,
        layout,
        require_zero_padding=False,
    )
    _check_output(
        captured_cublas_output,
        reference,
        layout,
        require_zero_padding=False,
    )
    actual_label = ",".join(str(m) for m in actual_ms)
    shape_label = (
        f"row{shape.row_index} G{shape.groups} M=[{actual_label}] "
        f"N={shape.n} K={shape.k}"
    )
    return (
        helion_replay,
        [
            (BASELINE_NAME, deepgemm_replay),
            (CUBLAS_BASELINE_NAME, cublas_replay),
        ],
        f"{shape_label:<68s}",
    )


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured implementation graphs."""
    return True


def _run_aot_training(verbose: bool) -> dict[str, object]:
    """Tune every official shape without importing the external baseline."""
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    for shape, actual_ms in zip(OFFICIAL_SHAPES, ACTUAL_MS, strict=True):
        a, b, layout, reference, worklist = _HARNESS.make_case(
            shape,
            actual_ms,
            device,
            M_ALIGNMENT,
        )
        kernel_args = (a, b, worklist)
        bound = grouped_gemm_deepgemm.bind(kernel_args)
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        output = bound(*kernel_args)
        torch.cuda.synchronize()
        _check_output(
            output,
            reference,
            layout,
            require_zero_padding=True,
        )
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


def main(verbose: bool = True) -> dict[str, object]:
    """Benchmark the eight official DeepGEMM grouped-BF16 shapes on B200."""
    env_name = "HELION_CUTE_MMA_IMPL"
    previous_mma_impl = os.environ.get(env_name)
    os.environ[env_name] = "tcgen05"
    try:
        if os.environ.get("HELION_AOT_MODE", "evaluate").lower() in {
            "collect",
            "compile",
            "measure",
        }:
            return _run_aot_training(verbose)

        deep_gemm, _provenance = _HARNESS.import_deepgemm(
            _deepgemm_root(),
            M_ALIGNMENT,
        )
        torch.manual_seed(0)
        cases = tuple(zip(OFFICIAL_SHAPES, ACTUAL_MS, strict=True))

        def make_calls(
            shape_and_ms: tuple[OfficialShape, tuple[int, ...]],
        ) -> tuple[Callable[[], object], list[tuple[str, Callable[[], object]]], str]:
            return _captured_calls(shape_and_ms, deep_gemm)

        return _BENCH.run_sweep(
            cases,
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            rep=102,
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=f"{'official shape and actual per-group M':<68s}",
        )
    finally:
        if previous_mma_impl is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous_mma_impl


if __name__ == "__main__":
    main()
