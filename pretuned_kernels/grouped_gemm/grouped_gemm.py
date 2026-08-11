"""Pretuned B200 grouped FP16 GEMM versus CUTLASS CuTeDSL.

This benchmark uses seven CUTLASS-example-derived heterogeneous FP16 NT
validation cases (3--4 GEMMs, including M/N tails). Both implementations consume
the same A/B tensors and write the same output buffers sequentially. Each passes
correctness checks and is timed through CUDA-graph replay. CUTLASS initializes
its device pointer table before graph capture, matching Helion's setup.

The pretuned dashboard clears L2 before every replay and batches CUDA event
pairs so host graph-submission work is not charged as device time. This matches
the ``triton.testing.do_bench`` device-timing semantics used in PR #3033.

The CUTLASS source is pinned by commit and SHA256 in the shared comparison
harness. Before a manual run, export
``HELION_CUTLASS_GROUPED_GEMM_SOURCE=/path/to/grouped_gemm.py`` from the pinned
checkout. The nightly workflow downloads and verifies those exact bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import starmap
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast

from benchmarks.cute import compare_grouped_gemm_backends as _COMPARE
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

    from benchmarks.cute.compare_grouped_gemm_backends import Case
    from benchmarks.cute.compare_grouped_gemm_backends import PreparedLaunch

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import CompiledConfig


CUTLASS_COMMIT: str = _COMPARE.CUTLASS_COMMIT
CUTLASS_SHA256: str = _COMPARE.CUTLASS_SHA256
CASES: tuple[Case, ...] = _COMPARE.CASES
CTA_M: int = _COMPARE.CTA_M
CTA_N: int = _COMPARE.CTA_N
CTA_K: int = _COMPARE.CTA_K
CUTLASS_SOURCE_ENV = "HELION_CUTLASS_GROUPED_GEMM_SOURCE"
CUTLASS_KERNEL_BASELINE: str = _COMPARE.CUTLASS_KERNEL_BASELINE
STATIC_PROBLEM_SIGNATURE_CONFIG_KEY = "tcgen05_grouped_static_problem_signature"
_MAX_GROUPS = max(len(case.problems) for case in CASES)


def _problem_signature(
    problems: Sequence[tuple[int, int, int, int]],
) -> tuple[int, ...]:
    """Return a fixed-width AOT feature vector with every group's real M/N/K."""
    if any(batch != 1 for _m, _n, _k, batch in problems):
        raise ValueError("grouped_gemm only supports L=1 problems")
    values = [len(problems)]
    for m, n, k, _l in problems:
        values.extend((m, n, k))
    values.extend((0, 0, 0) * (_MAX_GROUPS - len(problems)))
    return tuple(values)


def _direct_config(ab_stages: int, acc_stages: int, c_stages: int) -> helion.Config:
    config: dict[str, object] = {
        "block_sizes": [CTA_M, CTA_N, CTA_K],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "num_stages": 2,
        "num_warps": 8,
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": ab_stages,
        "tcgen05_acc_stages": acc_stages,
        "tcgen05_c_stages": c_stages,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_grouped_mode": "direct",
        "tcgen05_grouped_external_direct_pointers": "direct_pointers",
        "tcgen05_grouped_external_direct_strides": "direct_strides",
    }
    return helion.Config.from_dict(config)


_PIPELINE_STAGE_TRIPLES = ((2, 1, 2), (8, 2, 4))
_AOT_CONFIGS = tuple(starmap(_direct_config, _PIPELINE_STAGE_TRIPLES))
_AOT_BENCH_REPETITIONS = 500


def _configs_for_problem_signature(
    signature: tuple[int, ...],
) -> tuple[helion.Config, ...]:
    active_signature = list(signature[: 1 + 3 * signature[0]])
    return tuple(
        helion.Config.from_dict(
            {
                **config.config,
                STATIC_PROBLEM_SIGNATURE_CONFIG_KEY: active_signature,
            }
        )
        for config in _AOT_CONFIGS
    )


@dataclass(frozen=True)
class _GroupedValidation:
    outputs: tuple[torch.Tensor, ...]
    expected: tuple[torch.Tensor, ...]


class _ColdCudagraphBenchmarkProvider(_BENCH.CapturedCudagraphBenchmarkProvider):
    """Score direct-pointer candidates under the dashboard's cold-L2 protocol."""

    def __init__(
        self,
        *args: object,
        validation: _GroupedValidation | None,
        **kwargs: object,
    ) -> None:
        if validation is None:
            raise ValueError("grouped-GEMM autotuning requires validation context")
        self._validation = validation
        super().__init__(
            *args,
            repetitions=_AOT_BENCH_REPETITIONS,
            **kwargs,
        )

    def _compute_baseline(self) -> tuple[object, Sequence[int], None]:
        """Skip the generic clone, which cannot rebase the raw pointer table."""
        return None, (), None

    def _capture_validated_replay(
        self, config: helion.Config, fn: CompiledConfig
    ) -> Callable[[], object] | None:
        for _ in range(2):
            fn(*self.args)
        torch.cuda.synchronize()
        with helion_runtime.cute_cuda_graph() as graph:
            fn(*self.args)
        torch.cuda.synchronize()

        if self.settings.autotune_accuracy_check:
            for output in self._validation.outputs:
                output.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            try:
                _COMPARE.check_correctness(
                    self._validation.outputs, self._validation.expected
                )
            except AssertionError as error:
                self._record_accuracy_failure(config)
                if not self.settings.autotune_ignore_errors:
                    self.log.warning(
                        "Skipping grouped-GEMM config with accuracy mismatch: "
                        f"{config!r}\n{error}"
                    )
                return None
        return graph.replay


def _grouped_aot_key(
    a_placeholder: torch.Tensor,
    b_placeholder: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out_placeholder: torch.Tensor,
    direct_pointers: torch.Tensor,
    direct_strides: torch.Tensor,
    problem_signature: tuple[int, ...],
) -> tuple[int, ...]:
    """Expose real per-group problem sizes instead of placeholder shapes."""
    return problem_signature


def _grouped_aot_autotuner(
    bound_kernel: BoundKernel,
    args: Sequence[object],
    **kwargs: object,
) -> AOTAutotuneCache:
    """Use AOT caching around the direct path's finite valid config space."""
    configs = _configs_for_problem_signature(cast("tuple[int, ...]", args[-1]))
    provider = partial(
        _ColdCudagraphBenchmarkProvider,
        validation=cast("_GroupedValidation | None", kwargs.get("benchmark_context")),
    )

    def make_search() -> FiniteSearch:
        return FiniteSearch(
            bound_kernel,
            args,
            configs=configs,
            benchmark_provider_cls=provider,
        )

    return AOTAutotuneCache(make_search(), autotuner_factory=make_search)


@helion.aot_kernel(
    backend="cute",
    key=_grouped_aot_key,
    static_shapes=True,
    autotuner_fn=_grouped_aot_autotuner,
    autotune_precompile=None,
)
def grouped_gemm(
    a_placeholder: torch.Tensor,
    b_placeholder: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out_placeholder: torch.Tensor,
    direct_pointers: torch.Tensor,
    direct_strides: torch.Tensor,
    problem_signature: tuple[int, ...],
) -> torch.Tensor:
    """Grouped A[M,K] @ B[G,N,K].T with direct runtime metadata."""
    m, max_k = a_placeholder.size()
    _groups, max_n, _k = b_placeholder.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        valid_rows = layout[tile_m] == safe_group_id
        valid_cols = tile_n.index < n_sizes[safe_group_id]
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):  # pyrefly: ignore[bad-assignment]
            valid_k = (tile_k.index < group_k)[None, :]  # pyrefly: ignore[bad-index]
            a_tile = a_placeholder[tile_m, tile_k]
            b_tile = b_placeholder[safe_group_id, tile_n, tile_k]
            a_tile = torch.where(valid_k, a_tile, torch.zeros_like(a_tile))
            b_tile = torch.where(valid_k, b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, a_tile, b_tile.T)
        old = out_placeholder[tile_m, tile_n]
        out_placeholder[tile_m, tile_n] = torch.where(
            valid, acc.to(out_placeholder.dtype), old
        )
    return out_placeholder


def _placeholder(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    base = torch.empty(max(1, shape[-1]), device=device, dtype=torch.float16)
    return torch.as_strided(base, shape, (0,) * (len(shape) - 1) + (1,))


def _prepare_helion(
    case: Case,
    group_a: tuple[torch.Tensor, ...],
    group_b: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> PreparedLaunch:
    problems = case.problems
    device = group_a[0].device
    aligned_m = tuple((m + CTA_M - 1) // CTA_M * CTA_M for m, _n, _k, _l in problems)
    padded_m = sum(aligned_m)
    max_n = max(n for _m, n, _k, _l in problems)
    max_k = max(k for _m, _n, k, _l in problems)
    layout = torch.empty(padded_m, device=device, dtype=torch.int32)
    cursor = 0
    for group, ((m, _n, _k, _l), padded) in enumerate(
        zip(problems, aligned_m, strict=True)
    ):
        layout[cursor : cursor + m].fill_(group)
        layout[cursor + m : cursor + padded].fill_(-1)
        cursor += padded
    n_sizes = torch.tensor([p[1] for p in problems], device=device, dtype=torch.int32)
    k_sizes = torch.tensor([p[2] for p in problems], device=device, dtype=torch.int32)
    direct_pointers = torch.tensor(
        [
            (a.data_ptr(), b.data_ptr(), out.data_ptr())
            for a, b, out in zip(group_a, group_b, outputs, strict=True)
        ],
        device=device,
        dtype=torch.int64,
    )
    direct_strides = torch.tensor(
        [
            (tuple(a.stride()), tuple(b.stride()), tuple(out.stride()))
            for a, b, out in zip(group_a, group_b, outputs, strict=True)
        ],
        device=device,
        dtype=torch.int32,
    )
    kernel_args = (
        _placeholder((padded_m, max_k), device),
        _placeholder((len(problems), max_n, max_k), device),
        layout,
        n_sizes,
        k_sizes,
        _placeholder((padded_m, max_n), device),
        direct_pointers,
        direct_strides,
        _problem_signature(problems),
    )

    if os.environ.get("HELION_AOT_MODE", "evaluate").lower() in {
        "collect",
        "measure",
    }:
        grouped_gemm.autotune(
            kernel_args,
            force=False,
            benchmark_context=_GroupedValidation(outputs, expected),
        )

    def launch() -> object:
        with torch.cuda.device(device):
            return grouped_gemm(*kernel_args)

    owners: tuple[object, ...] = (*group_a, *group_b, *outputs, *kernel_args)
    return _COMPARE.PreparedLaunch(launch, owners)


def _cutlass_source_path() -> Path:
    raw_path = os.environ.get(CUTLASS_SOURCE_ENV)
    if not raw_path:
        raise RuntimeError(
            f"{CUTLASS_SOURCE_ENV} must point to CUTLASS commit {CUTLASS_COMMIT} "
            "examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/"
            "grouped_gemm.py"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{CUTLASS_SOURCE_ENV} does not exist: {path}")
    return path


def _captured_calls(
    case: Case,
    cutlass_module: ModuleType,
) -> tuple[Callable[[], object], list[tuple[str, Callable[[], object]]], str]:
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    group_a, group_b, expected = _COMPARE.make_inputs(case.problems, device)
    outputs = _COMPARE.make_outputs(group_a, group_b)
    helion_launch = _prepare_helion(case, group_a, group_b, outputs, expected)
    cutlass_launch = _COMPARE.prepare_cutlass(
        cutlass_module, case.problems, group_a, group_b, outputs
    )
    helion_graph = _COMPARE.capture_launch(helion_launch, 2, track_cute=True)
    cutlass_kernel_graph = _COMPARE.capture_launch(cutlass_launch, 2)

    for graph in (helion_graph, cutlass_kernel_graph):
        for output in outputs:
            output.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        _COMPARE.check_correctness(outputs, expected)

    # Default arguments retain every TensorMap, pointer table, tensor, launch,
    # and graph for the lifetime of the timed closure.
    def helion_call(
        graph: torch.cuda.CUDAGraph = helion_graph,
        owners: tuple[object, ...] = (helion_launch, helion_graph),
    ) -> object:
        return graph.replay()

    def cutlass_kernel_call(
        graph: torch.cuda.CUDAGraph = cutlass_kernel_graph,
        owners: tuple[object, ...] = (cutlass_launch, cutlass_kernel_graph),
    ) -> object:
        return graph.replay()

    return (
        helion_call,
        [(CUTLASS_KERNEL_BASELINE, cutlass_kernel_call)],
        case.shape_label,
    )


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured implementation CUDA graphs."""
    return True


def _run_aot_training(verbose: bool) -> dict[str, object]:
    """Exercise every case once; the active AOT mode owns any tuning work."""
    for case in CASES:
        torch.manual_seed(0)
        device = torch.device("cuda", torch.cuda.current_device())
        group_a, group_b, expected = _COMPARE.make_inputs(case.problems, device)
        outputs = _COMPARE.make_outputs(group_a, group_b)
        launch = _prepare_helion(case, group_a, group_b, outputs, expected)
        launch()
        torch.cuda.synchronize()
        _COMPARE.check_correctness(outputs, expected)
        if verbose:
            print(f"AOT {os.environ['HELION_AOT_MODE']}: {case.name} passed")
    return {
        "helion_wins": 0,
        "total": 0,
        "geomean": 0.0,
        "best_speedup": 0.0,
        "baselines": {},
    }


def main(verbose: bool = True) -> dict[str, object]:
    """Benchmark seven CUTLASS-derived grouped-GEMM cases for the dashboard."""
    previous_mma_impl = os.environ.get("HELION_CUTE_MMA_IMPL")
    os.environ["HELION_CUTE_MMA_IMPL"] = "tcgen05"
    try:
        if os.environ.get("HELION_AOT_MODE", "evaluate").lower() in {
            "collect",
            "compile",
            "measure",
        }:
            return _run_aot_training(verbose)

        cutlass_module, _provenance = _COMPARE.load_cutlass_source(
            _cutlass_source_path()
        )

        def make_calls(
            case: Case,
        ) -> tuple[Callable[[], object], list[tuple[str, Callable[[], object]]], str]:
            return _captured_calls(case, cutlass_module)

        # The closures replay already-captured graphs, so ``use_cudagraph`` is
        # false here to avoid illegal nested capture. The dedicated timer still
        # clears L2 before every replay and balances implementation order.
        return _BENCH.run_sweep(
            CASES,
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            rep=204,
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header="shape",
        )
    finally:
        if previous_mma_impl is None:
            os.environ.pop("HELION_CUTE_MMA_IMPL", None)
        else:
            os.environ["HELION_CUTE_MMA_IMPL"] = previous_mma_impl


if __name__ == "__main__":
    main()
