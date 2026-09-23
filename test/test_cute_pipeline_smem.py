"""CPU proofs for explicit CTA groups and epilogue-aware pipeline depths."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.cute_mma import _tcgen05_use_2cta_instrs
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import max_pipeline_ab_stages
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        accumulator = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, col])
        out[row, col] = accumulator.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _biased_gemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        accumulator = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, col])
        out[row, col] = (accumulator + bias[col]).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fanout_gemm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        accumulator = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, col])
        out[row, col] = accumulator.to(out.dtype)
        second[row, col] = torch.relu(accumulator).to(second.dtype)
    return out, second


def _args(
    shape: tuple[int, int, int] = (512, 1024, 512),
    dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k, n = shape
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty((k, n), dtype=dtype),
        torch.empty((m, n), dtype=output_dtype or dtype),
    )


def _config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 64],
            "tcgen05_cta_group": "two",
            "tcgen05_cluster_m": 2,
            "pid_type": "persistent_blocked",
            "tcgen05_ab_stages": 13,
        }
        | overrides
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_automatic_m128_retains_legacy_one_cta(dtype: torch.dtype) -> None:
    assert not _tcgen05_use_2cta_instrs(bm=128, cluster_m=2, input_dtype=dtype)
    assert _tcgen05_use_2cta_instrs(
        bm=128, cluster_m=2, input_dtype=dtype, cta_group="two"
    )
    assert _tcgen05_use_2cta_instrs(bm=256, cluster_m=2, input_dtype=dtype)
    assert _tcgen05_use_2cta_instrs(
        bm=128, cluster_m=2, input_dtype=torch.float8_e4m3fn
    )


@pytest.mark.parametrize("output_bytes", (2, 4))
def test_actual_output_ring_leaves_thirteen_ab_stages(output_bytes: int) -> None:
    facts = Tcgen05PipelineSmemFacts(2, output_bytes, 232448)
    parameters = {"bm": 128, "bn": 128, "bk": 64, "c_stages": 2, "acc_stages": 2}
    maximum = 13 if output_bytes == 2 else 12
    assert max_pipeline_ab_stages(facts, **parameters) == maximum
    assert pipeline_smem_bytes(facts, ab_stages=maximum, **parameters) <= 232448
    assert pipeline_smem_bytes(facts, ab_stages=maximum + 1, **parameters) > 232448
    parameters["c_stages"] = 4
    assert max_pipeline_ab_stages(facts, **parameters) == (
        12 if output_bytes == 2 else 10
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("shape", ((512, 1024, 512), (73, 136, 72)))
def test_codegen_owns_matching_device_and_wrapper_pair(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    args = _args(shape, dtype)
    bound = _gemm._bind_isolated(args)
    assert bound.config_spec._cute_tcgen05_config.pipeline_smem_facts is not None
    code = bound.to_code(_config())
    assert "CtaGroup.TWO" in code
    assert "num_stages=13" in code
    assert "'use_2cta_instrs': True" in code
    assert "compute_epilogue_tile_shape((64, 128), True" in code
    assert "CtaGroup.ONE" not in code


def test_output_dtype_is_part_of_the_budget_proof() -> None:
    args = _args(output_dtype=torch.float32)
    bound = _gemm._bind_isolated(args)
    facts = bound.config_spec._cute_tcgen05_config.pipeline_smem_facts
    assert facts is not None and facts.output_dtype_bytes == 4
    code = bound.to_code(_config(tcgen05_ab_stages=12))
    assert "num_stages=12" in code
    assert "tcgen05_c_layout, cutlass.Float32)" in code


def test_existing_role_lifecycle_keeps_its_pipeline_state_helpers() -> None:
    from cutlass.pipeline import PipelineState

    from helion._compiler.cute.tcgen05_pipeline import HelionPipelineState
    from helion._compiler.cute.tcgen05_pipeline import make_pipeline_state

    args = _args()
    code = _gemm._bind_isolated(args).to_code(
        helion.Config(
            block_sizes=[128, 128, 64],
            pid_type="flat",
            tcgen05_strategy="pure_matmul_role_lifecycle",
        )
    )
    assert "_helion_tcgen05_pipeline.make_pipeline_state(" in code
    assert issubclass(HelionPipelineState, PipelineState)
    assert callable(make_pipeline_state)


@pytest.mark.parametrize("paired", (False, True))
def test_ab_full_and_empty_barriers_fit_the_allocation(paired: bool) -> None:
    args = _args()
    bound = _gemm._bind_isolated(args)
    config = (
        _config()
        if paired
        else _config(
            tcgen05_cta_group="auto",
            tcgen05_cluster_m=1,
            tcgen05_ab_stages=2,
        )
    )
    module = ast.parse(bound.to_code(config))
    allocations = {
        statement.targets[0].id: statement.value
        for statement in ast.walk(module)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }
    matched = 0
    for call in ast.walk(module):
        if (
            not isinstance(call, ast.Call)
            or ast.unparse(call.func) != "cutlass.pipeline.PipelineTmaUmma.create"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        stages = ast.literal_eval(keywords["num_stages"])
        storage = keywords["barrier_storage"]
        assert isinstance(storage, ast.Name)
        allocation = allocations[storage.id]
        assert isinstance(allocation, ast.Call)
        assert ast.unparse(allocation.func) == "cute.arch.alloc_smem"
        assert ast.unparse(allocation.args[0]) == "cutlass.Int64"
        count = allocation.args[1]
        assert isinstance(count, ast.Call)
        allocated_bytes = ast.literal_eval(count.args[0]) * 8
        # PipelineTmaUmma.create initializes [0,N) full barriers and
        # [N,2N) empty barriers, with the last 8-byte cell at (2N-1)*8.
        highest_initialized_offset = (2 * stages - 1) * 8
        assert highest_initialized_offset + 8 <= allocated_bytes
        matched += 1
    assert matched == 1


@pytest.mark.parametrize(
    "invalid", ("overfull_ab", "overfull_c", "cluster", "layout", "clc")
)
def test_rejects_unmodelled_or_overfull_configs(invalid: str) -> None:
    args = _args()
    bound = _gemm._bind_isolated(args)
    overrides = {
        "overfull_ab": {"tcgen05_ab_stages": 14},
        "overfull_c": {"tcgen05_c_stages": 4},
        "cluster": {"tcgen05_cluster_n": 2},
        "layout": {"tcgen05_layout_strategy": "explicit_epi_tile"},
        "clc": {"tcgen05_persistence_model": "clc_persistent"},
    }[invalid]
    with pytest.raises(helion.exc.InvalidConfig, match="shared memory|two-CTA"):
        bound.to_code(_config(**overrides))


@pytest.mark.parametrize("other", ("bias", "fanout", "column_major", "no_capacity"))
def test_unknown_allocations_keep_conservative_profile(other: str) -> None:
    a, b, out = args = _args()
    if other == "bias":
        bias = torch.empty(b.shape[1], dtype=a.dtype)
        bound = _biased_gemm._bind_isolated((a, b, bias, out))
    elif other == "fanout":
        second = torch.empty_like(out)
        bound = _fanout_gemm._bind_isolated((a, b, out, second))
    elif other == "column_major":
        out = out.t().contiguous().t()
        bound = _gemm._bind_isolated((a, b, out))
    else:
        with patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=0
        ):
            bound = _gemm._bind_isolated(args)
    config = bound.config_spec._cute_tcgen05_config
    assert config.pipeline_smem_facts is None
    assert not config._paired_pipeline_seed_configs()
    assert (
        config.optional_fragments(for_search=True)["tcgen05_cta_group"].default()
        == "auto"
    )
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


def test_cold_search_can_encode_and_repair_deep_geometry_seeds() -> None:
    args = _args()
    bound = _gemm._bind_isolated(args)
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    with bound.env, host:
        seeds = [
            seed
            for seed in spec.autotune_seed_configs()
            if seed.config.get("tcgen05_cta_group") == "two"
        ]
        assert seeds
        generation = ConfigGeneration(spec)
        selected = []
        for seed in seeds:
            normalized = bound._normalized_config_copy(
                helion.Config.from_dict(spec.default_config().config | seed.config)
            )
            repaired = generation.unflatten(generation.flatten(normalized))
            assert repaired == normalized
            if repaired.block_sizes == [128, 128, 64]:
                selected.append(repaired.config["tcgen05_ab_stages"])
        assert selected == [2, 13]
        requested = (
            spec.default_config().config
            | _config(tcgen05_ab_stages=16, tcgen05_c_stages=4).config
        )
        spec._cute_tcgen05_config.fix_search_config(requested)
        assert requested["tcgen05_ab_stages"] == 12
        assert requested["tcgen05_cta_group"] == "two"
        assert spec.default_config().config.get("tcgen05_cta_group", "auto") == "auto"
