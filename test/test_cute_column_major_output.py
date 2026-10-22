"""CPU contracts for ordinary GEMMs writing a column-major output view."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import atomic_output_mm
from test._cute_dense_tcgen05 import column_major_args
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import gb300_target
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import scaled_mm_column_major
from test._cute_dense_tcgen05 import wrapper_plans

import helion
from helion._compiler.cute import cute_mma
from helion._compiler.cute.dense_gemm import prove_dense_gemm_facts
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import max_pipeline_ab_stages
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.tcgen05_schedule_facts import analyze_native_gemm
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])

# These are physical M/N/K tiles of the frontend's B.T @ A.T expression.
# They are not compiler geometry requests or logical-axis projections.
_TILES = ((256, 64, 128), (128, 64, 256))
_DEPTHS = (10, 8)


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _config(index: int = 1, **changes: object) -> helion.Config:
    values = {
        "block_sizes": list(_TILES[index]),
        "tcgen05_cta_group": "two",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": _DEPTHS[index],
        "tcgen05_c_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_aux_load_placement": "post_acc_wait",
    }
    values.update(changes)
    return helion.Config.from_dict(values)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _borrowed_column_major_output(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out_t = out.T
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out_t[rows, cols] = acc.to(torch.bfloat16)
    return out


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("acc_stages", (1, 2))
def test_physical_operands_and_column_major_output_roundtrip(
    index: int, mode: int, acc_stages: int
) -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args(mode=mode))
    state = bound.config_spec._cute_tcgen05_config
    facts = state.dense_gemm_facts
    assert facts is not None
    assert facts.shape == (2048, 256, 4096)
    assert facts.output_column_major
    assert state.pipeline_smem_facts is not None
    assert state.pipeline_smem_facts.output_column_major
    request = _config(index, tcgen05_acc_stages=acc_stages)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat, canonical = generation.canonicalize_flat(generation.flatten(request))
        assert generation.unflatten(flat) == canonical
        assert generation.canonicalize_flat(flat) == (flat, canonical)
    assert canonical.block_sizes == list(_TILES[index])
    assert canonical.config["tcgen05_ab_stages"] == _DEPTHS[index]
    assert canonical.config["tcgen05_acc_stages"] == acc_stages
    source = bound.to_code(canonical)
    ab, output = wrapper_plans(source)
    assert (ab["lhs_idx"], ab["rhs_idx"]) == (0, 1)
    assert tuple(ab[key] for key in ("bm", "bn", "bk")) == _TILES[index]
    assert ab["ab_stage_count"] == _DEPTHS[index]
    assert output["c_stage_count"] == 2
    assert output["d_column_major"] is True
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source
    assert "CtaGroup.TWO" in source
    assert "LayoutEnum.COL_MAJOR" in source
    assert "epilogue_tmem_copy_and_partition(" in source
    assert "epilogue_smem_copy_and_partition(" in source
    # Descriptor construction must preserve these already-physical operands.
    wrapper: list[str] = []
    for descriptor in (ab, output):
        _append_cute_wrapper_plan(wrapper, [], descriptor)
    assert "CtaGroup.TWO" in "\n".join(wrapper)


@pytest.mark.parametrize("step", (1, 3))
def test_column_major_view_preserves_scale_addresses_and_multiplication_order(
    step: int,
) -> None:
    """Distinct nonuniform scales expose swapped axes and reassociation."""
    m, n = 128, 512
    generator = torch.Generator().manual_seed(19)
    acc = torch.randn((m, n), generator=generator, dtype=torch.float32)
    scale_a = torch.rand(m * step, generator=generator)[::step]
    scale_b = torch.rand(n * step, generator=generator)[::step]
    physical_a = scale_a[None, :].expand(n, m)
    physical_b = scale_b[:, None].expand(n, m)
    physical_result = (acc.T * physical_a) * physical_b
    expected = (acc * scale_a[:, None]) * scale_b[None, :]
    assert torch.equal(physical_result.T, expected)
    assert not torch.equal(physical_result, acc.T * (physical_a * physical_b))
    output = torch.empty((m, n), dtype=torch.float32)
    output_t = output.T
    assert output_t.stride() == (1, n)
    assert output_t.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    for rows in (slice(0, 256), slice(256, 512)):
        for cols in (slice(0, 64), slice(64, 128)):
            output_t[rows, cols] = physical_result[rows, cols]
    assert output.is_contiguous()
    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    "index,used,overflow", ((0, 223424, 243920), (1, 215200, 239792))
)
@pytest.mark.parametrize("acc_stages", (1, 2))
def test_column_major_complete_allocation_boundary(
    index: int, used: int, overflow: int, acc_stages: int
) -> None:
    facts = Tcgen05PipelineSmemFacts(1, 2, 232448, output_column_major=True)
    bm, bn, bk = _TILES[index]
    parameters = {"bm": bm, "bn": bn, "bk": bk, "c_stages": 2, "acc_stages": acc_stages}
    assert max_pipeline_ab_stages(facts, **parameters) == _DEPTHS[index]
    assert pipeline_smem_bytes(facts, ab_stages=_DEPTHS[index], **parameters) == (
        used - 16 * (2 - acc_stages)
    )
    assert pipeline_smem_bytes(facts, ab_stages=_DEPTHS[index] + 1, **parameters) == (
        overflow - 16 * (2 - acc_stages)
    )
    exact = pipeline_smem_bytes(facts, ab_stages=_DEPTHS[index], **parameters)
    assert (
        max_pipeline_ab_stages(facts._replace(capacity_bytes=exact), **parameters)
        == _DEPTHS[index]
    )
    assert (
        max_pipeline_ab_stages(facts._replace(capacity_bytes=exact - 1), **parameters)
        == _DEPTHS[index] - 1
    )


@pytest.mark.parametrize("index", (0, 1))
def test_overfull_column_major_pipeline_rejected_and_search_repairs(index: int) -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args())
    state = bound.config_spec._cute_tcgen05_config
    request = _config(index, tcgen05_ab_stages=_DEPTHS[index] + 1)
    with pytest.raises(helion.exc.InvalidConfig, match="shared.memory|SMEM"):
        bound.to_code(request)
    repaired = dict(request.config)
    with bound.env:
        state.fix_search_config(repaired)
    assert repaired["block_sizes"] == list(_TILES[index])
    assert repaired["tcgen05_cta_group"] == "two"
    assert repaired["tcgen05_ab_stages"] == _DEPTHS[index]


@pytest.mark.parametrize("kind", ("dtype", "stride"))
def test_column_major_scale_proof_rejects_unmodelled_inputs(kind: str) -> None:
    args = list(column_major_args())
    if kind == "dtype":
        args[2] = args[2].to(torch.bfloat16)
    else:
        args[2] = torch.empty(512, dtype=torch.float32)[::2]
    bound = scaled_mm_column_major._bind_isolated(tuple(args))
    assert bound.config_spec._cute_tcgen05_config.pipeline_smem_facts is None
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


@pytest.mark.parametrize(
    "changes",
    (
        {"tcgen05_cluster_n": 2},
        {"tcgen05_persistence_model": "clc_persistent"},
        {"tcgen05_warp_spec_scheduler_warps": 1},
        {"tcgen05_warp_spec_c_input_warps": 1},
        {"tcgen05_warp_spec_store_warps": 1},
        {"tcgen05_aux_load_mode": "tma"},
    ),
)
def test_column_major_pipeline_rejects_unpriced_protocols(
    changes: dict[str, object],
) -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args())
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config(**changes))


@pytest.mark.parametrize(
    "shape", ((255, 2048, 4096), (256, 2047, 4096), (256, 2048, 64))
)
def test_column_major_budget_rejects_output_edges_and_partial_first_k_tile(
    shape: tuple[int, int, int],
) -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args(*shape))
    state = bound.config_spec._cute_tcgen05_config
    assert state._paired_pipeline_parameters(_config().config) is None
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


def test_dense_allocation_proof_requires_fresh_output_and_one_function() -> None:
    args = column_major_args()
    borrowed = _borrowed_column_major_output._bind_isolated(
        (*args[:2], torch.empty((256, 2048), dtype=torch.bfloat16))
    )
    assert borrowed.config_spec._cute_tcgen05_config.dense_gemm_facts is None
    bound = scaled_mm_column_major._bind_isolated(args)
    assert bound.config_spec._cute_tcgen05_config.dense_gemm_facts is not None
    with patch.object(bound.env, "cute_fission_plan", object()):
        gemm = analyze_native_gemm(bound.env, bound.host_function.device_ir)
        assert prove_dense_gemm_facts(bound.env, gemm) is None
    atomic = atomic_output_mm._bind_isolated(
        (*scaled_args()[:2], torch.empty((1,), dtype=torch.int32))
    )
    assert atomic.config_spec._cute_tcgen05_config.dense_gemm_facts is None


def test_generic_paired_seed_keeps_independent_pid_vetoes() -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args())
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state._paired_pipeline_seed_configs()
    for pid in ("persistent_interleaved", "persistent_blocked"):
        spec.disallow_pid_type(pid, "independent ownership restriction")
    causes = {
        pid: set(reasons) for pid, reasons in spec.disallowed_pid_type_causes.items()
    }
    assert not state._paired_pipeline_seed_configs()
    assert spec.disallowed_pid_type_causes == causes


def test_physical_block_axes_survive_config_encoding_reordering() -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args())
    blocks = bound.config_spec.block_sizes
    original = list(blocks)
    for index, old_index in enumerate((2, 0, 1)):
        blocks[index] = original[old_index]
    state = bound.config_spec._cute_tcgen05_config
    request = _config(block_sizes=[256, 128, 64])
    assert state._paired_pipeline_parameters(request.config)[:3] == (128, 64, 256)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat, canonical = generation.canonicalize_flat(generation.flatten(request))
        assert canonical.block_sizes == request.block_sizes
        assert generation.canonicalize_flat(flat) == (flat, canonical)


@pytest.mark.parametrize("acc_stages", (1, 2))
def test_column_major_config_observes_live_k_bounds(acc_stages: int) -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args())
    state = bound.config_spec._cute_tcgen05_config
    request = _config(tcgen05_acc_stages=acc_stages)
    assert state._paired_pipeline_parameters(request.config) is not None
    k_block = bound.config_spec.block_sizes.block_id_lookup(state.matmul_block_ids[2])
    k_block.update_max(128)
    assert state._paired_pipeline_parameters(request.config) is None
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(request)


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("mode", (0, 3))
def test_early_mma_selection_receives_already_physical_tiles(
    index: int, mode: int
) -> None:
    calls: list[tuple[str, tuple[int, int, int], str]] = []
    choose = cute_mma._choose_mma_impl

    def record(*args, **kwargs):
        result = choose(*args, **kwargs)
        calls.append(
            (
                sys._getframe(1).f_code.co_name,
                (kwargs["bm"], kwargs["bn"], kwargs["bk"]),
                result,
            )
        )
        return result

    with gb300_target():
        bound = scaled_mm_column_major._bind_isolated(column_major_args(mode=mode))
        with patch.object(cute_mma, "_choose_mma_impl", record):
            source = bound.to_code(_config(index))
    callers = {caller for caller, _tile, _result in calls}
    assert {
        "_detect_specialized_mma_loop",
        "_analyzed_specialized_mma_plan",
        "_emit_mma_pipeline",
    } <= callers
    assert all(
        tile == _TILES[index] and result == "tcgen05" for _caller, tile, result in calls
    ), calls
    assert "PipelineTmaStore.create" in source


@pytest.mark.parametrize("index", (0, 1))
def test_column_major_acc1_pipeline_releases_after_last_tmem_load(index: int) -> None:
    bound = scaled_mm_column_major._bind_isolated(column_major_args())
    source = bound.to_code(_config(index, tcgen05_acc_stages=1))
    assert "PipelineUmmaAsync.create(num_stages=1," in source
    assert (
        "tcgen05_acc_pipeline_barriers = cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(2))"
        in source
    )
    assert source.count("tcgen05_acc_producer_state.advance()") == 1
    assert source.count("tcgen05_acc_consumer_state.advance()") == 1
    assert source.index("fence_view_async_tmem_load()") < source.index(
        "tcgen05_acc_pipeline.consumer_release("
    )
    assert "if _tcgen05_subtile == tcgen05_subtile_count - 1:" in source
