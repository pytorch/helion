"""CPU allocation and admission checks for explicit paired FP8 pipelines."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import extra_store_mm
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import vector_scaled_mm

import helion
from helion._compiler.cute.aux_tensor import _tcgen05_aux_detector_mma_nodes
from helion._compiler.cute.cute_mma import analyze_cute_mma_node
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import analyze_pipeline_smem_facts
from helion._compiler.cute.pipeline_smem import max_pipeline_ab_stages
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.tcgen05_constants import Tcgen05RowvecAuxRow
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _args() -> tuple[torch.Tensor, ...]:
    return scaled_args(256, 2048, 4096)


def _config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 128],
            "tcgen05_cta_group": "two",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "pid_type": "persistent_blocked",
            "tcgen05_ab_stages": 13,
            "tcgen05_c_stages": 2,
            "tcgen05_acc_stages": 2,
        }
        | overrides
    )


@pytest.mark.parametrize(
    ("output_bytes", "c_stages", "limit"), ((2, 2, 13), (2, 4, 12), (4, 2, 12))
)
def test_fp8_budget_boundary(output_bytes: int, c_stages: int, limit: int) -> None:
    facts = Tcgen05PipelineSmemFacts(1, output_bytes, 232448)
    params = {"bm": 128, "bn": 128, "bk": 128, "c_stages": c_stages, "acc_stages": 2}
    assert max_pipeline_ab_stages(facts, **params) == limit
    used = pipeline_smem_bytes(facts, ab_stages=limit, **params)
    assert used <= facts.capacity_bytes
    assert (
        pipeline_smem_bytes(facts, ab_stages=limit + 1, **params) > facts.capacity_bytes
    )
    assert (
        max_pipeline_ab_stages(facts._replace(capacity_bytes=used - 1), **params)
        == limit - 1
    )


@pytest.mark.parametrize(
    ("bm", "placement", "store", "row_bytes"),
    (
        (128, "pre_acc_wait", "normal", 0),
        (256, "pre_acc_wait", "normal", 2048),
        (256, "post_acc_wait", "normal", 0),
        (256, "pre_acc_wait", "direct", 0),
    ),
)
def test_scale_row_cost_matches_epilogue_protocol(
    bm: int, placement: str, store: str, row_bytes: int
) -> None:
    plain = Tcgen05PipelineSmemFacts(1, 2, 232448)
    scaled = plain._replace(
        rows=(Tcgen05RowvecAuxRow(4, False),), has_auxiliary_loads=True
    )
    params = {
        "bm": bm,
        "bn": 128,
        "bk": 128,
        "ab_stages": 8,
        "c_stages": 2,
        "acc_stages": 2,
        "aux_load_placement": placement,
        "store_mode": store,
    }
    assert (
        pipeline_smem_bytes(scaled, **params) - pipeline_smem_bytes(plain, **params)
        == row_bytes
    )


@pytest.mark.parametrize("placement", ("post_acc_wait", "pre_acc_wait"))
def test_fp8_ab13_is_native_and_survives_search(placement: str) -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    facts = state.pipeline_smem_facts
    assert facts is not None and facts.rows == (Tcgen05RowvecAuxRow(4, False),)
    config = _config(tcgen05_aux_load_placement=placement)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat, canonical = generation.canonicalize_flat(generation.flatten(config))
        assert generation.unflatten(flat) == canonical
        assert canonical.block_sizes == [128, 128, 128]
        assert canonical.config["tcgen05_cta_group"] == "two"
        assert canonical.config["tcgen05_ab_stages"] == 13
        seeds = state._paired_pipeline_seed_configs()
        matches = [
            seed
            for seed in seeds
            if seed.block_sizes == [128, 128, 128]
            and seed.config["tcgen05_ab_stages"] == 13
        ]
        assert len(matches) == 2
        for seed in matches:
            _flat, normalized = generation.canonicalize_flat(generation.flatten(seed))
            assert normalized.config["tcgen05_cta_group"] == "two"
            assert normalized.config["tcgen05_ab_stages"] == 13
    source = bound.to_code(config)
    assert "CtaGroup.TWO" in source
    assert "num_stages=13" in source
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source
    assert "aux_smem" not in source
    module = ast.parse(source)
    stores = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cutlass.pipeline.PipelineTmaStore.create"
    ]
    assert len(stores) == 1


@pytest.mark.parametrize(
    "overrides",
    (
        {"tcgen05_ab_stages": 14},
        {"tcgen05_c_stages": 4},
        {"tcgen05_cluster_n": 4},
        {"tcgen05_warp_spec_scheduler_warps": 1},
        {"tcgen05_warp_spec_c_input_warps": 1},
        {"tcgen05_warp_spec_store_warps": 1},
        {"tcgen05_aux_load_mode": "tma"},
        {"tcgen05_persistence_model": "clc_persistent"},
        {"block_sizes": [256, 128, 128], "tcgen05_ab_stages": 9},
    ),
)
def test_fp8_rejects_overfull_and_unmodelled_protocols(
    overrides: dict[str, object],
) -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    with pytest.raises(
        helion.exc.InvalidConfig,
        match=r"cta_group='two'|per-CTA shared memory|tcgen05_cluster_n",
    ):
        bound.to_code(_config(**overrides))


def test_fp8_pair_cluster_n2_is_modelled() -> None:
    # The 2x2 cluster of fp8 CTA pairs keeps the per-CTA SMEM model.
    bound = vector_scaled_mm._bind_isolated(_args())
    with bound.env:
        paired = bound.config_spec.normalized_config(
            _config(tcgen05_cluster_n=2, tcgen05_ab_stages=2)
        )
    assert paired.config["tcgen05_cta_group"] == "two"
    assert paired.config["tcgen05_cluster_n"] == 2
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in bound.to_code(paired)


@pytest.mark.parametrize(
    "kind", ("scale_dtype", "scale_stride", "unaligned", "extra_store")
)
def test_unproved_inputs_cannot_bypass_old_budget(kind: str) -> None:
    args = list(_args())
    kernel = vector_scaled_mm
    if kind == "scale_dtype":
        args[3] = args[3].to(torch.bfloat16)
    elif kind == "scale_stride":
        args[3] = torch.empty(4096, dtype=torch.float32)[::2]
    elif kind == "unaligned":
        args[0] = torch.empty(256 * 4096 + 1, dtype=args[0].dtype)[1:].view(256, 4096)
    else:
        kernel = extra_store_mm
        args = args[:2]
    bound = kernel._bind_isolated(tuple(args))
    state = bound.config_spec._cute_tcgen05_config
    assert not state.paired_pipeline_search_enabled()
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


def test_stage_repair_keeps_c_ring_and_generic_fp8_limit() -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    assert state._get_dtype_ab_stages_hard_cap(1) == 12
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        _flat, canonical = generation.canonicalize_flat(
            generation.flatten(_config(tcgen05_ab_stages=16, tcgen05_c_stages=4))
        )
    assert canonical.config["tcgen05_cta_group"] == "two"
    assert canonical.config["tcgen05_c_stages"] == 4
    assert canonical.config["tcgen05_ab_stages"] == 12


def test_memory_facts_do_not_restore_unrelated_pid_restriction() -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    facts = state.pipeline_smem_facts
    assert facts is not None
    bound.config_spec.disallow_pid_type("persistent_blocked", reason="unrelated proof")
    state.register_pipeline_smem_facts(facts)
    assert "persistent_blocked" not in state.allowed_pid_types
    assert not state.paired_pipeline_search_enabled()


@pytest.mark.parametrize(
    "shape", ((256, 1984, 4096), (192, 2048, 4096), (256, 2048, 64))
)
def test_new_budget_rejects_output_edges_and_partial_first_k_tile(
    shape: tuple[int, int, int],
) -> None:
    bound = vector_scaled_mm._bind_isolated(scaled_args(*shape))
    assert (
        bound.config_spec._cute_tcgen05_config._paired_pipeline_parameters(
            _config().config
        )
        is None
    )
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


@pytest.mark.parametrize("rhs_k_major", (True, False))
def test_fp8_paired_k_tail_budget_and_search(rhs_k_major: bool) -> None:
    a = torch.empty((128, 2880), dtype=torch.float8_e4m3fn)
    b = torch.empty((5120, 2880), dtype=torch.float8_e4m3fn).T
    if not rhs_k_major:
        b = b.contiguous()
    bound = vector_scaled_mm._bind_isolated(
        (
            a,
            b,
            torch.empty(128, dtype=torch.float32),
            torch.empty(5120, dtype=torch.float32),
        )
    )
    state = bound.config_spec._cute_tcgen05_config
    if not rhs_k_major:
        assert state._paired_pipeline_parameters(_config().config) is None
        with pytest.raises(helion.exc.InvalidConfig):
            bound.to_code(_config())
        return
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        _flat, canonical = generation.canonicalize_flat(generation.flatten(_config()))
        assert canonical.block_sizes == [128, 128, 128]
        assert canonical.config["tcgen05_cta_group"] == "two"
        assert canonical.config["tcgen05_ab_stages"] == 13
        assert any(
            seed.block_sizes == [128, 128, 128]
            and seed.config["tcgen05_ab_stages"] == 13
            for seed in state._paired_pipeline_seed_configs()
        )
    source = bound.to_code(canonical)
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source
    assert "CtaGroup.TWO" in source
    assert "num_stages=13" in source


def test_other_pipeline_analyzer_callers_do_not_inherit_fp8_admission() -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    ir = bound.host_function.device_ir
    node = next(iter(_tcgen05_aux_detector_mma_nodes(ir.graphs)))
    with bound.env, bound.host_function:
        candidate = analyze_cute_mma_node(node, device_ir=ir)
    assert candidate is not None
    assert (
        analyze_pipeline_smem_facts(candidate, node, ir.graphs, capacity_bytes=232448)
        is None
    )
    assert (
        analyze_pipeline_smem_facts(
            candidate, node, ir.graphs, capacity_bytes=232448, allow_fp8_broadcasts=True
        )
        is not None
    )
