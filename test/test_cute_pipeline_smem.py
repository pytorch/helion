"""CPU proofs for explicit CTA groups and epilogue-aware pipeline depths.

Also the one SMEM pricer: ``SmemArena`` is the two byte models
(``pipeline_smem_bytes`` / ``one_cta_pipeline_smem_bytes``) and their stage
limits over the whole modeled domain, and ``max_pipeline_ab_stages`` with
``cluster_m=1`` is the one-CTA wrapper's historical result.
"""

from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import target_facts
from helion._compiler.cute.aux_tensor import _tcgen05_aux_detector_mma_nodes
from helion._compiler.cute.cute_mma import _tcgen05_use_2cta_instrs
from helion._compiler.cute.cute_mma import analyze_cute_mma_node
from helion._compiler.cute.one_cta_pipeline import max_one_cta_pipeline_ab_stages
from helion._compiler.cute.one_cta_pipeline import one_cta_pipeline_smem_bytes
from helion._compiler.cute.pipeline_smem import SmemArena
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import analyze_pipeline_smem_facts
from helion._compiler.cute.pipeline_smem import max_pipeline_ab_stages
from helion._compiler.cute.pipeline_smem import one_cta_pipeline_tile_modeled
from helion._compiler.cute.pipeline_smem import paired_pipeline_tile_modeled
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import smem
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_constants import Tcgen05RowvecAuxRow
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bias_mm_into
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with cpu_target():
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
def _masked_gemm(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        accumulator = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[row, inner], b[inner, col])
        out[row, col] = accumulator.to(out.dtype) * (mask[row, col] > 0)
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
    # The clustered (paired) kernel builds the ring through the runtime's
    # no-init constructor, the one-CTA kernel through the DSL constructor.
    constructor = (
        "_helion_tc_pro.make_tma_umma_pipeline"
        if paired
        else "cutlass.pipeline.PipelineTmaUmma.create"
    )
    matched = 0
    for call in ast.walk(module):
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != constructor:
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
        "cluster": {"tcgen05_cluster_n": 4},
        "layout": {"tcgen05_layout_strategy": "explicit_epi_tile"},
        "clc": {"tcgen05_persistence_model": "clc_persistent"},
    }[invalid]
    with pytest.raises(helion.exc.InvalidConfig, match="shared memory|two-CTA|cluster"):
        bound.to_code(_config(**overrides))


def test_pair_cluster_n2_is_a_modelled_paired_configuration() -> None:
    # The 2x2 cluster of CTA pairs (A multicast across the N neighbours) only
    # shapes the mainloop and the scheduler; the per-CTA SMEM model is the
    # same, so the paired family admits it at the same ring depths.
    bound = _gemm._bind_isolated(_args())
    with bound.env:
        spec = bound.config_spec
        limit = smem._paired_pipeline_stage_limit(
            spec._cute_tcgen05_config,
            spec.normalized_config(_config(tcgen05_ab_stages=2)).config,
        )
        paired = spec.normalized_config(
            _config(tcgen05_cluster_n=2, tcgen05_ab_stages=2)
        )
    assert limit is not None and limit >= 2
    assert paired.config["tcgen05_cta_group"] == "two"
    assert paired.config["tcgen05_cluster_n"] == 2
    source = bound.to_code(paired)
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in source
    assert "CtaGroup.TWO" in source


@pytest.mark.parametrize("other", ("fanout", "no_capacity"))
def test_unknown_allocations_keep_conservative_profile(other: str) -> None:
    a, b, out = args = _args()
    if other == "fanout":
        second = torch.empty_like(out)
        bound = _fanout_gemm._bind_isolated((a, b, out, second))
    else:
        with patch.object(target_facts, "per_cta_smem_capacity_bytes", return_value=0):
            bound = _gemm._bind_isolated(args)
    config = bound.config_spec._cute_tcgen05_config
    assert config.pipeline_smem_facts is None
    assert not seeds._paired_pipeline_seed_configs(config)
    assert (
        surface.optional_fragments(config, for_search=True)[
            "tcgen05_cta_group"
        ].default()
        == "auto"
    )
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


@pytest.mark.parametrize("placement", ("post_acc_wait", "pre_acc_wait"))
def test_bias_row_epilogue_keeps_the_paired_pipeline(placement: str) -> None:
    """A 16-bit bias row (``bias[col]``, an N-broadcast) loads per subtile
    under ``post_acc_wait`` and stages through one row of shared memory under
    ``pre_acc_wait``, which the allocation proof charges: the paired pipeline,
    and with it the autotune-off default tile, reach Inductor's addmm
    epilogues (bf16 16384x30528x768 + bias defaulted to the flat 128x16
    kernel without the proof)."""
    a, b, out = _args()
    bias = torch.empty(b.shape[1], dtype=a.dtype)
    bound = _biased_gemm._bind_isolated((a, b, bias, out))
    config = bound.config_spec._cute_tcgen05_config
    facts = config.pipeline_smem_facts
    assert facts is not None and facts.has_auxiliary_loads
    # the row only the FP32 root ``acc + bias`` reads stages promoted to FP32
    assert facts.rows == (Tcgen05RowvecAuxRow(itemsize=2, promoted=True),)
    assert surface.paired_pipeline_search_enabled(config)
    stages = {
        seed.config["tcgen05_aux_load_placement"]: seed.config["tcgen05_ab_stages"]
        for seed in seeds._paired_pipeline_seed_configs(config)
        if seed.config["block_sizes"] == [256, 256, 64]
        and seed.config["tcgen05_ab_stages"] > 2
    }
    # ``pre_acc_wait`` is priced at the (128, 64) C subtile plus the row:
    # codegen takes the (128, 32) subtile for a promoted row only on facts
    # the model does not see (``tcgen05_promoted_rowvec_epilogue``), so the
    # model stays one stage conservative rather than over-admit a ring
    assert stages == {"post_acc_wait": 6, "pre_acc_wait": 5}
    code = bound.to_code(
        _config(
            block_sizes=[256, 256, 64],
            tcgen05_ab_stages=stages[placement],
            tcgen05_aux_load_placement=placement,
        )
    )
    assert "CtaGroup.TWO" in code


def test_inductor_cast_bias_row_stages_promoted() -> None:
    """Inductor's addmm epilogue converts the row explicitly
    (``bias[col].to(torch.float32) + acc``): the row reaches the FP32 root
    unrounded like the bare load, so ``pre_acc_wait`` stages it promoted to
    FP32 (the conversion of the staged value is the identity) instead of
    loading it per subtile."""
    a, b, out = _args()
    bias = torch.empty(b.shape[1], dtype=a.dtype)
    bound = bias_mm_into._bind_isolated((a, b, bias, out))
    facts = bound.config_spec._cute_tcgen05_config.pipeline_smem_facts
    assert facts is not None
    assert facts.rows == (Tcgen05RowvecAuxRow(itemsize=2, promoted=True),)
    code = bound.to_code(
        _config(
            block_sizes=[256, 256, 64],
            tcgen05_ab_stages=5,
            tcgen05_aux_load_placement="pre_acc_wait",
        )
    )
    assert "alloc_smem(cutlass.Float32" in code
    assert "cute.autovec_copy(tcgen05_tTR_gAux_subtile" not in code


def test_bias_rows_record_their_width_and_columns_decline() -> None:
    """An FP32 bias row on a 16-bit GEMM stages unpromoted, four bytes a
    column; a broadcast column (``scale[row]``) has no modeled stage and
    still declines the proof; an explicit (1, N) row needs no stage."""
    a, b, out = _args()
    bias = torch.empty(b.shape[1], dtype=torch.float32)
    facts = _biased_gemm._bind_isolated(
        (a, b, bias, out)
    ).config_spec._cute_tcgen05_config.pipeline_smem_facts
    assert facts is not None
    assert facts.rows == (Tcgen05RowvecAuxRow(itemsize=4, promoted=False),)
    column = torch.empty(a.shape[0], 1, dtype=a.dtype).expand_as(out)
    bound = _masked_gemm._bind_isolated((a, b, column, out))
    assert bound.config_spec._cute_tcgen05_config.pipeline_smem_facts is None
    # an explicit (1, N) row is read in place: no stage to charge
    row = torch.empty(1, out.shape[1], dtype=a.dtype)
    facts = _masked_gemm._bind_isolated(
        (a, b, row, out)
    ).config_spec._cute_tcgen05_config.pipeline_smem_facts
    assert facts is not None and facts.has_auxiliary_loads
    assert facts.rows == () and facts.exact_aux_itemsizes == ()


def test_other_pipeline_analyzer_callers_refuse_bias_rows() -> None:
    """Only the single-root launch, whose stage limit prices the
    ``pre_acc_wait`` row stage, admits a 16-bit bias row: a materialized
    region's limits price no placement and would over-admit the ring."""
    a, b, out = _args()
    bias = torch.empty(b.shape[1], dtype=a.dtype)
    bound = _biased_gemm._bind_isolated((a, b, bias, out))
    ir = bound.host_function.device_ir
    node = next(iter(_tcgen05_aux_detector_mma_nodes(ir.graphs)))
    with bound.env, bound.host_function:
        candidate = analyze_cute_mma_node(node, device_ir=ir)
    assert candidate is not None
    assert (
        analyze_pipeline_smem_facts(candidate, node, ir.graphs, capacity_bytes=232448)
        is None
    )
    facts = analyze_pipeline_smem_facts(
        candidate, node, ir.graphs, capacity_bytes=232448, allow_broadcasts=True
    )
    assert facts is not None and len(facts.rows) == 1


def test_column_major_output_prices_its_actual_store_ring() -> None:
    import cutlass
    from cutlass import cute
    from cutlass._mlir import ir
    from cutlass.utils import blackwell_helpers
    from cutlass.utils.layout import LayoutEnum

    from helion._compiler.cute.tcgen05_constants import tcgen05_ab_smem_bytes_per_cta

    a, b, out = _args()
    out = out.t().contiguous().t()
    with forbid_cuda_work():
        bound = _gemm._bind_isolated((a, b, out))
        facts = bound.config_spec._cute_tcgen05_config.pipeline_smem_facts
        assert facts is not None and facts.output_column_major
        code = bound.to_code(_config())
        assert "LayoutEnum.COL_MAJOR" in code
        assert "PipelineTmaStore.create" in code
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                tile = blackwell_helpers.compute_epilogue_tile_shape(
                    (64, 128), True, LayoutEnum.COL_MAJOR, cutlass.BFloat16
                )
                layout = blackwell_helpers.make_smem_layout_epi(
                    cutlass.BFloat16, LayoutEnum.COL_MAJOR, tile, 2
                )
                actual_output_bytes = 2 * cute.cosize(layout.outer)
        ab_bytes = tcgen05_ab_smem_bytes_per_cta(
            bm=128, bn=128, bk=64, dtype_bytes=2, ab_stages=13, cluster_m=2
        )
        assert (
            pipeline_smem_bytes(
                facts, bm=128, bn=128, bk=64, ab_stages=13, c_stages=2, acc_stages=2
            )
            == ab_bytes + actual_output_bytes + 2048 + 16 * 13 + 16 * 2
        )


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
        repair.fix_search_config(spec._cute_tcgen05_config, requested)
        assert requested["tcgen05_ab_stages"] == 12
        assert requested["tcgen05_cta_group"] == "two"
        assert spec.default_config().config.get("tcgen05_cta_group", "auto") == "auto"


@pytest.mark.parametrize("placement", ("post_acc_wait", "pre_acc_wait"))
def test_exact_shape_auxiliary_epilogue_keeps_the_paired_pipeline(
    placement: str,
) -> None:
    """A 16-bit epilogue reading an exact-shape auxiliary tile (a mask, a
    residual) loads it per subtile into registers: no shared memory beyond
    the one-ring budget, so the complete allocation proof and the two-CTA
    paired pipeline (both auxiliary placements seeded) stay available.
    Broadcast rows (``bias[col]``) keep it too
    (``test_bias_row_epilogue_keeps_the_paired_pipeline``)."""
    a, b, out = _args()
    mask = torch.empty_like(out)
    bound = _masked_gemm._bind_isolated((a, b, mask, out))
    config = bound.config_spec._cute_tcgen05_config
    facts = config.pipeline_smem_facts
    assert facts is not None
    assert facts.has_auxiliary_loads and facts.rows == ()
    assert surface.paired_pipeline_search_enabled(config)
    seed_configs = seeds._paired_pipeline_seed_configs(config)
    assert {seed.config["tcgen05_aux_load_placement"] for seed in seed_configs} == {
        "post_acc_wait",
        "pre_acc_wait",
    }
    fragment = surface.optional_fragments(config, for_search=True)["tcgen05_cta_group"]
    assert fragment.choices == ("auto", "two")  # type: ignore[attr-defined]
    code = bound.to_code(_config(tcgen05_aux_load_placement=placement))
    assert "CtaGroup.TWO" in code
    assert " > 0.0" in code


_FACTS = [
    Tcgen05PipelineSmemFacts(
        in_bytes, out_bytes, 232448, exact_aux_itemsizes=(out_bytes,) * aux
    )
    for in_bytes, out_bytes, aux in itertools.product((1, 2), (2, 4), (0, 1))
]
_TILES = list(itertools.product((64, 128, 256), (16, 64, 128, 256), (32, 64, 128, 256)))
_RINGS = list(itertools.product((2, 4), (1, 2)))


@pytest.mark.parametrize("facts", _FACTS, ids=lambda f: f"{f[0]}B/{f[1]}B/aux{f[6]}")
def test_arena_is_the_two_byte_models(facts: Tcgen05PipelineSmemFacts) -> None:
    for (bm, bn, bk), (c_stages, acc_stages) in itertools.product(_TILES, _RINGS):
        kwargs = {"bm": bm, "bn": bn, "bk": bk, "c_stages": c_stages}
        for cluster_m in (1, 2):
            modeled = (
                one_cta_pipeline_tile_modeled(acc_stages=acc_stages, **kwargs)
                if cluster_m == 1
                else paired_pipeline_tile_modeled(acc_stages=acc_stages, **kwargs)
            )
            if not modeled:
                continue
            arena = SmemArena(facts, cluster_m)
            for ab_stages, aux_load_mode, epi_warp_groups in itertools.product(
                (1, 3, 8), ("simt", "epi_tma"), (1, 2)
            ):
                ring = {
                    "ab_stages": ab_stages,
                    "acc_stages": acc_stages,
                    "aux_load_mode": aux_load_mode,
                    "epi_warp_groups": epi_warp_groups,
                }
                if cluster_m == 1:
                    assert arena.bytes(**kwargs, **ring) == one_cta_pipeline_smem_bytes(
                        facts, **kwargs, **ring
                    )
                else:
                    assert arena.bytes(**kwargs, **ring) == pipeline_smem_bytes(
                        facts, **kwargs, **ring
                    )
            assert arena.deepest_ab_stages(
                acc_stages=acc_stages, **kwargs
            ) == max_pipeline_ab_stages(
                facts, acc_stages=acc_stages, cluster_m=cluster_m, **kwargs
            )


def test_one_cta_wrapper_is_the_merged_wrapper() -> None:
    facts = Tcgen05PipelineSmemFacts(1, 2, 232448)
    for (bm, bn, bk), (c_stages, acc_stages), max_stages in itertools.product(
        _TILES, _RINGS, (2, 5, 16)
    ):
        if not one_cta_pipeline_tile_modeled(
            bm=bm, bn=bn, bk=bk, c_stages=c_stages, acc_stages=acc_stages
        ):
            continue
        kwargs = {
            "bm": bm,
            "bn": bn,
            "bk": bk,
            "c_stages": c_stages,
            "acc_stages": acc_stages,
            "max_stages": max_stages,
        }
        assert max_one_cta_pipeline_ab_stages(
            facts, **kwargs
        ) == max_pipeline_ab_stages(facts, cluster_m=1, **kwargs)
