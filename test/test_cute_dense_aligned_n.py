"""Aligned physical widths preserve public carriers and native admission.

Normal (TMA store) and direct (R2G) epilogues share one physical plan.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma
from helion._compiler.cute.dense_aligned_n import ALIGNED_N_CHOICES
from helion._compiler.cute.dense_aligned_n import carrier_n
from helion._compiler.cute.dense_aligned_n import dense_aligned_n_defaults
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.strategies import tcgen05_two_cta_m128_epilogue_tile_expr
from helion._compiler.cute.tcgen05_config import facts
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_schedule_facts import analyze_native_gemm
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import aligned_n_bf16_config
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import column_scale_mm_into
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import output_alias_reading_mm
from helion._testing.cute_dense_tcgen05 import output_reading_mm
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion._testing.cute_dense_tcgen05 import scaled_args
from helion._testing.cute_dense_tcgen05 import scaled_mm
from helion._testing.cute_dense_tcgen05 import scaled_mm_into
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
ir = pytest.importorskip("cutlass._mlir").ir
h = pytest.importorskip("cutlass.utils.blackwell_helpers")
epi = pytest.importorskip("cutlass.utils.gemm.sm100")
LayoutEnum = pytest.importorskip("cutlass.utils.layout").LayoutEnum

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.fixture(autouse=True)
def _gb300() -> Iterator[None]:
    with gb300_target(), forbid_cuda_work():
        yield


def _args(mode=0, shape=(128, 5120, 2880)):
    return scaled_args(*shape, mode=mode)


def _smem_bytes(width, *, ab_stages, c_stages=2, acc_stages=2, bk=128):
    return pipeline_smem_bytes(
        Tcgen05PipelineSmemFacts(1, 2, 232448),
        bm=128,
        bn=width,
        bk=bk,
        ab_stages=ab_stages,
        c_stages=c_stages,
        acc_stages=acc_stages,
    )


def _d_tile_expr(width):
    return tcgen05_two_cta_m128_epilogue_tile_expr(
        128,
        width,
        "cutlass.BFloat16",
        c_layout="cutlass.utils.layout.LayoutEnum.ROW_MAJOR",
    )


def _config(**changes):
    return helion.Config(
        **(
            {
                **dense_aligned_n_defaults(),
                "tcgen05_tile_n": 80,
                "block_sizes": [128, 128, 128],
                "pid_type": "persistent_interleaved",
                "tcgen05_ab_stages": 16,
                "tcgen05_c_stages": 2,
            }
            | changes
        )
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fixed_n(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile((m, n), block_size=(None, 128)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _source_c(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = out[rows, cols].to(torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


@pytest.mark.parametrize("mode", (0, 1, 2, 3))
@pytest.mark.parametrize("acc_stages", (2, 1))
def test_native_selection_uses_physical_n(mode, acc_stages):
    choose = cute_mma._choose_mma_impl
    results = []

    def record(*args, **kwargs):
        results.append(choose(*args, **kwargs))
        return results[-1]

    with patch.object(cute_mma, "_choose_mma_impl", record):
        bound = scaled_mm._bind_isolated(_args(mode))
        state = bound.config_spec._cute_tcgen05_config
        with bound.env:
            config = _config(tcgen05_acc_stages=acc_stages)
            plan = facts._dense_aligned_n_plan(state, config.config)
            assert plan is not None
            assert plan.public_tile == (128, 128, 128)
            assert plan.physical_tile == (128, 80, 128)
            assert plan.max_ab_stages == 16
            assert _smem_bytes(80, ab_stages=16, acc_stages=acc_stages) == (
                219424 - 16 * (2 - acc_stages)
            )
        source = bound.to_code(config)
    assert results and all(result == "tcgen05" for result in results), results
    assert "make_tma_umma_pipeline(" in source
    assert "PipelineTmaStore.create" in source
    assert "(128, 80, 128)" in source


@pytest.mark.parametrize("acc_stages", (2, 1))
def test_public_roundtrips_grid_and_descriptors(acc_stages):
    bound = scaled_mm._bind_isolated(_args())
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        seed = _config(tcgen05_acc_stages=acc_stages)
        flat, canonical = generation.canonicalize_flat(generation.flatten(seed))
        assert canonical.block_sizes == [128, 128, 128]
        assert canonical.config["tcgen05_tile_n"] == 80
        assert canonical.config["tcgen05_acc_stages"] == acc_stages
        assert generation.canonicalize_flat(flat) == (flat, canonical)
        assert generation.unflatten(flat) == canonical
        assert any(
            s.config.get("tcgen05_tile_n") == 80
            and s.config["tcgen05_ab_stages"] == 16
            and s.config["tcgen05_acc_stages"] == acc_stages
            for s in bound.config_spec.compiler_seed_configs
        )
    source = bound.to_code(canonical)
    assert canonical.block_sizes == [128, 128, 128]
    ab, d = wrapper_plans(source)
    assert ab["kind"] == "tcgen05_ab_tma"
    assert (ab["bm"], ab["bn"], ab["bk"]) == (128, 80, 128)
    assert ab["use_2cta_instrs"] and ab["b_k_major"]
    assert (ab["cluster_m"], ab["cluster_n"]) == (2, 1)
    assert d["kind"] == "tcgen05_d_tma"
    assert (
        d["epi_tile_raw_expr"]
        == "cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape((64, 80), True, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, cutlass.BFloat16)"
    )
    scope = {
        "_BLOCK_SIZE_0": 128,
        "_BLOCK_SIZE_1": 80,
        "_BLOCK_SIZE_2": 128,
        "_NUM_SM": 152,
        "min": min,
        "max": max,
    }
    launch = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_launcher"
    )
    assert eval(
        compile(ast.Expression(launch.args[1]), "<generated-grid>", "eval"),
        {"__builtins__": {}},
        scope,
    ) == (2, 1, 64)


@pytest.mark.parametrize(
    "changes",
    (
        {"block_sizes": [128, 64, 128]},
        {"tcgen05_tile_n": 81},
        {"tcgen05_tile_n": 128},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_c_store_mode": "invalid"},
        {"tcgen05_aux_load_placement": "pre_acc_wait"},
        {"tcgen05_ab_stages": 17},
        {"tcgen05_acc_stages": 0},
        {"tcgen05_acc_stages": 3},
        {"tcgen05_acc_stages": True},
        {"tcgen05_acc_stages": "1"},
        {"pid_type": "flat"},
        {"loop_orders": [[1, 0]]},
    ),
)
def test_explicit_invalid_schedule_rejected(changes):
    bound = scaled_mm._bind_isolated(_args())
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        bound.to_code(_config(**changes))


@pytest.mark.parametrize("store_mode", ("normal", "direct"))
@pytest.mark.parametrize(
    "shape", ((127, 5120, 2880), (128, 5104, 2880), (128, 5120, 32), (128, 5120, 32800))
)
def test_unsupported_tails_and_traversals_rejected(shape, store_mode):
    bound = scaled_mm._bind_isolated(_args(shape=shape))
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        bound.to_code(_config(tcgen05_c_store_mode=store_mode))


@pytest.mark.parametrize("restriction", ("max64", "min128", "fixed128", "outer"))
def test_projection_preserves_live_n_bounds(restriction):
    bound = scaled_mm._bind_isolated(_args())
    spec = bound.config_spec.block_sizes[1]
    if restriction == "max64":
        spec.update_max(64)
    elif restriction == "min128":
        spec.update_min(128)
    elif restriction == "fixed128":
        spec.update_min(128)
        spec.update_max(128)
    else:
        spec.bounded_by_block_id = 0
    before = (spec.min_size, spec.max_size, spec.max_size_revision)
    assert all(
        seed.config["tcgen05_tile_n"] != 80
        for seed in seeds.dense_aligned_n_seed_configs(
            bound.config_spec._cute_tcgen05_config
        )
    )
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        bound.to_code(_config())
    assert before == (spec.min_size, spec.max_size, spec.max_size_revision)


def test_fixed_override_and_independent_pid_veto():
    bound = scaled_mm._bind_isolated(_args())
    with bound.env:
        fixed = ConfigGeneration(
            bound.config_spec, overrides={"block_sizes": [128, 128, 64]}
        )
        ordinary = _config(block_sizes=[128, 128, 64], tcgen05_ab_stages=16)
        canonical = fixed.unflatten(fixed.flatten(ordinary))
        assert canonical.config.get("tcgen05_tile_n", 0) == 0
        bad = ConfigGeneration(
            bound.config_spec,
            overrides={"block_sizes": [128, 128, 64], "tcgen05_tile_n": 80},
        )
        with pytest.raises(InvalidConfig, match="different physical N tile"):
            bad.unflatten(bad.flatten(ordinary))
    for pid in ("persistent_blocked", "persistent_interleaved"):
        bound.config_spec.disallow_pid_type(pid, "independent veto")
    assert not seeds.dense_aligned_n_seed_configs(
        bound.config_spec._cute_tcgen05_config
    )
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        bound.to_code(_config())


@pytest.mark.parametrize("store_mode", ("normal", "direct"))
def test_plan_ownership_tracks_live_mma_facts(store_mode):
    bound = scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    original = state.dense_gemm_facts
    assert original is not None
    config = _config(tcgen05_c_store_mode=store_mode)
    assert facts._dense_aligned_n_plan(state, config.config) is not None
    for changed in (
        replace(original, block_ids=(2, 1, 0)),
        replace(original, shape=(256, 5120, 2880)),
        replace(original, input_dtype=torch.bfloat16),
    ):
        with patch.object(state, "dense_gemm_facts", changed):
            assert facts._dense_aligned_n_plan(state, config.config) is None


@pytest.mark.parametrize(
    "width,c_stages,expected_max,expected_d",
    ((80, 2, 16, 4096), (80, 4, 16, 8192), (96, 2, 15, 8192), (96, 4, 14, 16384)),
)
@pytest.mark.parametrize("acc_stages", (2, 1))
def test_actual_d_ring_and_complete_stage_limit(
    width, c_stages, expected_max, expected_d, acc_stages
):
    bound = scaled_mm._bind_isolated(_args(shape=(128, width * 64, 4096)))
    config = _config(
        tcgen05_tile_n=width,
        block_sizes=[128, carrier_n(width), 128],
        tcgen05_c_stages=c_stages,
        tcgen05_acc_stages=acc_stages,
        tcgen05_ab_stages=expected_max,
    )
    plan = facts._dense_aligned_n_plan(
        bound.config_spec._cute_tcgen05_config, config.config
    )
    assert plan is not None
    assert plan.max_ab_stages == expected_max
    used = _smem_bytes(
        width, ab_stages=expected_max, c_stages=c_stages, acc_stages=acc_stages
    )
    # Two D stages: the difference between the four- and two-stage rings.
    assert (
        _smem_bytes(width, ab_stages=1, c_stages=4)
        - _smem_bytes(width, ab_stages=1, c_stages=2)
        == 2 * expected_d // c_stages
    )
    assert used <= 232448
    if expected_max < 16:
        assert (
            _smem_bytes(
                width,
                ab_stages=expected_max + 1,
                c_stages=c_stages,
                acc_stages=acc_stages,
            )
            > 232448
        )
    source = bound.to_code(config)
    assert wrapper_plans(source)[1]["c_stage_count"] == c_stages


# CuTe proofs of the emitted paired M128 epilogue copies.


@pytest.fixture
def mlir_module() -> Iterator[None]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield


def _native_epilogue(width, acc_stages):
    """The paired M128 accumulator and stage-zero T2R copy emitted for ``width``.

    Generated code loads TMEM with ``get_tmem_load_op(..., cutlass.Float32,
    cutlass.Float32, ...)`` for both normal and direct stores.
    """
    tile = h.compute_epilogue_tile_shape(
        (64, width), True, LayoutEnum.ROW_MAJOR, cutlass.BFloat16
    )
    mma = h.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        cute.nvgpu.tcgen05.CtaGroup.TWO,
        (128, width),
        cute.nvgpu.tcgen05.OperandSource.SMEM,
    )
    fragment = mma.make_fragment_C(
        cute.append(mma.partition_shape_C((128, width)), acc_stages)
    )
    acc = epi.transform_partitioned_tensor_layout(
        cute.make_tensor(
            cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16),
            fragment.layout,
        )
    )
    load = h.get_tmem_load_op(
        (64, width, 128),
        LayoutEnum.ROW_MAJOR,
        cutlass.Float32,
        cutlass.Float32,
        tile,
        True,
    )
    t2r = cute.nvgpu.tcgen05.make_tmem_copy(
        load, cute.flat_divide(acc, tile)[None, None, 0, 0, 0]
    )
    return tile, mma, fragment, t2r


def _rank_points(mma, rank, width, tile):
    """Logical (m, n) coordinates of CTA ``rank``'s accumulator, per subtile."""
    return cute.flat_divide(
        epi.transform_partitioned_tensor_layout(
            mma.get_slice(rank).partition_C(cute.make_identity_tensor((128, width)))
        ),
        tile,
    )


@pytest.mark.usefixtures("mlir_module")
@pytest.mark.parametrize("width", ALIGNED_N_CHOICES)
def test_every_advertised_width_has_exact_compact_output_ownership(width):
    tile, mma, fragment, t2r = _native_epilogue(width, 2)
    tile_n = cute.size(tile[1])
    subtiles = width // tile_n
    r2s = cute.make_tiled_copy_D(
        h.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, cutlass.BFloat16, cutlass.Float32, t2r
        ),
        t2r,
    )
    # The (row, compact column, stage) slots of each thread's R2S registers in
    # the four-stage D ring; they do not depend on the CTA rank or subtile.
    slots = []
    for thread in range(128):
        destination = r2s.get_slice(thread).partition_D(
            cute.make_identity_tensor((64, tile_n, 4))
        )
        stages = [destination[None, None, None, stage] for stage in range(4)]
        slots.append(
            [
                [tuple(view[index]) for view in stages]
                for index in range(cute.size(stages[0].shape))
            ]
        )
    coverage = set()
    for rank in range(2):
        divided = _rank_points(mma, rank, width, tile)
        written = [[] for _ in range(subtiles)]
        for thread in range(128):
            partition = t2r.get_slice(thread).partition_D(divided)
            for subtile, coordinates in enumerate(written):
                reg = r2s.retile(partition[None, None, None, 0, subtile])
                assert cute.size(reg.shape) == len(slots[thread])
                for index, stage_slots in enumerate(slots[thread]):
                    point = tuple(reg[index])
                    coordinates.append(point)
                    compact_n = (
                        point[1] % (width // 2)
                        - subtile * (tile_n // 2)
                        + point[1] // (width // 2) * (tile_n // 2)
                    )
                    assert stage_slots == [
                        (point[0] - rank * 64, compact_n, stage) for stage in range(4)
                    ]
        for coordinates in written:
            assert len(coordinates) == len(set(coordinates)) == 64 * tile_n
            assert not coverage.intersection(coordinates)
            coverage.update(coordinates)
    assert coverage == {(m, n) for m in range(128) for n in range(width)}
    # ACC1 changes only the trailing TMEM stage mode, so its stage-zero copy
    # (and therefore the ownership proved above) is the ACC2 copy.
    _tile, _mma, acc1_fragment, acc1_t2r = _native_epilogue(width, 1)
    assert str(acc1_t2r) == str(t2r)
    # The admission budget prices exactly these CuTe allocations.
    a = h.make_smem_layout_a(
        mma, (128, width, 128), cutlass.Float8E4M3FN, 1, is_k_major=True
    )
    b = h.make_smem_layout_b(
        mma, (128, width, 128), cutlass.Float8E4M3FN, 1, is_k_major=True
    )
    d = h.make_smem_layout_epi(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, tile, 1)
    assert cute.cosize(a) + cute.cosize(b) == (128 + width) * 128 // 2
    assert 2 * cute.cosize(d) == 64 * tile_n * 2
    for acc_stages, staged in ((1, acc1_fragment), (2, fragment)):
        assert _smem_bytes(width, ab_stages=1, acc_stages=acc_stages) == (
            cute.cosize(a)
            + cute.cosize(b)
            + 2 * 2 * cute.cosize(d)
            + 2048
            + 16
            + 16 * acc_stages
        )
        assert cutlass.utils.get_num_tmem_alloc_cols(staged, arch="sm_100") <= 512


@pytest.mark.parametrize("width", ALIGNED_N_CHOICES)
@pytest.mark.parametrize("acc_stages", (2, 1))
def test_every_advertised_width_emits_native_complete_budget(width, acc_stages):
    bound = scaled_mm._bind_isolated(_args(3, shape=(256, width * 160, 4096)))
    config = _config(
        tcgen05_tile_n=width,
        block_sizes=[128, carrier_n(width), 128],
        tcgen05_ab_stages=1,
        tcgen05_acc_stages=acc_stages,
    )
    state = bound.config_spec._cute_tcgen05_config
    plan = facts._dense_aligned_n_plan(state, config.config)
    assert plan is not None
    config.config["tcgen05_ab_stages"] = plan.max_ab_stages
    source = bound.to_code(config)
    descriptors = wrapper_plans(source)
    assert descriptors[0]["bn"] == width
    assert descriptors[0]["ab_stage_count"] == plan.max_ab_stages
    assert descriptors[1]["epi_tile_raw_expr"] == _d_tile_expr(width)
    assert "make_tma_umma_pipeline(" in source
    assert "PipelineTmaStore.create" in source
    assert "block=(32, 6, 1)" in source
    # The TMEM load atom that the ownership proofs above model.
    assert (
        f"get_tmem_load_op((64, {width}, 128), tcgen05_c_layout, cutlass.Float32, "
        "cutlass.Float32, tcgen05_epi_tile, True)"
    ) in source
    wrapper = []
    for descriptor in descriptors:
        _append_cute_wrapper_plan(wrapper, [], descriptor)
    assert _d_tile_expr(width) in "\n".join(wrapper)


def test_seed_and_coordinate_search_share_the_same_physical_plan():
    bound = scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        fragments = surface.optional_fragments(state, for_search=True)
        assert fragments["tcgen05_tile_n"].choices == (0, 80, 160)
        for seed in seeds.dense_aligned_n_seed_configs(state):
            flat, canonical = generation.canonicalize_flat(generation.flatten(seed))
            assert canonical.config["tcgen05_tile_n"] == seed.config["tcgen05_tile_n"]
            assert canonical.block_sizes == seed.block_sizes
            assert facts._dense_aligned_n_plan(state, canonical.config) is not None
            assert generation.canonicalize_flat(flat) == (flat, canonical)
        # A raw coordinate mutation repairs the carrier/protocol and
        # retains the advertised physical width.
        mutated = _config(tcgen05_tile_n=160, tcgen05_ab_stages=1)
        _flat, canonical = generation.canonicalize_flat(generation.flatten(mutated))
        assert canonical.config["tcgen05_tile_n"] == 160
        assert canonical.block_sizes == [128, 256, 128]


def test_wrapper_uses_device_exact_d_layout():
    bound = scaled_mm._bind_isolated(_args(3))
    source = bound.to_code(_config())
    plan = facts._dense_aligned_n_plan(
        bound.config_spec._cute_tcgen05_config, _config().config
    )
    assert plan is not None
    wrapper = []
    for descriptor in wrapper_plans(source):
        _append_cute_wrapper_plan(wrapper, [], descriptor)
    text = "\n".join(wrapper)
    assert _d_tile_expr(80) in text
    assert "(128, 80, 128)" in text
    assert "CtaGroup.TWO" in text


@pytest.mark.parametrize(
    "implementation", (None, "auto", "tcgen05", "warp", "universal")
)
def test_explicit_native_implementation_contract(implementation, monkeypatch):
    if implementation is None:
        monkeypatch.delenv("HELION_CUTE_MMA_IMPL", raising=False)
    else:
        monkeypatch.setenv("HELION_CUTE_MMA_IMPL", implementation)
    bound = scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    seed_configs = seeds.dense_aligned_n_seed_configs(state)
    fragments = surface.optional_fragments(state, for_search=True)
    if implementation in ("warp", "universal"):
        assert not seed_configs
        assert "tcgen05_tile_n" not in fragments
        with pytest.raises(BackendUnsupported, match="requires its supported native"):
            bound.to_code(_config())
    else:
        assert seed_configs and fragments["tcgen05_tile_n"].choices == (0, 80, 160)
        code = bound.to_code(_config())
        assert "make_tma_umma_pipeline(" in code
    ordinary = _config(tcgen05_tile_n=0, tcgen05_ab_stages=2)
    omitted = helion.Config.from_dict(
        {
            key: value
            for key, value in ordinary.config.items()
            if key != "tcgen05_tile_n"
        }
    )
    assert bound.to_code(ordinary) == bound.to_code(omitted)


@pytest.mark.parametrize(
    "failure", ("unsupported_dtype", "shape", "budget", "wrong_tile")
)
def test_positive_family_cannot_fall_back_from_native_admission(failure, monkeypatch):
    monkeypatch.setenv("HELION_CUTE_MMA_IMPL", "auto")
    bound = scaled_mm._bind_isolated(_args())
    config = _config()
    support = cute_mma.get_cute_mma_support()
    with (
        bound.env,
        patch.object(
            cute_mma,
            "get_cute_mma_support",
            return_value=SimpleNamespace(**(vars(support) | {"tcgen05_f8": False}))
            if failure == "unsupported_dtype"
            else support,
        ),
        patch.object(
            cute_mma,
            "_mma_impl_matches_problem_shape",
            return_value=failure != "shape",
        ),
        patch.object(
            cute_mma,
            "_tcgen05_candidate_exceeds_smem",
            return_value=failure == "budget",
        ),
        pytest.raises(BackendUnsupported, match="requires its supported native"),
    ):
        cute_mma._choose_mma_impl(
            torch.float8_e4m3fn,
            bm=128,
            bn=64 if failure == "wrong_tile" else 128,
            bk=128,
            config=config,
            input_device=torch.device("cuda:0"),
        )


@pytest.mark.parametrize(
    "kind", ("fixed_n", "bf16", "non_k_major", "source_c", "scale_dtype")
)
@pytest.mark.parametrize("store_mode", ("normal", "direct"))
def test_unproved_graphs_never_advertise_or_admit_physical_width(kind, store_mode):
    args = list(_args(3))
    kernel = scaled_mm
    if kind == "fixed_n":
        kernel = _fixed_n
        args = args[:2]
    elif kind == "bf16":
        args[:2] = [tensor.to(torch.bfloat16) for tensor in args[:2]]
    elif kind == "non_k_major":
        args[1] = args[1].contiguous()
    elif kind == "source_c":
        kernel = _source_c
        args = [*args[:2], torch.empty((128, 5120), dtype=torch.bfloat16)]
    elif kind == "scale_dtype":
        args[2:] = [tensor.to(torch.bfloat16) for tensor in args[2:]]
    bound = kernel._bind_isolated(args)
    state = bound.config_spec._cute_tcgen05_config
    assert not seeds.dense_aligned_n_seed_configs(state)
    assert "tcgen05_tile_n" not in surface.optional_fragments(state, for_search=True)
    if kind == "fixed_n":
        assert facts._matmul_block_fragments(state) is None
        config = _config(block_sizes=[128, 128], tcgen05_c_store_mode=store_mode)
    else:
        config = _config(tcgen05_c_store_mode=store_mode)
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        bound.to_code(config)


# Direct (R2G) epilogues retain collective and source-admission invariants.


@pytest.mark.parametrize("width", (80, 160))
@pytest.mark.parametrize("acc_stages", (1, 2))
@pytest.mark.parametrize("c_stages", (2, 4))
def test_direct_roundtrip_keeps_budget_and_collective_release(
    width, acc_stages, c_stages
):
    bound = scaled_mm._bind_isolated(_args(3))
    state = bound.config_spec._cute_tcgen05_config
    direct = _config(
        block_sizes=[128, carrier_n(width), 128],
        tcgen05_tile_n=width,
        tcgen05_acc_stages=acc_stages,
        tcgen05_c_stages=c_stages,
        tcgen05_c_store_mode="direct",
        tcgen05_ab_stages=1,
    )
    normal_values = direct.config | {"tcgen05_c_store_mode": "normal"}
    with bound.env:
        normal_plan = facts._dense_aligned_n_plan(state, normal_values)
        plan = facts._dense_aligned_n_plan(state, direct.config)
        assert plan is not None and plan == normal_plan
        direct.config["tcgen05_ab_stages"] = plan.max_ab_stages
        generation = ConfigGeneration(bound.config_spec)
        flat, canonical = generation.canonicalize_flat(generation.flatten(direct))
        assert canonical.config["tcgen05_c_store_mode"] == "direct"
        assert canonical.block_sizes == direct.block_sizes
        assert generation.canonicalize_flat(flat) == (flat, canonical)
        assert generation.unflatten(flat) == canonical
        for fix in (False, True):
            values = dict(direct.config)
            repair._normalize_dense_aligned_n(state, values, fix_invalid=fix)
            assert values["tcgen05_c_store_mode"] == "direct"
        with pytest.raises(
            InvalidConfig,
            match=rf"tcgen05_tile_n .* AB stages 1\.\.{plan.max_ab_stages}$",
        ):
            repair._normalize_dense_aligned_n(
                state,
                direct.config | {"tcgen05_ab_stages": plan.max_ab_stages + 1},
                fix_invalid=False,
            )
    source = bound.to_code(canonical)
    assert [plan["kind"] for plan in wrapper_plans(source)] == ["tcgen05_ab_tma"]
    assert "CopyR2GOp" in source
    assert "make_tma_umma_pipeline(" in source
    assert "PipelineTmaStore.create" not in source
    assert "tcgen05_sD_ptr = cute.arch.alloc_smem" not in source
    assert "tcgen05_tiled_copy_r2s" not in source
    assert "tcgen05_tma_store_role_tile" not in source
    assert "cutlass.pipeline.Agent.Thread, cutlass.Int32(8)" in source
    assert "tcgen05_acc_consumer_state.index" in source
    final = source.index("if _tcgen05_subtile == tcgen05_subtile_count - 1:")
    fence = source.index("fence_view_async_tmem_load()", final)
    release = source.index(".consumer_release(tcgen05_acc_consumer_state)", fence)
    store = source.index("cute.copy(tcgen05_simt_atom", release)
    advance = source.index("tcgen05_acc_consumer_state.advance()", store)
    free = source.index("tcgen05_tmem_allocator.free(", advance)
    assert final < fence < release < store < advance < free


def test_direct_seeds_append_exact_normal_counterparts():
    bound = scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    seed_configs = seeds.dense_aligned_n_seed_configs(state)
    direct_at = next(
        i
        for i, seed in enumerate(seed_configs)
        if seed.config["tcgen05_c_store_mode"] == "direct"
    )
    normal, direct = seed_configs[:direct_at], seed_configs[direct_at:]
    assert normal and len(normal) == len(direct)
    assert all(s.config["tcgen05_c_store_mode"] == "normal" for s in normal)
    for before, after in zip(normal, direct, strict=True):
        assert after.config == before.config | {"tcgen05_c_store_mode": "direct"}
        assert facts._dense_aligned_n_plan(
            state, after.config
        ) == facts._dense_aligned_n_plan(state, before.config)
    assert dense_aligned_n_defaults()["tcgen05_c_store_mode"] == "normal"


@pytest.mark.parametrize("store_mode", ("invalid", None, True, 0))
def test_invalid_store_mode_rejects_explicit_and_repairs_search(store_mode):
    bound = scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    values = _config(tcgen05_c_store_mode=store_mode).config
    with bound.env:
        assert facts._dense_aligned_n_plan(state, values) is None
        with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
            repair._normalize_dense_aligned_n(state, dict(values), fix_invalid=False)
        repaired = dict(values)
        repair._normalize_dense_aligned_n(state, repaired, fix_invalid=True)
        assert repaired["tcgen05_c_store_mode"] == "normal"
        assert facts._dense_aligned_n_plan(state, repaired) is not None


@pytest.mark.parametrize(
    "changes",
    (
        {"tcgen05_cluster_n": 2},
        {"tcgen05_aux_load_placement": "pre_acc_wait"},
        {"tcgen05_aux_load_mode": "tma"},
        {"tcgen05_warp_spec_store_warps": 1},
        {"tcgen05_epilogue_layout": "split_first_t2r"},
        {"loop_orders": [[1, 0]]},
        {"pid_type": "flat"},
    ),
)
def test_direct_does_not_widen_other_protocols(changes):
    bound = scaled_mm._bind_isolated(_args())
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        bound.to_code(_config(tcgen05_c_store_mode="direct", **changes))


@pytest.mark.usefixtures("mlir_module")
@pytest.mark.parametrize("width", ALIGNED_N_CHOICES)
def test_direct_output_ownership_and_vector_alignment(width):
    """Prove the actual permuted partition, including every subtile pointer.

    A physical width being divisible by eight does not itself prove that each
    thread's reasserted pointer is 16-byte aligned. Enumerate both CTA ranks
    and check the copy atom's actual common layout against global addresses.
    """
    row_stride = 5120
    tile, mma, _fragment, t2r = _native_epilogue(width, 2)
    # The stage-zero copy does not depend on the accumulator stage count.
    assert str(_native_epilogue(width, 1)[3]) == str(t2r)
    coverage = set()
    for rank in range(2):
        points = _rank_points(mma, rank, width, tile)
        memory = cute.flat_divide(
            epi.transform_partitioned_tensor_layout(
                mma.get_slice(rank).partition_C(
                    cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            0,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((128, width), stride=(row_stride, 1)),
                    )
                )
            ),
            tile,
        )
        rank_points = []
        for thread in range(128):
            pc = t2r.get_slice(thread).partition_D(points)
            pm = t2r.get_slice(thread).partition_D(memory)
            for subtile in range(width // int(cute.size(tile[1]))):
                coordinates = pc[None, None, None, 0, subtile]
                destination = pm[None, None, None, 0, subtile]
                registers = cute.make_rmem_tensor(destination.shape, cutlass.BFloat16)
                common = cute.max_common_layout(registers.layout, destination.layout)
                copy_bits = min(128, int(cute.size(common)) * 16)
                assert copy_bits == 128
                values = [
                    tuple(coordinates[i])
                    for i in range(int(cute.size(coordinates.shape)))
                ]
                # This is the base pointer explicitly asserted 16B
                # aligned in the generated direct-store epilogue.
                assert (values[0][0] * row_stride + values[0][1]) * 2 % 16 == 0
                vector_elements = copy_bits // 16
                for first in range(0, len(values), vector_elements):
                    chunk = values[first : first + vector_elements]
                    assert len({m for m, _n in chunk}) == 1
                    addresses = [(m * row_stride + n) * 2 for m, n in chunk]
                    assert addresses == list(
                        range(addresses[0], addresses[0] + 2 * vector_elements, 2)
                    )
                    assert addresses[0] % (copy_bits // 8) == 0
                rank_points.extend(values)
        assert len(rank_points) == len(set(rank_points)) == 64 * width
        assert not coverage.intersection(rank_points)
        coverage.update(rank_points)
    assert coverage == {(m, n) for m in range(128) for n in range(width)}
    # Arbitrary subsequent full physical scheduler tiles preserve the
    # same pointer and copy alignment in both logical dimensions.
    assert width * 2 % 16 == 0
    assert 128 * row_stride * 2 % 16 == 0


@pytest.mark.parametrize("rhs_k_major", (True, False))
def test_sixteen_bit_pair_writes_physical_n_into_argument_output(rhs_k_major):
    """The 16-bit family is the paired M256 instruction at a 192-wide physical N
    behind the 256 carrier; an Inductor-style argument output the kernel never
    reads owns its store ring, and 16-bit operands may be MN-major."""
    bound = plain_mm_into._bind_isolated(
        bf16_mm_args(8192, 768, 768, rhs_k_major=rhs_k_major)
    )
    state = bound.config_spec._cute_tcgen05_config
    dense_facts = state.dense_gemm_facts
    assert dense_facts is not None and dense_facts.plain_epilogue
    with bound.env:
        plan = facts._dense_aligned_n_plan(state, aligned_n_bf16_config().config)
        assert plan is not None
        assert plan.public_tile == (256, 256, 64)
        assert plan.physical_tile == (256, 192, 64)
        # (128, 32) C subtiles leave room for seven 28 KiB stages; CuTe's
        # with-source (128, 64) subtile would cap the ring at six.
        assert plan.max_ab_stages == 7
        # One deepest-ring seed per width that divides N.
        seeds = {
            s.config["tcgen05_tile_n"]: s.config["tcgen05_ab_stages"]
            for s in bound.config_spec.compiler_seed_configs
            if s.config.get("tcgen05_tile_n")
        }
        assert set(seeds) == {96, 192} and seeds[192] == 7
        # Two N tiles per pair split N evenly (A multicast).
        assert (
            facts._dense_aligned_n_plan(
                state, aligned_n_bf16_config(tcgen05_cluster_n=2).config
            )
            is not None
        )
    source = bound.to_code(aligned_n_bf16_config())
    ab, d = wrapper_plans(source)
    assert (ab["bm"], ab["bn"], ab["bk"]) == (256, 192, 64)
    assert ab["ab_stage_count"] == 7
    assert ab.get("b_k_major", False) is rhs_k_major
    assert (d["bm"], d["bn"], d["epi_tile_n"]) == (256, 192, 32)


def test_sixteen_bit_pair_prices_configured_epilogue_and_keeps_sampled_raster():
    bound = plain_mm_into._bind_isolated(bf16_mm_args(8192, 768, 768))
    state = bound.config_spec._cute_tcgen05_config
    with bound.env:
        # A configured wider subtile or a second epilogue warp group grows the
        # C ring past what seven AB stages leave.
        for changes in (
            {"tcgen05_epi_subtile_n": 64},
            {"tcgen05_epi_warp_groups": 2},
        ):
            plan = facts._dense_aligned_n_plan(
                state, aligned_n_bf16_config(**changes).config
            )
            assert plan is not None and plan.max_ab_stages == 6
        # A configured subtile must partition the physical width (96), not
        # only the 128 carrier.
        assert (
            facts._dense_aligned_n_plan(
                state,
                aligned_n_bf16_config(
                    tcgen05_tile_n=96,
                    block_sizes=[256, 128, 64],
                    tcgen05_epi_subtile_n=64,
                ).config,
            )
            is None
        )
        # The search projection keeps a sampled L2 grouping and N pairing the
        # 16-bit pair admits instead of resetting them to the family default,
        # pins the 64-deep K block, and drops a configured subtile that does
        # not partition the physical width before judging them.
        values = dict(
            aligned_n_bf16_config(
                l2_groupings=[8], tcgen05_cluster_n=2, tcgen05_ab_stages=9
            )
        )
        bound.config_spec.normalize(values, _fix_invalid=True)
        narrow = dict(
            aligned_n_bf16_config(
                tcgen05_tile_n=96,
                block_sizes=[256, 128, 32],
                tcgen05_epi_subtile_n=64,
                l2_groupings=[8],
                tcgen05_cluster_n=2,
            )
        )
        bound.config_spec.normalize(narrow, _fix_invalid=True)
    assert values["l2_groupings"] == [8]
    assert values["tcgen05_cluster_n"] == 2
    assert values["tcgen05_ab_stages"] == 7
    assert narrow["block_sizes"] == [256, 128, 64]
    assert narrow["tcgen05_epi_subtile_n"] is None
    assert narrow["l2_groupings"] == [8]
    assert narrow["tcgen05_cluster_n"] == 2


def test_sixteen_bit_pair_rings_tails_and_refusals():
    # An argument output the epilogue also reads owns no store ring.
    for reader in (output_reading_mm, output_alias_reading_mm):
        reads = reader._bind_isolated(bf16_mm_args(8192, 768, 768))
        gemm = analyze_native_gemm(reads.env, reads.host_function.device_ir)
        assert gemm is not None and not gemm.owns_output_ring
    # The TMA store clips at the destination's edge: an output with rows past
    # M leaves the family.
    x, y, _out = bf16_mm_args(1152, 768, 768)
    taller = plain_mm_into._bind_isolated(
        (x, y, torch.empty((1280, 768), dtype=torch.bfloat16))
    )
    assert taller.config_spec._cute_tcgen05_config.dense_gemm_facts is None
    # Any 16-bit output: an fp16 GEMM keeping its dtype, or bf16 into fp16.
    x, y, out = bf16_mm_args(8192, 768, 768)
    for args in (
        (x.to(torch.float16), y.to(torch.float16), out.to(torch.float16)),
        (x, y, out.to(torch.float16)),
    ):
        half = plain_mm_into._bind_isolated(args)
        assert facts._dense_aligned_n_widths(half.config_spec._cute_tcgen05_config) == (
            96,
            192,
        )
    # A partial last M tile is clipped by the TMA store: 8320 rows keep the
    # family, its (128, 32) C ring and the full-tile TMA-store epilogue.
    tail = plain_mm_into._bind_isolated(bf16_mm_args(8320, 768, 768))
    tail_state = tail.config_spec._cute_tcgen05_config
    with tail.env:
        tail_plan = facts._dense_aligned_n_plan(
            tail_state, aligned_n_bf16_config().config
        )
    assert tail_plan is not None and tail_plan.max_ab_stages == 7
    tail_source = tail.to_code(aligned_n_bf16_config())
    assert "tcgen05_tma_store_atom" in tail_source
    assert "(cute.make_layout(128), cute.make_layout(32))" in tail_source
    # A partial last K tile is not: the C stage falls back to the with-source
    # (128, 64) subtile and the ring to six stages.
    k_tail = plain_mm_into._bind_isolated(bf16_mm_args(8192, 768, 800))
    with k_tail.env:
        k_tail_plan = facts._dense_aligned_n_plan(
            k_tail.config_spec._cute_tcgen05_config, aligned_n_bf16_config().config
        )
    assert k_tail_plan is not None and k_tail_plan.max_ab_stages == 6
    # The M-edge schedule needs whole K tiles: M and K tails together leave.
    both = plain_mm_into._bind_isolated(bf16_mm_args(8320, 768, 800))
    with both.env:
        assert (
            facts._dense_aligned_n_plan(
                both.config_spec._cute_tcgen05_config, aligned_n_bf16_config().config
            )
            is None
        )
        # A non-positive configured subtile is left to the epilogue validation.
        assert (
            facts._dense_aligned_n_plan(
                k_tail.config_spec._cute_tcgen05_config,
                aligned_n_bf16_config(tcgen05_epi_subtile_n=0).config,
            )
            is not None
        )
    # Each CTA's half of an 80-wide K-major B tile is not whole 16-row atoms.
    odd = plain_mm_into._bind_isolated(bf16_mm_args(8192, 960, 768))
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        odd.to_code(
            aligned_n_bf16_config(tcgen05_tile_n=80, block_sizes=[256, 128, 64])
        )
    assert {
        s.config["tcgen05_tile_n"]
        for s in odd.config_spec.compiler_seed_configs
        if s.config.get("tcgen05_tile_n")
    } == {96, 160, 192}
    # 960 columns split into five 192-wide tiles: no N pairing.
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        odd.to_code(aligned_n_bf16_config(tcgen05_cluster_n=2))


@pytest.mark.parametrize("m", [8192, 8320])
def test_sixteen_bit_pair_renders_the_direct_store(m):
    # The search's direct register-to-global store on the 192-wide pair,
    # whole and over an M tail (the edge tiles' predicated SIMT store); the
    # GPU tests run it on an N tail.
    bound = plain_mm_into._bind_isolated(bf16_mm_args(m, 768, 768))
    source = bound.to_code(
        aligned_n_bf16_config(tcgen05_ab_stages=6, tcgen05_c_store_mode="direct")
    )
    assert "_BLOCK_SIZE_1 = 192" in source
    assert "tcgen05_tma_store_atom" not in source


def test_sixteen_bit_pair_scales_its_physical_columns():
    # A per-column FP32 scale on the 16-bit family: the 192-wide pair is
    # seeded and renders (the GPU test runs it,
    # ``test_sixteen_bit_aligned_n_pair_scales_columns``).
    x, y, out = bf16_mm_args(8192, 768, 768)
    scale = torch.empty((768,), dtype=torch.float32)
    bound = column_scale_mm_into._bind_isolated((x, y, scale, out))
    assert any(
        seed.config.get("tcgen05_tile_n") == 192
        for seed in bound.config_spec.compiler_seed_configs
    )
    source = bound.to_code(aligned_n_bf16_config(tcgen05_ab_stages=6))
    assert "_BLOCK_SIZE_1 = 192" in source


def test_fp8_argument_output_owns_its_ring():
    """An FP8 scaled GEMM writing an argument output it never reads owns its
    store ring: the dense aligned-N and one-CTA pipeline families apply."""
    a, b, scale_a, scale_b = scaled_args(256, 5120, 2880)
    bound = scaled_mm_into._bind_isolated(
        (a, b, scale_a, scale_b, torch.empty((256, 5120), dtype=torch.bfloat16))
    )
    state = bound.config_spec._cute_tcgen05_config
    assert state.dense_gemm_facts is not None
    assert state.one_cta_pipeline_facts is not None
    assert any(
        s.config.get("tcgen05_tile_n") == 80
        for s in bound.config_spec.compiler_seed_configs
    )
