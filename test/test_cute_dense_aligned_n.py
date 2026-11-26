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
from helion._compiler.cute.dense_aligned_n import pair_tile_wave_cost
from helion._compiler.cute.dense_aligned_n import transposed_double_edge_without_pairs
from helion._compiler.cute.dense_aligned_n import transposed_pair_tile_gain
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.strategies import tcgen05_two_cta_m128_epilogue_tile_expr
from helion._compiler.cute.tcgen05_config import facts
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_constants import TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
from helion._compiler.cute.tcgen05_schedule_facts import analyze_native_gemm
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import kernel_body
from helion._testing.cute_dense_tcgen05 import aligned_n_bf16_config
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import bias_mm_into
from helion._testing.cute_dense_tcgen05 import column_scale_mm_into
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import output_alias_reading_mm
from helion._testing.cute_dense_tcgen05 import output_reading_mm
from helion._testing.cute_dense_tcgen05 import plain_mm
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


def _seeded_widths(bound) -> set[int]:
    return {
        seed.config["tcgen05_tile_n"]
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("tcgen05_tile_n")
    }


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
        # One deepest-ring seed per searched width: the widths dividing N
        # (MN-major B: those whose half rows keep 64-byte swizzle atoms).
        seeds = {
            s.config["tcgen05_tile_n"]: s.config["tcgen05_ab_stages"]
            for s in bound.config_spec.compiler_seed_configs
            if s.config.get("tcgen05_tile_n")
        }
        assert set(seeds) == set(facts._dense_aligned_n_widths(state))
        assert set(seeds) == ({96, 192} if rhs_k_major else {192})
        assert seeds[192] == 7
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


def test_pair_tile_wave_cost():
    def cost(m: int, n: int, k: int, width: int) -> float:
        return pair_tile_wave_cost(m, n, k, width=width, block_m=256, sm_pairs=74)

    # C4_15: one wave of 72 256x192 pairs against one of 48 256x256.
    assert cost(3072, 1024, 6272, 192) / cost(3072, 1024, 6272, 256) == (
        pytest.approx(0.78, abs=0.01)
    )
    assert cost(3072, 1024, 6272, 224) / cost(3072, 1024, 6272, 256) == (
        pytest.approx(0.88, abs=0.01)
    )
    # Narrow tiles are stream-bound: one wave of 256x64 pairs beats the
    # clipped 96-wide one.
    assert cost(1024, 1024, 512, 64) < cost(1024, 1024, 512, 96)
    # A fuller wave count costs whole waves: 2048^3 takes two of any width
    # under 256.
    assert cost(2048, 2048, 2048, 224) > cost(2048, 2048, 2048, 256)


def test_sixteen_bit_pair_judges_full_tiles_at_its_width():
    # 960 columns are five whole 192-wide tiles but no whole 256-wide ones:
    # the full-tile rules (two epilogue warp groups here) judge the
    # physical width, as codegen does, so the config renders.
    bound = plain_mm_into._bind_isolated(bf16_mm_args(8192, 960, 768))
    source = bound.to_code(
        aligned_n_bf16_config(tcgen05_ab_stages=4, tcgen05_epi_warp_groups=2)
    )
    assert "_BLOCK_SIZE_1 = 192" in source


def test_sixteen_bit_pair_n_tails_and_fp32_outputs():
    # A weight gradient: bf16 operands into an fp32 output whose N (1024) no
    # width divides. A width with a partial last N tile (its loads zero-fill,
    # its TMA store clips) is admitted; its C ring takes the with-source
    # subtile at the stored width (six AB stages): the output is fp32, or
    # (bf16) its 24 tiles fill less than one wave, so no epilogue hides behind
    # a next tile (``tcgen05_plain_narrow_subtile_covers_n``; 8192 rows take
    # the plain (128, 32) ring and seven, below). N tiles pair only when
    # whole. Here no clipped width is seeded or searched: one wave of 256x64
    # pairs beats them.
    x, y, out = bf16_mm_args(1024, 1024, 512)
    for dtype in (torch.float32, torch.bfloat16):
        bound = plain_mm_into._bind_isolated((x, y, out.to(dtype)))
        state = bound.config_spec._cute_tcgen05_config
        assert facts._dense_aligned_n_widths(state) == ()
        assert _seeded_widths(bound) == set()
        with bound.env:
            plan = facts._dense_aligned_n_plan(state, aligned_n_bf16_config().config)
            paired = facts._dense_aligned_n_plan(
                state, aligned_n_bf16_config(tcgen05_cluster_n=2).config
            )
        assert plan is not None and plan.max_ab_stages == 6
        assert paired is None
        # both outputs render at the priced depth, through CuTe's tile: the
        # FP32 one is the C4_15 weight gradient's
        source = bound.to_code(aligned_n_bf16_config(tcgen05_ab_stages=6))
        assert "_BLOCK_SIZE_1 = 192" in source
        assert "tcgen05_tma_store_atom" in source
        ab, d = wrapper_plans(source)
        assert ab["ab_stage_count"] == 6
        assert d.get("epi_tile_n") is None
    # 32 x 6 = 192 tiles of the 192-wide pair over 8192 x 1024 run more than
    # one per CTA pair: the 16-bit output prices and renders the (128, 32)
    # ring, which leaves room for seven AB stages.
    tall = plain_mm_into._bind_isolated(bf16_mm_args(8192, 1024, 512))
    with tall.env:
        plan = facts._dense_aligned_n_plan(
            tall.config_spec._cute_tcgen05_config, aligned_n_bf16_config().config
        )
    assert plan is not None and plan.max_ab_stages == 7
    ab, d = wrapper_plans(tall.to_code(aligned_n_bf16_config(tcgen05_ab_stages=7)))
    assert ab["ab_stage_count"] == 7
    assert d["epi_tile_n"] == 32
    # Multiple epilogue warp groups need whole physical tiles: refused as
    # given, dropped when the autotuner repairs a sampled config.
    groups = aligned_n_bf16_config(tcgen05_ab_stages=4, tcgen05_epi_warp_groups=2)
    with pytest.raises(InvalidConfig, match="static full output tiles"):
        bound.to_code(groups)
    repaired = dict(groups.config)
    bound.config_spec.normalize(repaired, _fix_invalid=True)
    assert repaired["tcgen05_epi_warp_groups"] == 1
    assert repaired["tcgen05_tile_n"] == 192
    # 3072x1024x6272 (C4_15): one wave of 72 256x192 pairs against one of 48
    # 256x256 (22% less by the cost model) is seeded; 224 (12%) is not, and
    # neither are the widths that are not multiples of 32 (176 and 208, 25%
    # and 16%). The search, which measures its samples, also tries every
    # clipped width the model ranks ahead of 256x256.
    x, y, out = bf16_mm_args(3072, 1024, 6272)
    c4_15 = plain_mm_into._bind_isolated((x, y, out.float()))
    assert _seeded_widths(c4_15) == {192}
    assert facts._dense_aligned_n_widths(c4_15.config_spec._cute_tcgen05_config) == (
        176,
        192,
        208,
        224,
        240,
    )
    # 2048^3: 64 256x256 pairs fill one wave; every clipped width but 240
    # takes two (72 256x240 pairs still fit the 76 SM pairs, 6% less by the
    # model: searched, not seeded).
    square = plain_mm_into._bind_isolated(bf16_mm_args(2048, 2048, 2048))
    assert facts._dense_aligned_n_widths(square.config_spec._cute_tcgen05_config) == (
        240,
    )
    assert _seeded_widths(square) == set()
    # A width that divides N is always searched. An fp32 output still prices
    # the with-source subtile (six stages); a 16-bit one keeps the (128, 32)
    # ring and seven.
    x, y, out = bf16_mm_args(1024, 768, 512)
    for dtype, depth in ((torch.float32, 6), (torch.bfloat16, 7)):
        bound = plain_mm_into._bind_isolated((x, y, out.to(dtype)))
        state = bound.config_spec._cute_tcgen05_config
        assert facts._dense_aligned_n_widths(state) == (96, 192)
        with bound.env:
            plan = facts._dense_aligned_n_plan(state, aligned_n_bf16_config().config)
        assert plan is not None and plan.max_ab_stages == depth
    # 160 columns: a clipped 96-wide tile has as many tiles as the 128-wide
    # pairs and is not seeded; 160 itself (a 16-bit N between 128 and 256,
    # carried by the 256-wide N tile) is one whole N tile and is.
    assert _seeded_widths(
        plain_mm_into._bind_isolated(bf16_mm_args(1024, 160, 512))
    ) == {160}
    # The edge schedule needs whole K tiles: N and K tails together leave.
    both = plain_mm_into._bind_isolated(bf16_mm_args(1024, 1024, 544))
    with both.env:
        assert (
            facts._dense_aligned_n_plan(
                both.config_spec._cute_tcgen05_config, aligned_n_bf16_config().config
            )
            is None
        )


def test_plain_pair_prices_its_allocations_in_kernel_order():
    # The 224-wide pair's seventh AB stage beside two (128, 32) C stages: the
    # transposed product of bf16 32768x512x512 (column-major [512, 32768]
    # output) renders exactly the allocations the measured arena model
    # prices (``tcgen05_fixed_smem_overhead_bytes``: ptxas lays them out in
    # reverse), and nothing else: 231 596 of the 232 448-byte opt-in, where
    # the flat 2 KiB reserve priced 233 616.
    x, y, out = bf16_mm_args(32768, 512, 512)
    bound = plain_mm_into._bind_isolated((y.T, x.T, out.T))
    config = aligned_n_bf16_config(tcgen05_tile_n=224, tcgen05_ab_stages=7)
    source = bound.to_code(config)
    allocations = [
        line.strip().split(" = ")[0]
        + " "
        + line.split("alloc_smem(")[1].split(",")[0]
        + (
            " align" + line.split("alignment=")[1].rstrip(")")
            if "alignment=" in line
            else ""
        )
        for line in kernel_body(source).splitlines()
        if "cute.arch.alloc_smem(" in line
    ]
    assert allocations == [
        "tcgen05_tmem_holding_buf cutlass.Int32",
        "tcgen05_tmem_dealloc_mbar_ptr cutlass.Int64",
        "tcgen05_acc_pipeline_barriers cutlass.Int64",
        "smem_a cutlass.BFloat16 align128",
        "smem_b cutlass.BFloat16 align128",
        "tcgen05_ab_pipeline_mbars cutlass.Int64",
        "tcgen05_sD_ptr cutlass.BFloat16 align1024",
    ]
    assert "cutlass.Int32(14))" in source  # 2 x 7 AB barriers
    facts_ = Tcgen05PipelineSmemFacts(2, 2, 232448, output_column_major=True)

    def priced(*, plain_layout: bool) -> int:
        return pipeline_smem_bytes(
            facts_,
            bm=256,
            bn=224,
            bk=64,
            ab_stages=7,
            c_stages=2,
            acc_stages=2,
            plain_layout=plain_layout,
        )

    # 16 384 (C ring) + 128 (AB barriers) + 215 040 (AB ring) + 44
    assert priced(plain_layout=True) == 231596
    assert priced(plain_layout=False) == 233616
    ab, _ = wrapper_plans(source)
    assert ab["ab_stage_count"] == 7


def test_sixteen_bit_pair_stores_a_column_major_output():
    # The transposed product of a tall GEMM: out.T = b.T @ a.T for
    # bf16 16384x1024x4096 (a row-major, b column-major), whose output is a
    # column-major [1024, 16384] view. Its pairs run 256 rows of the 1024
    # and a physical width of the 16384: 224 makes 74 x 4 = 296 tiles, four
    # waves of the 76 SM pairs (256x256 makes 256 tiles, also four).
    x, y, out = bf16_mm_args(16384, 1024, 4096)
    bound = plain_mm_into._bind_isolated((y.T, x.T, out.T))
    state = bound.config_spec._cute_tcgen05_config
    dense_facts = state.dense_gemm_facts
    assert dense_facts is not None and dense_facts.output_column_major
    assert 224 in facts._dense_aligned_n_widths(state)
    config = aligned_n_bf16_config(tcgen05_tile_n=224, tcgen05_ab_stages=6)
    with bound.env:
        plan = facts._dense_aligned_n_plan(state, config.config)
        assert plan is not None
        assert plan.physical_tile == (256, 224, 64)
        # The column-major store keeps CuTe's with-source (128, 32) tile:
        # seven 30 KiB stages fit beside it (231 596 of the 232 448-byte
        # opt-in, ``pipeline_smem_bytes``' ``plain_layout``).
        assert plan.max_ab_stages == 7 and plan.store_subtile_n is None
        # The direct store (no D ring) stays row-major only; the search's
        # repair moves a sampled one to the normal store.
        direct = aligned_n_bf16_config(
            tcgen05_tile_n=224, tcgen05_ab_stages=6, tcgen05_c_store_mode="direct"
        )
        assert facts._dense_aligned_n_plan(state, direct.config) is None
        gen = ConfigGeneration(bound.config_spec)
        for _ in range(100):
            sample = gen.unflatten(gen.random_flat()).config
            if sample.get("tcgen05_tile_n"):
                assert sample.get("tcgen05_c_store_mode", "normal") == "normal"
    # Where a plain row-major store would take the narrow (128, 32) or the
    # (128, 48) subtile (192 and 240 over whole tiles of a 3072-wide N), the
    # column-major one is priced at CuTe's with-source tile, which codegen
    # renders for it: 192 fits six stages beside its (128, 64) tile, not the
    # seven the (128, 32) one leaves room for; 240 seven beside its
    # (128, 16) tile.
    x, y, out = bf16_mm_args(3072, 1024, 4096)
    whole = plain_mm_into._bind_isolated((y.T, x.T, out.T))
    whole_state = whole.config_spec._cute_tcgen05_config
    with whole.env:
        for width, stages in ((192, 6), (240, 7)):
            plan = facts._dense_aligned_n_plan(
                whole_state,
                aligned_n_bf16_config(tcgen05_tile_n=width, tcgen05_ab_stages=6).config,
            )
            assert plan is not None
            assert (plan.max_ab_stages, plan.store_subtile_n) == (stages, None)
        # One N tile per cluster: A multicast is the row-major store's only.
        assert (
            facts._dense_aligned_n_plan(
                whole_state,
                aligned_n_bf16_config(
                    tcgen05_tile_n=192, tcgen05_ab_stages=6, tcgen05_cluster_n=2
                ).config,
            )
            is None
        )
    source = bound.to_code(config)
    assert "LayoutEnum.COL_MAJOR" in source
    ab, d = wrapper_plans(source)
    assert (ab["bm"], ab["bn"], ab["bk"]) == (256, 224, 64)
    assert ab["ab_stage_count"] == 6
    assert (d["bm"], d["bn"]) == (256, 224)
    assert d.get("d_column_major") is True and d.get("epi_tile_n") is None


def test_sixteen_bit_pair_stages_its_last_tile_in_the_drained_a_ring():
    # A plain row-major 16-bit pair whose width 32 does not divide (176 =
    # 16 x 11) stores (128, 16) subtiles through a two-stage C ring. Its CTA's
    # last tile stages all eleven in the A ring, which that tile's MMAs have
    # drained, when the ring holds the tile (8 x 64 K columns >= 176):
    #
    #     if not tcgen05_epi_next_work.is_valid_tile:
    #         tcgen05_last_sD = cute.make_tensor(cute.recast_ptr(smem_a, ...), ...)
    #         ...
    #     else:
    #         <the C-ring loop>
    def render(m, n, k, *, transpose=False, **changes):
        x, y, out = bf16_mm_args(m, n, k)
        args = (y.T, x.T, out.T) if transpose else (x, y, out)
        source = plain_mm_into._bind_isolated(args).to_code(
            aligned_n_bf16_config(**changes)
        )
        return source, "if not tcgen05_epi_next_work.is_valid_tile:" in source

    source, staged = render(6272, 1024, 2048, tcgen05_tile_n=176, tcgen05_ab_stages=8)
    assert staged
    # The epilogue role looks one tile ahead on its own scheduler.
    assert "tcgen05_epi_next_work = " in source
    assert "cute.recast_ptr(smem_a, tcgen05_last_sD_layout.inner" in source
    # Two stages hold 128 of the 176 columns; the 80-wide pair's five
    # subtiles, the 48-column (144) and narrow (192 over whole N tiles)
    # subtiles and the column-major store keep the C ring alone.
    for shape, changes in (
        ((6272, 1024, 2048), {"tcgen05_tile_n": 176, "tcgen05_ab_stages": 2}),
        (
            (6272, 1024, 2048),
            {
                "tcgen05_tile_n": 80,
                "block_sizes": [256, 128, 64],
                "tcgen05_ab_stages": 8,
            },
        ),
        ((6144, 1152, 2048), {"tcgen05_tile_n": 144, "tcgen05_ab_stages": 6}),
        # (a deep C ring leaves 144 on (128, 16) subtiles, still the ring's)
        (
            (6144, 1152, 2048),
            {"tcgen05_tile_n": 144, "tcgen05_ab_stages": 8, "tcgen05_c_stages": 4},
        ),
        ((6144, 1152, 2048), {"tcgen05_tile_n": 192, "tcgen05_ab_stages": 6}),
    ):
        assert not render(*shape, **changes)[1]
    assert not render(
        2112, 1024, 2048, transpose=True, tcgen05_tile_n=176, tcgen05_ab_stages=6
    )[1]
    # An epilogue chain (two scalar FP32 scales: no auxiliary descriptor)
    # keeps the C ring too.
    a, b, scale_a, scale_b = scaled_args(6272, 1024, 512, mode=0, dtype=torch.bfloat16)
    source = scaled_mm._bind_isolated((a, b, scale_a, scale_b)).to_code(
        aligned_n_bf16_config(tcgen05_tile_n=176, tcgen05_ab_stages=8)
    )
    assert "CtaGroup.TWO, (256, 176)" in source
    assert "tcgen05_epi_next_work" not in source


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
        assert _seeded_widths(half) == {96, 192}
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
    # Each CTA stages 40 rows of an 80-wide K-major B tile, an odd count of
    # the swizzle's 8-row atoms: the TMA box is the stage-0 ``select`` of the
    # SMEM layout (``cute.slice_`` aborts on it). The seeds keep the widths
    # that are multiples of 32; the search also samples 80 and 240.
    odd = plain_mm_into._bind_isolated(bf16_mm_args(8192, 960, 768))
    source = odd.to_code(
        aligned_n_bf16_config(tcgen05_tile_n=80, block_sizes=[256, 128, 64])
    )
    assert "sB_tma_layout = cute.select(sB_layout, mode=[0, 1, 2])" in source
    assert "(256, 80)" in source
    # the host descriptor takes the device's form from the wrapper plan
    assert "'b_stage_select': True" in source
    assert "'b_stage_select'" not in odd.to_code(aligned_n_bf16_config())
    assert _seeded_widths(odd) == {96, 160, 192}
    assert facts._dense_aligned_n_widths(odd.config_spec._cute_tcgen05_config) == (
        80,
        96,
        160,
        192,
        240,
    )
    # MN-major B: each CTA stages half of an N tile as one SMEM row; only the
    # 192-wide pair's 96-column (192-byte) halves keep 64-byte swizzle atoms.
    mn_b = plain_mm_into._bind_isolated(bf16_mm_args(8192, 960, 768, rhs_k_major=False))
    assert _seeded_widths(mn_b) == {192}
    # 960 columns split into five 192-wide tiles: no N pairing.
    with pytest.raises(InvalidConfig, match="tcgen05_tile_n"):
        odd.to_code(aligned_n_bf16_config(tcgen05_cluster_n=2))


def test_seeds_compare_only_the_widths_they_take():
    # bf16 512x5632x2048: 176 divides N, but the seeds take multiples of 32
    # only, so 176 is no reference of their 15% filter: they keep the clipped
    # 160 and 192 and the default its 256x160 pair. The search samples the
    # seeded widths and 176.
    bound = plain_mm_into._bind_isolated(bf16_mm_args(512, 5632, 2048))
    state = bound.config_spec._cute_tcgen05_config
    assert _seeded_widths(bound) == {160, 192}
    assert bound.config_spec.default_config().config["tcgen05_tile_n"] == 160
    assert facts._dense_aligned_n_widths(state) == (160, 176, 192)


def test_n_major_b_keeps_widths_in_multiples_of_32():
    # Over an N-major B each CTA stages its half of the N tile as SMEM rows;
    # the pair keeps widths that are multiples of 32 there.
    bound = plain_mm_into._bind_isolated(
        bf16_mm_args(8192, 2304, 768, rhs_k_major=False)
    )
    with pytest.raises(InvalidConfig, match="multiple of 32 over an N-major B"):
        bound.to_code(aligned_n_bf16_config(tcgen05_tile_n=144))


def test_odd_width_pairs_store_48_column_subtiles():
    # A plain 16-bit pair of a width 48 divides and 32 does not stores
    # (128, 48) subtiles where CuTe's with-source rule halves to (128, 16),
    # where that C ring fits beside the configured AB ring; a deeper ring
    # keeps CuTe's tile and its AB depth (144 wide: four 48-wide C stages
    # leave seven AB stages, four 16-wide ones eight).
    wide = "tcgen05_store_epi_tile = (cute.make_layout(128), cute.make_layout(48))"
    cases = (
        ((1000, 2560, 2048), 144, 2, 8, True),
        ((1000, 2560, 2048), 144, 4, 7, True),
        ((1000, 2560, 2048), 144, 4, 8, False),
        ((4096, 2048, 2048), 240, 2, 6, True),
        ((1000, 2048, 2560), 112, 2, 9, False),
        ((6272, 1024, 1024), 176, 2, 8, False),
    )
    for shape, width, c_stages, ab_stages, takes_48 in cases:
        bound = plain_mm_into._bind_isolated(bf16_mm_args(*shape))
        config = aligned_n_bf16_config(
            tcgen05_tile_n=width,
            block_sizes=[256, 128 if width <= 128 else 256, 64],
            tcgen05_c_stages=c_stages,
            tcgen05_ab_stages=ab_stages,
        )
        state = bound.config_spec._cute_tcgen05_config
        with bound.env:
            plan = facts._dense_aligned_n_plan(state, config.config)
        assert plan is not None
        assert plan.store_subtile_n == (48 if takes_48 else None)
        source = bound.to_code(config)
        assert (wide in source) == takes_48
        assert ("'epi_tile_n': 48" in source) == takes_48
    # The ring CuTe's tile leaves stays the plan's limit at four C stages.
    bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 2560, 2048))
    with bound.env:
        deep = facts._dense_aligned_n_plan(
            bound.config_spec._cute_tcgen05_config,
            aligned_n_bf16_config(
                tcgen05_tile_n=144, tcgen05_c_stages=4, tcgen05_ab_stages=8
            ).config,
        )
    assert deep is not None and deep.max_ab_stages == 8


def test_one_cta_tiles_keep_the_stage_slice():
    # The stage-0 ``select`` is the 2-CTA pair's form for B halves of odd
    # 8-row atom counts; a one-CTA 64x8 tile (one 8-row atom) keeps the slice.
    a = torch.empty((4096, 4096), dtype=torch.bfloat16)
    b = torch.empty((8, 4096), dtype=torch.bfloat16).T
    bound = plain_mm._bind_isolated((a, b))
    source = bound.to_code(bound.config_spec.default_config())
    assert "sB_tma_layout = cute.slice_(sB_layout, (None, None, None, 0))" in source


def test_plain_256_wide_pair_takes_the_narrow_subtile_over_output_tails():
    # The 256x256 pair over a partial last M tile: bf16 25216x768x768 is 98.5
    # 256-row pairs over whole N and K tiles, every tile on the TMA store,
    # which clips the rows past M. Its plain store takes the (128, 32)
    # subtile, as the whole tiles of 25344x768x768 do; CuTe's with-source
    # rule would render (128, 64). So does a partial last N tile (1000
    # columns: 3.9 256-wide tiles), whose columns past N the TMA store clips
    # the same way, where a tile's epilogue hides behind the next tile's
    # mainloop: more tiles than SM pairs and a K loop past the tile's fixed
    # cost (twelve K blocks here; four, as 25344x1000x256 has, or one wave of
    # 64 tiles, as 4096x1000x2048 has, keep CuTe's tile). Only whole tiles
    # also take the no-unroll K loop; a K tail keeps CuTe's tile.
    config = helion.Config(
        block_sizes=[256, 256, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=6,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        l2_groupings=[16],
    )
    narrow = "tcgen05_epi_tile = (cute.make_layout(128), cute.make_layout(32))"
    for (m, n, k), takes_subtile, nounroll in (
        ((25344, 768, 768), True, True),
        ((25216, 768, 768), True, False),
        ((25344, 1000, 768), True, False),
        ((25216, 1000, 768), True, False),
        ((25344, 1000, 256), False, False),
        ((4096, 1000, 2048), False, False),
        ((25216, 768, 736), False, False),
    ):
        source = plain_mm_into._bind_isolated(bf16_mm_args(m, n, k)).to_code(config)
        assert (narrow in source) is takes_subtile, (m, n, k)
        assert ("'epi_tile_n': 32" in source) is takes_subtile, (m, n, k)
        assert ("compute_epilogue_tile_shape(" in source) is not takes_subtile
        assert ("unroll=1" in source) is nounroll, (m, n, k)


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


def test_sixteen_bit_pair_row_operands_keep_the_split_store():
    # The partial-output TMA store stages each row-vector operand per tile;
    # its arena model prices power-of-two tiles, not an aligned-N pair's own
    # ring (the 160-wide pair at eight AB stages leaves 864 B of the opt-in),
    # so any row operand -- an FP32 column scale here -- keeps the split
    # store over a partial last N tile (no staged row).
    x, y, out = bf16_mm_args(8192, 2000, 768)
    scale = torch.empty((2000,), dtype=torch.float32)
    bound = column_scale_mm_into._bind_isolated((x, y, scale, out))
    source = bound.to_code(
        aligned_n_bf16_config(tcgen05_tile_n=160, tcgen05_ab_stages=8)
    )
    assert "_BLOCK_SIZE_1 = 160" in source
    assert "tcgen05_aux_rowvec_smem" not in source


def test_a_short_k_loop_keeps_the_product_as_written() -> None:
    # bf16 32768x256x256 runs four 64-deep K blocks, no more than the fixed
    # prologue and epilogue the model charges per tile, so the model does not
    # rank its orientations (the same 128x256 CTA pair runs 1.04 of cuBLAS
    # as written and 0.98 transposed on GB300); 8192x512x512 (eight K blocks)
    # keeps its modeled gain for the transposed product.
    assert transposed_pair_tile_gain(32768, 256, 256, sm_pairs=76) == 1.0
    assert transposed_pair_tile_gain(8192, 512, 512, sm_pairs=76) > 1.08


def _search_range(bound, block_id: int) -> tuple[int, int, int]:
    """A block's floor for provided configs and its search range."""
    spec = bound.config_spec
    block = spec.block_sizes.block_id_lookup(block_id)
    fragment = block._fragment(spec)
    return block.min_size, fragment.low, fragment.high


def test_a_small_n_runs_one_zero_filled_tcgen05_tile() -> None:
    # The transposed GEMV of bf16 4x4096x4096 (out.T = b.T @ a.T, N = 4):
    # the search's N tile is the 8-column tcgen05 tile, whose edge TMA
    # zero-fills a.T's rows 4-7, not the scalar fallback's 4-column block,
    # over K tiles of at least 32 (a K-major B stage of 32-byte rows is
    # refused).
    a, b, out = bf16_mm_args(4, 4096, 4096)
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated((b.T, a.T, out.T))
        assert _search_range(bound, 1) == (4, 8, 8)
        assert _search_range(bound, 2) == (16, 32, 128)
        (plan, *_) = wrapper_plans(bound.to_code(bound.config_spec.default_config()))
        assert (plan["kind"], plan["bn"], plan["n_size"]) == ("tcgen05_ab_tma", 8, 4)
        # The block floor stays the scalar fallback's: a provided config
        # keeps its tiles, [64, 4, 16] the SIMT dot it ran before (raised to
        # 8 columns, its 16-deep K tile would make the refused B stage).
        code = bound.to_code(helion.Config(block_sizes=[64, 4, 16], pid_type="flat"))
        assert "_BLOCK_SIZE_1 = 4\n_BLOCK_SIZE_2 = 16\n" in code
        assert "cute.gemm(" not in code
    # A short K is no memory-bound GEMM: the 4-column block stays.
    a, b, out = bf16_mm_args(4, 4096, 64)
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated((b.T, a.T, out.T))
        assert _search_range(bound, 1) == (4, 4, 4)


def test_a_small_n_b_off_the_tensormap_rule_keeps_the_scalar_fallback() -> None:
    # The GEMV as written, bf16 4096x4x1024 with b stored [K, N]: its 8-byte
    # rows are off the 16-byte TensorMap rule, so the tile would load B on
    # the scalar shared-memory producer; the default and the search keep the
    # scalar fallback's SIMT dot. 16-byte rows (four columns of an
    # eight-column b) take the tile.
    a, b, out = bf16_mm_args(4096, 4, 1024, rhs_k_major=False)
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated((a, b, out))
        assert _search_range(bound, 1) == (4, 4, 4)
        assert "cute.gemm(" not in bound.to_code(bound.config_spec.default_config())
        b = torch.randn((1024, 8), dtype=torch.bfloat16)[:, :4]
        bound = plain_mm_into._bind_isolated((a, b, out))
        assert _search_range(bound, 1) == (4, 8, 8)
        (plan, *_) = wrapper_plans(bound.to_code(bound.config_spec.default_config()))
        assert (plan["kind"], plan["bn"]) == ("tcgen05_ab_tma", 8)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_k_mm_into(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, maximum: int
) -> None:
    m, k = a.shape
    n = b.shape[1]
    block_k = hl.register_block_size(16, maximum)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = acc.to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _k16_mm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k, block_size=16):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = acc.to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_n_mm_into(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, minimum: int, maximum: int
) -> None:
    m, k = a.shape
    n = b.shape[1]
    block_n = hl.register_block_size(minimum, maximum)
    for rows, cols in hl.tile((m, n), block_size=[None, block_n]):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = acc.to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _n4_mm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n), block_size=[None, 4]):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = acc.to(out.dtype)


def test_a_small_n_block_range_without_8_keeps_the_scalar_fallback() -> None:
    # bf16 4096x4x1024 over a K-major B with N registered at (4, 4) (block
    # ids N, M, K; the range bounds the search, and the block may size
    # allocations) or fixed at 4 (M, K): neither takes the 8-column tile,
    # and the search and the default stay the scalar fallback's, K tiles
    # from 16 deep included; registered up to 16, N takes the tile.
    a, b, out = bf16_mm_args(4096, 4, 1024)
    with gb300_target(), forbid_cuda_work():
        bound = _bounded_n_mm_into._bind_isolated((a, b, out, 4, 4))
        assert _search_range(bound, 0) == (4, 4, 4)
        assert _search_range(bound, 2) == (16, 16, 128)
        config = bound.config_spec.default_config()
        assert config.config["block_sizes"] == [4, 64, 64]
        assert "cute.gemm(" not in bound.to_code(config)
        bound = _n4_mm_into._bind_isolated((a, b, out))
        assert _search_range(bound, 2) == (16, 16, 128)
        assert "cute.gemm(" not in bound.to_code(bound.config_spec.default_config())
        bound = _bounded_n_mm_into._bind_isolated((a, b, out, 4, 16))
        assert _search_range(bound, 0) == (4, 8, 8)
        assert _search_range(bound, 2) == (16, 32, 128)
        (plan, *_) = wrapper_plans(bound.to_code(bound.config_spec.default_config()))
        assert (plan["kind"], plan["bn"]) == ("tcgen05_ab_tma", 8)


def test_a_small_n_k_block_under_64_bytes_keeps_the_scalar_fallback() -> None:
    # The transposed GEMV with K registered at 16 (block ids K, M, N) or
    # fixed at 16 (M, N): no K tile feeds the 8-column tile its K-major B,
    # so the search keeps the scalar fallback's 4-column tiles, which
    # render; registered up to 32, it takes the tile over 32-deep K tiles.
    a, b, out = bf16_mm_args(4, 4096, 4096)
    with gb300_target(), forbid_cuda_work():
        bound = _bounded_k_mm_into._bind_isolated((b.T, a.T, out.T, 16))
        assert _search_range(bound, 2) == (4, 4, 4)
        code = bound.to_code(helion.Config(block_sizes=[16, 64, 4], pid_type="flat"))
        assert "cute.gemm(" not in code
        bound = _k16_mm_into._bind_isolated((b.T, a.T, out.T))
        assert _search_range(bound, 1) == (4, 4, 4)
        assert "cute.gemm(" not in bound.to_code(bound.config_spec.default_config())
        bound = _bounded_k_mm_into._bind_isolated((b.T, a.T, out.T, 32))
        assert _search_range(bound, 0) == (16, 32, 32)
        assert _search_range(bound, 2) == (4, 8, 8)
        config = bound.config_spec.default_config()
        assert config.config["block_sizes"] == [32, 64, 8]
        (plan, *_) = wrapper_plans(bound.to_code(config))
        assert (plan["kind"], plan["bn"], plan["bk"]) == ("tcgen05_ab_tma", 8, 32)


def test_a_one_wave_product_no_pair_tiles_as_written_runs_transposed() -> None:
    # bf16 512x1000x1280: as written no CTA pair tiles N (1000 = 15.6 x 64:
    # the one-wave family's 128-row pairs take whole N tiles) and the
    # 256-row pairs give 8 tiles, so the flat one-CTA kernel runs it (0.86 of
    # cuBLAS on GB300); transposed (1000x512), 64 pairs of 128x64 do (0.96).
    assert (
        transposed_pair_tile_gain(512, 1000, 1280, sm_pairs=76)
        >= TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
    )
    assert transposed_pair_tile_gain(1000, 512, 1280, sm_pairs=76) < 1.0
    # The 128-row pairs count only where the search runs them: not under
    # one 128-row tile of M (4096x2x2048 transposed: 2 rows), nor on a grid
    # under a quarter of the SMs (128x1000x768 transposed: 16 pairs).
    for m, n, k in ((4096, 2, 2048), (128, 1000, 768)):
        assert (
            transposed_pair_tile_gain(m, n, k, sm_pairs=76)
            < TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
        )


def _transposed_args(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    """helion.inductor's transposed product of a bf16 RC ``m`` x ``n`` x ``k``
    mm: out.T (n x m, column-major) = b.T (n x k) @ a.T (k x m)."""
    a, b, out = bf16_mm_args(m, n, k)
    return b.T, a.T, out.T


def test_the_transposed_product_runs_the_swapped_128_row_pair() -> None:
    # bf16 1000x1024xK transposed (out.T 1024 x 1000): the one-wave pairs
    # clip N's last tile (1000 is no multiple of 64), the aligned-N family
    # runs the 128-row pair over the column-major output, and 72 pairs of
    # 128x112 (cuBLAS's swapped tile) put 144 CTAs on GB300's 152 SMs.
    bound = plain_mm_into._bind_isolated(_transposed_args(1000, 1024, 2048))
    state = bound.config_spec._cute_tcgen05_config
    assert state.cluster_m2_search_constraints.allow_one_wave_tiles
    assert facts._dense_aligned_n_block_ms(state) == (256, 128)
    # the 256-row pair ranks no clipped width ahead of its 64-wide tile; the
    # 128-row pair ranks 112 ahead of its 128-wide one
    assert facts._dense_aligned_n_clipped_widths(state) == frozenset()
    assert facts._dense_aligned_n_widths(state) == (112,)
    pairs = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("tcgen05_tile_n") == 112
    ]
    assert {
        (
            tuple(seed["block_sizes"]),
            seed["tcgen05_ab_stages"],
            seed["tcgen05_warp_spec_ab_load_warps"],
        )
        for seed in pairs
    } == {((128, 128, 64), 14, 1), ((128, 128, 64), 14, 2)}
    # The search keeps the sampled 128-row pair and its two A/B-load warps.
    gen = ConfigGeneration(bound.config_spec)
    two = next(seed for seed in pairs if seed["tcgen05_warp_spec_ab_load_warps"] == 2)
    _flat, normalized = gen.strict_config_pair(helion.Config(**two))
    assert normalized.config["block_sizes"][0] == 128
    assert normalized.config["tcgen05_warp_spec_ab_load_warps"] == 2
    # Its CTA's one tile stages its seven (64, 16) subtiles in the drained A
    # ring (fourteen 8 KB stages hold the 14 KB tile).
    code = bound.to_code(helion.Config(**two))
    assert "CtaGroup.TWO, (128, 112)" in code
    assert "if not tcgen05_epi_next_work.is_valid_tile:" in code
    assert "tcgen05_last_sD = cute.make_tensor(cute.recast_ptr(smem_a" in code


def test_the_last_tile_stages_over_a_column_major_output() -> None:
    def staged(args, config) -> bool:
        code = plain_mm_into._bind_isolated(args).to_code(config)
        return "if not tcgen05_epi_next_work.is_valid_tile:" in code

    pair256 = aligned_n_bf16_config(
        tcgen05_tile_n=112,
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=9,
    )
    pair128 = aligned_n_bf16_config(
        tcgen05_tile_n=112,
        block_sizes=[128, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=14,
    )
    # The 256-row pair over either output (1000x2048x2304 as written and
    # transposed); the 128-row pair over a column-major output, not over a
    # row-major one, whose (64, 16) rounds measured the same staged
    # (4864x224x1024: 0.937 -> 0.938 of cuBLAS, GB300).
    assert staged(bf16_mm_args(1000, 2048, 2304), pair256)
    assert staged(_transposed_args(1000, 2048, 2304), pair256)
    assert staged(_transposed_args(1000, 1024, 2048), pair128)
    assert not staged(bf16_mm_args(4864, 224, 1024), pair128)


def test_the_swapped_pair_flips_only_an_n_no_64_column_tile_divides() -> None:
    # The model prices the aligned-N 128-row pair over an N no 64-column
    # tile divides, where the family runs it (a column-major output: the
    # transposed side). bf16 1000x1024xK: as written 64 pairs of 128x128, 64
    # + 64 operand rows per CTA and K step; transposed 72 of 128x112, 64 + 56
    # (GB300, of cuBLAS: K=1024 0.950 -> 0.955, 2048 0.937 -> 0.957, 4096
    # 0.946 -> 0.970).
    for k in (1024, 2048, 2304, 2560, 3072, 4096):
        assert (
            transposed_pair_tile_gain(1000, 1024, k, sm_pairs=76)
            >= TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
        )
    # Whole 64-column N tiles keep their orientation: the transposed 128x176
    # (2048x768) and 128x224 (4096x512) pairs, and the as-written 128x240 of
    # 128x16384 (a row-major output wider than 256 runs no 128-row aligned
    # pair).
    assert (
        transposed_pair_tile_gain(2048, 768, 3072, sm_pairs=76)
        < TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
    )
    assert (
        transposed_pair_tile_gain(4096, 512, 384, sm_pairs=76)
        < TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
    )
    assert (
        transposed_pair_tile_gain(128, 16384, 4096, sm_pairs=76)
        >= TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
    )
    # Nor does an out.T whose M the search does not tile in 256-row pairs:
    # 1000x896 transposed (896 rows: an M ceiling of 128, no CTA pair runs
    # its N edge).
    assert (
        transposed_pair_tile_gain(1000, 896, 2048, sm_pairs=76)
        < TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
    )
    transposed = plain_mm_into._bind_isolated(_transposed_args(1000, 896, 2048))
    assert (
        transposed.config_spec._cute_tcgen05_config.cluster_m2_search_constraints
        is None
    )


def test_a_row_major_pair_under_a_128_row_ceiling_runs_128_rows() -> None:
    # bf16 25216x192x768: M is ragged for 256-row tiles and N for 128-wide
    # ones, so the search keeps M tiles of at most 128 rows. N (192) is itself
    # an aligned width whose carrier (256) is in the N bounds, and the family
    # runs it on the 128-row pair (64 rows per CTA): one tile per 128 rows
    # that reads A once (GB300: 1.03 of cuBLAS against 0.93 for the 128-row
    # pair over N tiles of 128 + 64).
    bound = plain_mm_into._bind_isolated(bf16_mm_args(25216, 192, 768))
    state = bound.config_spec._cute_tcgen05_config
    assert facts._matmul_search_bounds(state)[:2] == ((128, 128), (8, 256))
    assert facts._dense_aligned_n_block_m(state) == 128
    seeded = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("tcgen05_tile_n") == 192
    ]
    assert seeded
    assert {
        (
            tuple(seed["block_sizes"][:3]),
            seed["tcgen05_cta_group"],
            seed["tcgen05_ab_stages"],
        )
        for seed in seeded
    } == {((128, 256, 64), "two", 10)}
    source = bound.to_code(helion.Config(**seeded[0]))
    assert "CtaGroup.TWO, (128, 192)" in source
    assert "num_stages=10" in source
    # Its 64-row C tile stores CuTe's with-source subtile: the plan prices no
    # narrow subtile and refuses a configured one (codegen would reject it).
    with bound.env:
        plan = facts._dense_aligned_n_plan(state, dict(seeded[0]))
        configured = facts._dense_aligned_n_plan(
            state, {**seeded[0], "tcgen05_epi_subtile_n": 32}
        )
    assert plan is not None and plan.store_subtile_n is None
    assert configured is None
    # A wider N under the same ceiling keeps the 256-row family, which the
    # ceiling excludes: the 128-row pair runs only N that one tile spans.
    wide = plain_mm_into._bind_isolated(bf16_mm_args(25216, 1120, 768))
    wide_state = wide.config_spec._cute_tcgen05_config
    assert facts._matmul_search_bounds(wide_state)[0] == (128, 128)
    assert facts._dense_aligned_n_widths(wide_state) == ()
    assert not _seeded_widths(wide)
    # The transposed product (a column-major output, any N) runs the 128-row
    # pair under the 128-row M ceiling of its 192 rows too, its CTA's last
    # tile staged at once (``cute_mma``'s ``tcgen05_last_tile_staging``): the
    # search samples the widths the 128-row pair ranks ahead of its
    # references, and no width is seeded (none divides 25216 or clears the
    # seeds' margin).
    x = torch.empty((192, 768), dtype=torch.bfloat16)
    y = torch.empty((25216, 768), dtype=torch.bfloat16).T
    out = torch.empty((25216, 192), dtype=torch.bfloat16).T
    transposed = plain_mm_into._bind_isolated((x, y, out))
    transposed_state = transposed.config_spec._cute_tcgen05_config
    assert facts._dense_aligned_n_block_m(transposed_state) == 128
    assert facts._dense_aligned_n_widths(transposed_state) == (224, 240)
    assert not _seeded_widths(transposed)


def test_the_default_skips_power_of_two_tiles_wider_than_n() -> None:
    # bf16 32768x192x192: N's 256-wide aligned carrier also puts the plain
    # (256, 256) pair in the search. With a bias row (no aligned plan) its N
    # tile is wider than N, so its TMA store gives way to predicated SIMT
    # rows (GB300: 73.3 us against the flat (128, 16) default's 24.0); the
    # autotune-off default keeps the flat tile. The plain GEMM takes the
    # 256x192 aligned pair (7.0 us against the flat tile's 23.4).
    x, y, out = bf16_mm_args(32768, 192, 192)
    bias = torch.empty((192,), dtype=torch.bfloat16)
    biased = bias_mm_into._bind_isolated((x, y, bias, out)).config_spec
    assert biased.default_config().config["block_sizes"][:2] == [128, 16]
    assert biased.default_config().config["pid_type"] == "flat"
    plain = plain_mm_into._bind_isolated((x, y, out)).config_spec
    assert plain.default_config().config["block_sizes"][:3] == [256, 256, 64]
    assert plain.default_config().config["tcgen05_tile_n"] == 192


def test_a_transposed_double_edge_off_the_pairs_keeps_the_model_neutral() -> None:
    # A product ragged in both M and N against the 256x256 pair tile is a
    # double edge either way round, which the search runs on the 256-row
    # pairs only where no 64-row one-CTA tile divides its M. bf16
    # 264x2496x1024: as written M = 264 (4 x 64 + 8) runs them; transposed,
    # M = 2496 (39 x 64) runs only one-CTA tiles, so the model, which prices
    # the pairs, keeps it as written (it modeled 1.03).
    def pairs(args: tuple[torch.Tensor, ...]) -> bool:
        bound = plain_mm_into._bind_isolated(args)
        return (
            bound.config_spec._cute_tcgen05_config.cluster_m2_search_constraints
            is not None
        )

    with gb300_target(), forbid_cuda_work():
        a, b, out = bf16_mm_args(264, 2496, 1024)
        assert pairs((a, b, out))
        assert not pairs((b.T, a.T, out.T))
    assert transposed_double_edge_without_pairs(264, 2496)
    assert transposed_pair_tile_gain(264, 2496, 1024, sm_pairs=76) == 1.0
    # Whole pair tiles on either side make no double edge.
    assert not transposed_double_edge_without_pairs(256, 2496)
    assert not transposed_double_edge_without_pairs(264, 2560)
    assert transposed_pair_tile_gain(16384, 1024, 4096, sm_pairs=76) > 1.1
