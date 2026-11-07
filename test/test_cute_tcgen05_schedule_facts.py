"""CPU admission/codegen proofs for exact tiles within non-power-of-two shapes.

Also covers the paired TMA K-tail protocol and the Blackwell-only native proofs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics.cute_tcgen05_default import (
    CuteTcgen05DefaultTileHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_tcgen05_schedule import (
    CuteTcgen05ScheduleHeuristic,
)
from helion._compiler.cute.dense_gemm import prove_dense_gemm_facts
from helion._compiler.cute.one_cta_pipeline import prove_one_cta_pipeline_facts
from helion._compiler.cute.tcgen05_config import _TWO_CTA_PERSISTENT_PID_VETO
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from helion._compiler.cute.tcgen05_schedule_facts import Tcgen05FullTileScheduleFacts
from helion._compiler.cute.tcgen05_schedule_facts import analyze_native_gemm
from helion._compiler.cute.tcgen05_schedule_facts import full_tile_schedule_supported
from helion._compiler.cute.tcgen05_schedule_facts import native_tma_schedule_supported
from helion._compiler.cute.tcgen05_schedule_facts import (
    paired_k_tail_schedule_supported,
)
from helion._compiler.cute.tcgen05_schedule_facts import prove_full_tile_schedule_facts
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import blackwell_capability
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import extra_store_mm
from helion._testing.cute_dense_tcgen05 import plain_mm
from helion._testing.cute_dense_tcgen05 import scaled_args
from helion._testing.cute_dense_tcgen05 import scaled_mm
from helion._testing.cute_dense_tcgen05 import vector_scaled_mm
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _bind(shape: tuple[int, int, int]) -> BoundKernel:
    return vector_scaled_mm._bind_isolated(scaled_args(*shape))


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
@pytest.mark.parametrize(
    ("shape", "tile", "cluster_m"),
    (
        ((128, 5120, 2880), (128, 128, 64), 1),
        ((4096, 2880, 4096), (128, 64, 128), 1),
        ((256, 2048, 4096), (128, 128, 128), 2),
    ),
)
def test_exact_tile_is_reachable_and_emits_native_pipeline(
    shape: tuple[int, int, int],
    tile: tuple[int, int, int],
    cluster_m: int,
    capability: tuple[int, int],
) -> None:
    with blackwell_capability(capability):
        bound = _bind(shape)
        spec = bound.config_spec
        state = spec._cute_tcgen05_config
        assert spec.target_device_capability == capability
        assert state.full_tile_schedule_facts is not None
        assert "persistent_interleaved" in spec.allowed_pid_types
        config = helion.Config(
            block_sizes=list(tile),
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=cluster_m,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
        )
        with bound.env:
            generator = ConfigGeneration(spec)
            _flat, canonical = generator.canonicalize_flat(generator.flatten(config))
            assert canonical.block_sizes == list(tile)
            assert canonical.config["pid_type"] == "persistent_interleaved"
            assert canonical.config["tcgen05_cluster_m"] == cluster_m
            seeds = state.full_tile_schedule_seed_configs()
            assert seeds
            assert all(
                state._full_tile_schedule_config_supported(seed.config)
                for seed in seeds
            )
        source = bound.to_code(config)
        # The search-normalized schedule emits the same kernel.
        assert bound.to_code(canonical) == source
    assert "PipelineTmaUmma.create" in source or "make_tma_umma_pipeline(" in source
    assert "PipelineTmaStore.create" in source
    assert "StaticPersistentTileScheduler.create" in source
    assert f"CtaGroup.{'TWO' if cluster_m == 2 else 'ONE'}" in source


def _paired_fp8_config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 128, 128],
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        pid_type="persistent_blocked",
        tcgen05_ab_stages=13,
        tcgen05_c_stages=2,
        tcgen05_acc_stages=2,
    )


@pytest.mark.parametrize("capability", (None, (9, 0), (10, 1), (11, 0), (12, 0)))
def test_other_capability_cannot_reuse_native_proofs(
    capability: tuple[int, int] | None,
) -> None:
    """Every dense schedule family's proof is limited to SM100 and SM103."""
    bound = scaled_mm._bind_isolated(scaled_args(256, 2048, 4096))
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    paired = _paired_fp8_config().config

    def proofs() -> tuple[object, ...]:
        gemm = analyze_native_gemm(bound.env, bound.host_function.device_ir)
        return (
            prove_full_tile_schedule_facts(gemm),
            prove_one_cta_pipeline_facts(bound.env, gemm),
            prove_dense_gemm_facts(bound.env, gemm),
            state._paired_pipeline_parameters(paired),
        )

    # Positive control: the native target proves every family.
    assert None not in proofs()
    with patch.object(spec, "target_device_capability", capability):
        assert proofs() == (None, None, None, None)


@pytest.mark.parametrize("disabled", (False, True))
def test_registration_is_idempotent_even_without_heuristic_seeds(
    disabled: bool,
) -> None:
    kernel = helion.kernel(
        vector_scaled_mm.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        disable_autotuner_heuristics=disabled,
    )
    bound = kernel._bind_isolated(scaled_args(128, 5120, 2880))
    state = bound.config_spec._cute_tcgen05_config
    assert state.full_tile_schedule_facts is not None
    previous = (state.full_tile_schedule_facts, bound.config_spec.allowed_pid_types)
    CuteTcgen05ScheduleHeuristic.register_facts(
        bound.env, bound.host_function.device_ir
    )
    assert (
        state.full_tile_schedule_facts,
        bound.config_spec.allowed_pid_types,
    ) == previous
    if disabled:
        assert not bound.config_spec.compiler_seed_configs


def test_unaligned_operand_does_not_restore_persistence() -> None:
    shape = (128, 5120, 2880)
    args = list(scaled_args(*shape))
    storage = torch.empty(shape[0] * shape[2] + 1, dtype=torch.float8_e4m3fn)
    args[0] = storage[1:].view(shape[0], shape[2])
    state = vector_scaled_mm._bind_isolated(
        tuple(args)
    ).config_spec._cute_tcgen05_config
    assert not state.matmul_operands_tma_provable
    assert state.full_tile_schedule_facts is None


def test_extra_output_keeps_original_admission() -> None:
    args = scaled_args(128, 5120, 2880)
    state = extra_store_mm._bind_isolated(args[:2]).config_spec._cute_tcgen05_config
    assert state.full_tile_schedule_facts is None


@pytest.mark.parametrize(
    ("shape", "tile"),
    (
        ((128, 5120, 2880), (128, 128, 128)),
        ((4096, 2880, 4096), (128, 128, 128)),
    ),
)
def test_new_persistent_search_does_not_admit_tail_samples(
    shape: tuple[int, int, int], tile: tuple[int, int, int]
) -> None:
    bound = _bind(shape)
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        config = helion.Config(
            block_sizes=list(tile),
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
        )
        _flat, canonical = generator.canonicalize_flat(generator.flatten(config))
        assert canonical.config["pid_type"] == "flat"


@pytest.mark.parametrize(
    ("tile", "cluster_m", "cluster_n"),
    (
        ((128, 128, 128), 1, 1),  # K tail.
        ((256, 128, 64), 2, 1),  # M tail.
        ((128, 128, 64), 2, 2),  # Unproved multicast topology.
        ((128, 64, 64), 2, 1),  # Unproved narrow paired instruction.
        ((128, 128, 0), 1, 1),
        ((128, 128, 64), True, 1),
    ),
)
def test_common_predicate_rejects_unproved_schedules(
    tile: tuple[int, int, int], cluster_m: int, cluster_n: int
) -> None:
    facts = Tcgen05FullTileScheduleFacts(
        (0, 1, 2), (128, 5120, 2880), torch.float8_e4m3fn
    )
    assert not full_tile_schedule_supported(
        facts, tile, cluster_m=cluster_m, cluster_n=cluster_n
    )


def _unregistered_full_tile(
    *reasons: str | None,
) -> tuple[BoundKernel, Tcgen05FullTileScheduleFacts]:
    """Undo the full-tile registration, then veto both persistent pid types."""
    bound = _bind((128, 5120, 2880))
    state = bound.config_spec._cute_tcgen05_config
    facts = state.full_tile_schedule_facts
    assert facts is not None
    state.full_tile_schedule_facts = None
    state.full_tile_restored_pid_types = ()
    state.full_tile_added_pairing = False
    for pid in ("persistent_interleaved", "persistent_blocked"):
        for reason in reasons:
            bound.config_spec.disallow_pid_type(pid, reason=reason)
    return bound, facts


def _assert_owned_veto_is_lifted() -> None:
    """Positive control: registration lifts the CuTe-owned veto alone."""
    bound, facts = _unregistered_full_tile(_TWO_CTA_PERSISTENT_PID_VETO)
    state = bound.config_spec._cute_tcgen05_config
    state.register_full_tile_schedule_facts(facts)
    assert state.full_tile_schedule_facts == facts
    assert "persistent_interleaved" in bound.config_spec.allowed_pid_types


def test_unrelated_pid_restriction_is_not_resurrected() -> None:
    _assert_owned_veto_is_lifted()
    bound, facts = _unregistered_full_tile("another loop forbids persistence")
    state = bound.config_spec._cute_tcgen05_config
    state.register_full_tile_schedule_facts(facts)
    assert state.full_tile_schedule_facts is None
    assert "persistent_interleaved" not in bound.config_spec.allowed_pid_types


@pytest.mark.parametrize("second_reason", (None, "another loop forbids persistence"))
def test_later_independent_pid_veto_is_not_hidden_by_first_reason(
    second_reason: str | None,
) -> None:
    _assert_owned_veto_is_lifted()
    bound, facts = _unregistered_full_tile(_TWO_CTA_PERSISTENT_PID_VETO, second_reason)
    for pid in ("persistent_interleaved", "persistent_blocked"):
        assert (
            bound.config_spec.disallowed_pid_type_reasons[pid]
            == _TWO_CTA_PERSISTENT_PID_VETO
        )
        assert bound.config_spec.disallowed_pid_type_causes[pid] == {
            _TWO_CTA_PERSISTENT_PID_VETO,
            second_reason,
        }
    state = bound.config_spec._cute_tcgen05_config
    state.register_full_tile_schedule_facts(facts)
    assert state.full_tile_schedule_facts is None
    assert "persistent_interleaved" not in bound.config_spec.allowed_pid_types
    assert "persistent_blocked" not in bound.config_spec.allowed_pid_types


# Search and codegen preserve the existing paired TMA K-tail protocol.


@pytest.mark.parametrize("shape", ((128, 5120, 2880), (256, 512, 576)))
def test_paired_k_tail_survives_search_and_emits_tma(
    shape: tuple[int, int, int],
) -> None:
    bound = _bind(shape)
    state = bound.config_spec._cute_tcgen05_config
    facts = state.full_tile_schedule_facts
    assert facts is not None and facts.rhs_k_major
    config = helion.Config(
        block_sizes=[128, 128, 128],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
    )
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        _flat, canonical = generation.canonicalize_flat(generation.flatten(config))
        assert canonical.block_sizes == config.block_sizes
        assert canonical.config["pid_type"] == "persistent_interleaved"
        assert canonical.config["tcgen05_cluster_m"] == 2
        seeds = state.full_tile_schedule_seed_configs()
        assert any(
            seed.block_sizes == config.block_sizes
            and seed.config["tcgen05_cluster_m"] == 2
            for seed in seeds
        )
    source = bound.to_code(canonical)
    # The split-barrier-init prologue builds the pipeline through the runtime
    # helper instead of the DSL constructor.
    assert "PipelineTmaUmma.create" in source or "make_tma_umma_pipeline(" in source
    assert "PipelineTmaStore.create" in source
    assert "StaticPersistentTileScheduler.create" in source
    assert "CtaGroup.TWO" in source


@pytest.mark.parametrize(
    ("shape", "tile", "cluster_m", "cluster_n", "rhs_k_major"),
    (
        ((128, 5120, 2880), (128, 128, 128), 1, 1, True),
        ((128, 5120, 2880), (128, 128, 128), 2, 2, True),
        ((128, 5120, 2880), (128, 128, 128), 2, 1, False),
        ((128, 5120, 64), (128, 128, 128), 2, 1, True),
        ((128, 5120, 128), (128, 128, 128), 2, 1, True),
        ((128, 5120, 2880), (256, 128, 128), 2, 1, True),
        ((128, 5168, 2880), (128, 128, 128), 2, 1, True),
        ((128, 5120, 2880), (128, 64, 128), 2, 1, True),
        ((128, 5120, 2880), (128, 128, 0), 2, 1, True),
        ((128, 5120, 65568), (128, 128, 128), 2, 1, True),
    ),
)
def test_unproved_k_tail_schedules_stay_excluded(
    shape: tuple[int, int, int],
    tile: tuple[int, int, int],
    cluster_m: int,
    cluster_n: int,
    rhs_k_major: bool,
) -> None:
    facts = Tcgen05FullTileScheduleFacts(
        (0, 1, 2), shape, torch.float8_e4m3fn, rhs_k_major=rhs_k_major
    )
    assert not paired_k_tail_schedule_supported(
        facts, tile, cluster_m=cluster_m, cluster_n=cluster_n
    )


def test_k_tail_does_not_widen_other_instruction_families() -> None:
    facts = Tcgen05FullTileScheduleFacts(
        (0, 1, 2), (128, 5120, 2880), torch.float8_e4m3fn, rhs_k_major=True
    )
    assert native_tma_schedule_supported(
        facts, (128, 128, 128), cluster_m=2, cluster_n=1
    )
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert not native_tma_schedule_supported(
            replace(facts, dtype=dtype), (128, 128, 128), cluster_m=2, cluster_n=1
        )


def _plain_bind(
    m: int, n: int, k: int, dtype: torch.dtype = torch.bfloat16
) -> BoundKernel:
    return plain_mm._bind_isolated(
        (torch.empty((m, k), dtype=dtype), torch.empty((n, k), dtype=dtype).T)
    )


def _k_block_max(bound: BoundKernel) -> int:
    fragments = bound.config_spec._cute_tcgen05_config._matmul_block_fragments()
    assert fragments is not None
    return fragments[2].high


def test_output_edge_admits_two_cta_pairs_and_flattens_one_cta_partial_tiles() -> None:
    """Ragged M, whole K: persistent and CtaGroup.TWO search on 64-deep K tiles.

    One-CTA persistent kernels run whole tiles only (the host multi-tile
    guard), so the search flattens their partial-tile samples while the
    two-CTA pairs keep their persistent schedule.
    """
    bound = _plain_bind(1000, 4096, 2880)
    state = bound.config_spec._cute_tcgen05_config
    assert "persistent_interleaved" in bound.config_spec.allowed_pid_types
    constraints = state.cluster_m2_search_constraints
    assert constraints is not None and not constraints.allow_edge_k_tail_family
    assert state.output_edge_persistent_admitted
    assert _k_block_max(bound) == 64
    one_cta = {
        "block_sizes": [128, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
    }
    state._fix_cluster_m1_persistent_search_config(one_cta)
    assert one_cta["pid_type"] == "flat"
    pair = {
        "block_sizes": [256, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
    }
    state._fix_cluster_m1_persistent_search_config(pair)
    assert pair["pid_type"] == "persistent_interleaved"


@pytest.mark.parametrize(
    ("shape", "dtype"),
    (
        # Ragged M below the 256-row pair tile.
        ((16, 14336, 4096), torch.bfloat16),
        # Ragged in both M and N.
        ((3000, 3000, 3072), torch.bfloat16),
        # FP8 keeps its exact-tile restoration.
        ((1000, 4096, 2880), torch.float8_e4m3fn),
    ),
)
def test_output_edge_admission_excludes_unpaired_and_double_edge_shapes(
    shape: tuple[int, int, int], dtype: torch.dtype
) -> None:
    bound = _plain_bind(*shape, dtype=dtype)
    assert not bound.config_spec._cute_tcgen05_config.output_edge_persistent_admitted


def test_k_cap_applies_only_where_whole_k_tiles_admit_persistence() -> None:
    """200x1000x2880 is ragged in both M and N: no persistent schedule, no cap."""
    capped = _plain_bind(1000, 4096, 2880)
    uncapped = _plain_bind(200, 1000, 2880)
    assert _k_block_max(capped) == 64
    assert _k_block_max(uncapped) > 64


def test_m_edge_two_cta_ab6_keeps_tma_store_and_drains_after_dealloc() -> None:
    """1000x4096x2880 bf16 at 2x1 256x256x64 ab=6 keeps the TMA-store epilogue.

    The arena (192 KiB AB ring, 32 KiB (128, 64) C ring and the barriers) fits
    the opt-in capacity, which the AB-gate-budget check refused, and the two-CTA
    teardown frees TMEM before draining the bulk stores.
    """
    bound = plain_mm._bind_isolated(
        (
            torch.empty((1000, 2880), dtype=torch.bfloat16),
            torch.empty((4096, 2880), dtype=torch.bfloat16).T,
        )
    )
    source = bound.to_code(
        helion.Config(
            block_sizes=[256, 256, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=6,
            tcgen05_c_stages=2,
        )
    )
    assert "'kind': 'tcgen05_d_tma'" in source
    assert "'ab_stage_count': 6" in source
    assert "cute.copy(tcgen05_tma_store_atom" in source
    assert source.count("tcgen05_c_pipeline.producer_tail()") == 1
    assert source.rindex("tcgen05_tmem_allocator.free(") < source.index(
        "tcgen05_c_pipeline.producer_tail()"
    )


def _default_with_sms(shape: tuple[int, int, int], num_sms: int) -> helion.Config:
    with patch(
        "helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=num_sms
    ):
        return _plain_bind(*shape).config_spec.default_config()


def test_default_tile_one_cta_tile_quantizes_m_into_whole_waves() -> None:
    """25216 rows: 197 one-CTA 128x256 rows of tiles fill 4 waves of 148 SMs;
    the 256-row pair tile leaves a fifth, nearly empty wave."""
    config = _default_with_sms((25216, 768, 3072), 148)
    assert config.block_sizes == [128, 256, 64]
    assert config.config["tcgen05_cluster_m"] == 1
    assert config.config["pid_type"] == "persistent_interleaved"


def test_default_tile_pair_replaces_ffi_on_a_partial_wave() -> None:
    """8192x768x3072: 96 FFI pair tiles are 1.3 waves; the 256x128 pair wins,
    with the FFI controls off and the grouped raster for its several waves."""
    config = _default_with_sms((8192, 768, 3072), 148)
    assert config.block_sizes == [256, 128, 64]
    assert config.config["tcgen05_cluster_m"] == 2
    assert not config.config.get("tcgen05_tvm_ffi_launch")
    assert config.config.get("tcgen05_layout_strategy") == "default"
    assert config.config["l2_groupings"] == [TCGEN05_TWO_CTA_SEED_L2_GROUPING]


def test_default_tile_keeps_ffi_on_whole_waves() -> None:
    """The direct-entry default runs six 64-deep AB stages."""
    config = _default_with_sms((4096, 4096, 4096), 148)
    assert config.config.get("tcgen05_tvm_ffi_launch") is True
    assert config.block_sizes == [256, 256, 64]
    assert config.config["tcgen05_ab_stages"] == 6
    assert config.config["tcgen05_c_stages"] == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mm_k128(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_k = hl.register_block_size(128, 128)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


def test_direct_entry_seed_deepens_64_deep_k_tiles() -> None:
    """K=576 takes only 64-deep tiles: six stages instead of three."""
    seed = _plain_bind(
        8192, 4096, 576
    ).config_spec._tcgen05_full_tile_direct_entry_seed_config()
    assert seed is not None
    assert seed.block_sizes == [256, 256, 64]
    assert seed.config["tcgen05_ab_stages"] == 6
    assert seed.config["tcgen05_c_stages"] == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mm_three_outputs(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = a.shape
    n = b.shape[1]
    out0 = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    out1 = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    out2 = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out0[rows, cols] = acc.to(torch.bfloat16)
        out1[rows, cols] = (acc * 2.0).to(torch.bfloat16)
        out2[rows, cols] = (acc + 1.0).to(torch.bfloat16)
    return out0, out1, out2


def test_direct_entry_seed_stays_shallow_when_the_c_rings_do_not_fit() -> None:
    """K=576 with three outputs: 192 KiB of AB plus three 16 KiB C rings
    overflow the arena, so the 64-deep seed keeps three stages."""
    bound = _mm_three_outputs._bind_isolated(
        (
            torch.empty((8192, 576), dtype=torch.bfloat16),
            torch.empty((4096, 576), dtype=torch.bfloat16).T,
        )
    )
    seed = bound.config_spec._tcgen05_full_tile_direct_entry_seed_config()
    assert seed is not None
    assert seed.block_sizes == [256, 256, 64]
    assert seed.config["tcgen05_ab_stages"] == 3


def test_direct_entry_seed_keeps_a_declared_k_block_size() -> None:
    """A K fragment pinned to 128 keeps the three-stage 128-deep seed."""
    bound = _mm_k128._bind_isolated(
        (
            torch.empty((4096, 1024), dtype=torch.bfloat16),
            torch.empty((4096, 1024), dtype=torch.bfloat16).T,
        )
    )
    seed = bound.config_spec._tcgen05_full_tile_direct_entry_seed_config()
    assert seed is not None
    # The registered K block size is block 0; the M/N tile follows.
    assert seed.block_sizes == [128, 256, 256]
    assert seed.config["tcgen05_ab_stages"] == 3


def test_default_tile_promotion_survives_ffi_normalization() -> None:
    """A one-CTA pick over an FFI-eligible shape keeps its tile: the promoted
    values turn the FFI controls off instead of relying on the winning seed."""
    config = _default_with_sms((5376, 1792, 4096), 147)
    assert config.block_sizes == [128, 256, 64]
    assert not config.config.get("tcgen05_tvm_ffi_launch")


def test_default_tile_leaves_small_grids_to_their_seeds() -> None:
    """Fewer 128x128 tiles than SMs: per-CTA latency decides, no promotion."""
    bound = _plain_bind(1024, 1024, 4096)
    with bound.env:
        assert CuteTcgen05DefaultTileHeuristic._pick(bound.env) is None
