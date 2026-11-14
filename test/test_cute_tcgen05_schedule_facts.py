"""CPU admission/codegen proofs for exact tiles within non-power-of-two shapes.

Also covers the paired TMA K-tail protocol and the Blackwell-only native proofs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteTcgen05ThreadLocalEpilogueHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_tcgen05_default import (
    CuteTcgen05DefaultTileHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_tcgen05_schedule import (
    CuteTcgen05ScheduleHeuristic,
)
from helion._compiler.cute.dense_gemm import prove_dense_gemm_facts
from helion._compiler.cute.one_cta_pipeline import prove_one_cta_pipeline_facts
from helion._compiler.cute.tcgen05_config import _TWO_CTA_PERSISTENT_PID_VETO
from helion._compiler.cute.tcgen05_config import facts
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import smem
from helion._compiler.cute.tcgen05_config import surface
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
from helion._testing.cute_aux import rank_two_aux
from helion._testing.cute_aux import two_residual_aux
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import full_and_edge_store_bodies
from helion._testing.cute_dense_tcgen05 import atomic_output_mm
from helion._testing.cute_dense_tcgen05 import bias_mm_into
from helion._testing.cute_dense_tcgen05 import blackwell_capability
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import extra_store_mm
from helion._testing.cute_dense_tcgen05 import plain_mm
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion._testing.cute_dense_tcgen05 import promoted_bias_mm_into
from helion._testing.cute_dense_tcgen05 import scale_bias_mm_into
from helion._testing.cute_dense_tcgen05 import scaled_args
from helion._testing.cute_dense_tcgen05 import scaled_mm
from helion._testing.cute_dense_tcgen05 import user_bound_mm
from helion._testing.cute_dense_tcgen05 import vector_scaled_mm
from helion._testing.cute_inductor_epilogues import INDUCTOR_EPILOGUES
from helion._testing.cute_inductor_epilogues import dropout_mm_into
from helion._testing.cute_inductor_epilogues import gelu_tanh_grad_mm_into
from helion._testing.cute_inductor_epilogues import inductor_epilogue_args
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


CPU_DEVICE = "cpu"

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
            seed_configs = seeds.full_tile_schedule_seed_configs(state)
            assert seed_configs
            assert all(
                repair._full_tile_schedule_config_supported(state, seed.config)
                for seed in seed_configs
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
            prove_dense_gemm_facts(gemm),
            seeds._paired_pipeline_parameters(state, paired),
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
    bound, schedule_facts = _unregistered_full_tile(_TWO_CTA_PERSISTENT_PID_VETO)
    state = bound.config_spec._cute_tcgen05_config
    facts.register_full_tile_schedule_facts(state, schedule_facts)
    assert state.full_tile_schedule_facts == schedule_facts
    assert "persistent_interleaved" in bound.config_spec.allowed_pid_types


def test_unrelated_pid_restriction_is_not_resurrected() -> None:
    _assert_owned_veto_is_lifted()
    bound, schedule_facts = _unregistered_full_tile("another loop forbids persistence")
    state = bound.config_spec._cute_tcgen05_config
    facts.register_full_tile_schedule_facts(state, schedule_facts)
    assert state.full_tile_schedule_facts is None
    assert "persistent_interleaved" not in bound.config_spec.allowed_pid_types


@pytest.mark.parametrize("second_reason", (None, "another loop forbids persistence"))
def test_later_independent_pid_veto_is_not_hidden_by_first_reason(
    second_reason: str | None,
) -> None:
    _assert_owned_veto_is_lifted()
    bound, schedule_facts = _unregistered_full_tile(
        _TWO_CTA_PERSISTENT_PID_VETO, second_reason
    )
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
    facts.register_full_tile_schedule_facts(state, schedule_facts)
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
        seed_configs = seeds.full_tile_schedule_seed_configs(state)
        assert any(
            seed.block_sizes == config.block_sizes
            and seed.config["tcgen05_cluster_m"] == 2
            for seed in seed_configs
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


@helion.kernel(backend="cute")
def _addmm_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _relu_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc).to(x.dtype)
    return out


def _plain_bind(
    m: int, n: int, k: int, dtype: torch.dtype = torch.bfloat16
) -> BoundKernel:
    return plain_mm._bind_isolated(
        (torch.empty((m, k), dtype=dtype), torch.empty((n, k), dtype=dtype).T)
    )


def _k_block_max(bound: BoundKernel) -> int:
    fragments = facts._matmul_block_fragments(bound.config_spec._cute_tcgen05_config)
    assert fragments is not None
    return fragments[2].high


def test_output_edge_admits_two_cta_pairs_and_flattens_one_cta_partial_tiles() -> None:
    """Ragged M, whole K: persistent and CtaGroup.TWO search on 64-deep K tiles.

    One-CTA persistent kernels do not run ragged-M tiles (the host
    multi-tile guard; only N-edge tiles take the one-CTA partial-output
    store), so the search flattens their partial-tile samples while the
    two-CTA pairs keep their persistent schedule.
    """
    bound = _plain_bind(1000, 4096, 2880)
    state = bound.config_spec._cute_tcgen05_config
    assert "persistent_interleaved" in bound.config_spec.allowed_pid_types
    constraints = state.cluster_m2_search_constraints
    assert constraints is not None and not constraints.allow_edge_k_tail_family
    assert state.output_edge_persistent_admitted
    assert _k_block_max(bound) == 64
    # the FFI direct-entry seed assumes whole 256x256 tiles
    assert not seeds.full_tile_direct_entry_seed_eligible(state)
    one_cta = {
        "block_sizes": [128, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
    }
    repair._fix_cluster_m1_persistent_search_config(state, one_cta)
    assert one_cta["pid_type"] == "flat"
    pair = {
        "block_sizes": [256, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
    }
    repair._fix_cluster_m1_persistent_search_config(state, pair)
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


def test_n_edge_cluster_n_seeds_take_one_n_cluster() -> None:
    """4096x1000x4096 with an N-major B: the CtaGroup.TWO seeds' cluster_n=2
    variants (L2 grouped, or with a scheduler warp) fall outside the proved
    N-edge multicast schedule, which codegen requires; normalization takes
    them to one N cluster, and they render."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((4096, 4096), dtype=torch.bfloat16),
                torch.empty((4096, 1000), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        generation = spec.create_config_generation()
        paired = [
            generation.unflatten(generation.flatten(seed))
            for seed in spec.compiler_seed_configs
            if seed.config.get("tcgen05_cluster_n") == 2
        ]
        assert paired
        assert all(seed.config["tcgen05_cluster_n"] == 1 for seed in paired)
        bound.to_code(paired[0])


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
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=num_sms,
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
    """8192x768x3072: 96 FFI pair tiles are 1.3 waves; the aligned-N 192-wide
    pair (128 tiles, under two waves; 0.95 -> 1.03 of cuBLAS against the
    256x128 pair) wins, with the FFI controls off and the grouped raster for
    its several waves."""
    config = _default_with_sms((8192, 768, 3072), 148)
    assert config.block_sizes == [256, 256, 64]
    assert config.config["tcgen05_tile_n"] == 192
    assert config.config["tcgen05_cluster_m"] == 2
    assert not config.config.get("tcgen05_tvm_ffi_launch")
    assert config.config.get("tcgen05_layout_strategy") == "default"
    assert config.config["l2_groupings"] == [TCGEN05_TWO_CTA_SEED_L2_GROUPING]


def test_default_tile_counts_partial_tiles() -> None:
    """1000x2560x2048: 40 256x256 pair tiles counting the partial last row.
    30 whole ones fell below the quarter-wave gate: only the one-wave tiles,
    which need whole M tiles, stayed on, no two-CTA seed survived and the GEMM
    defaulted to the flat 128x16 kernel (0.11 of cuBLAS). Counted, the
    256x256 families seed again, and one wave of 64 aligned-N 256x160 pairs
    (0.93) beats one of 40 256x256 pairs (0.72) as the default."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _plain_bind(1000, 2560, 2048)
        state = bound.config_spec._cute_tcgen05_config
        constraints = state.cluster_m2_search_constraints
        assert constraints is not None and not constraints.one_wave_only
        with bound.env:
            assert any(
                seed.config.get("block_sizes") == [256, 256, 64]
                for seed in seeds.autotune_seed_configs(state)
            )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [256, 256, 64]
    assert config.config["tcgen05_tile_n"] == 160
    assert config.config["pid_type"] == "persistent_interleaved"


def test_default_tile_keeps_the_256x128_pair() -> None:
    """8192x1408x3072 seeds no aligned-N width (none divides 1408 or clears
    the clipped widths' gain), so the default stays the 256x128 pair."""
    config = _default_with_sms((8192, 1408, 3072), 148)
    assert config.block_sizes == [256, 128, 64]
    assert not config.config.get("tcgen05_tile_n")


def test_default_tile_mn_major_b_takes_192() -> None:
    """MN-major B seeds only the 192-wide aligned-N pair (whole 64-byte
    swizzle rows): 20008x768x2048 defaults to it."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((20008, 2048), dtype=torch.bfloat16),
                torch.empty((2048, 768), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [256, 256, 64]
    assert config.config["tcgen05_tile_n"] == 192


def test_default_tile_k_major_b_leaves_widths_under_128() -> None:
    """K-major B at 16384x960x1024 seeds the 96- and 192-wide aligned-N
    pairs; a width under 128 is no candidate of the wave-quantized default
    (the 96 pair ran 40-42 us against 38), so it takes the 192 pair."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((16384, 1024), dtype=torch.bfloat16),
                torch.empty((960, 1024), dtype=torch.bfloat16).T,
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [256, 256, 64]
    assert config.config["tcgen05_tile_n"] == 192


def test_default_tile_mn_major_b_skips_widths_off_whole_swizzle_rows() -> None:
    """MN-major B at 8008x2560x4096: each CTA's 80-column (160-byte) half of
    the 160-wide pair's B rows is not a whole 64-byte swizzle row, so the
    default stays the 256x256 pair (the 160 pair ran 0.74 of cuBLAS)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((8008, 4096), dtype=torch.bfloat16),
                torch.empty((4096, 2560), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [256, 256, 64]
    assert "tcgen05_tile_n" not in config.config


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _three_f32_rows_residual_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``acc`` plus three FP32 rows and a residual."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        acc = acc + r0[cols] + r1[cols] + r2[cols] + residual[rows, cols]
        out[rows, cols] = acc.to(out.dtype)


def test_rows_past_the_arena_keep_the_split_store() -> None:
    """Three FP32 rows (3 KiB of a 256-wide tile) do not fit beside a
    256x256 ab=6 pair: the output-edge kernel keeps the full tiles on the
    TMA store and the fringe tiles on the SIMT store, the residual read on
    both."""
    m, n, k = 8192, 9536, 768
    row = torch.empty((n,), dtype=torch.float32)
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _three_f32_rows_residual_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                row,
                row,
                row,
                torch.empty((m, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[256, 256, 64],
            pid_type="persistent_interleaved",
            tcgen05_persistence_model="static_persistent",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=6,
            tcgen05_c_stages=2,
            tcgen05_aux_load_placement="post_acc_wait",
        )
        code = bound.to_code(config)
    assert "tcgen05_tma_store_atom" in code
    assert "cute.nvgpu.CopyR2GOp()" in code


@pytest.mark.parametrize(
    "shape",
    (
        # 32 x 38 pair tiles: 17 per CTA pair
        (8192, 9536, 768),
        # 32 x 9: 4 per pair
        (8192, 2112, 768),
        # 26 tiles per pair, two K tiles
        (4096, 30000, 128),
        # 6 tiles per pair, two K tiles
        (32768, 576, 128),
    ),
)
def test_default_tile_output_edge_bias_takes_the_partial_output_store(
    shape: tuple[int, int, int],
) -> None:
    """An output-edge bias default stores every tile through TMA at any tile
    count (``post_acc_wait`` at the seed's ring, the row staged per tile,
    one tile ahead, through a predicated copy) instead of keeping fringe
    tiles, or every tile, on the SIMT store: Inductor corpus + bias on B200,
    of the ATen GEMM, 32768x576x192 0.50 -> 1.02, 25216x2304x768 0.69 ->
    1.02, 4096x30000x128 0.79 -> 0.90 (1.06 with the row one tile ahead)."""
    m, n, k = shape
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = bias_mm_into._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
    assert config.block_sizes == [256, 256, 64]
    assert config.config["tcgen05_aux_load_placement"] == "post_acc_wait"
    assert config.config["tcgen05_ab_stages"] == 6
    assert "pred=tcgen05_aux_rowvec_smem_0_pf_pred_row_lookahead)" in code
    assert "tcgen05_tma_store_atom" in code
    assert "cute.nvgpu.CopyR2GOp()" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _three_row_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``acc`` plus three unpromoted BF16 rows."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        acc = acc + r0[cols] + r1[cols] + r2[cols]
        out[rows, cols] = acc.to(out.dtype)


def test_rows_without_a_second_slot_keep_the_synchronous_copy() -> None:
    """Three BF16 rows fit beside a 256x256 ab=6 pair in one slot each (1.5
    KiB of stages: every tile stays on the partial-output TMA store) but not
    in the two slots of the one-tile-ahead copy (3 KiB), so the epilogue
    copies them at the top of each tile instead."""
    m, n, k = 8192, 9536, 768
    row = torch.empty((n,), dtype=torch.bfloat16)
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _three_row_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                row,
                row,
                row,
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        values = config.config
        bm, bn, bk = cast("list[int]", values["block_sizes"])
        fits = [
            smem.partial_output_row_stage_fits(
                bound.config_spec._cute_tcgen05_config,
                values,
                bm=bm,
                bn=bn,
                bk=bk,
                cluster_m=cast("int", values["tcgen05_cluster_m"]),
                ab_stages=cast("int", values["tcgen05_ab_stages"]),
                c_stages=cast("int", values["tcgen05_c_stages"]),
                slots=slots,
            )
            for slots in (1, 2)
        ]
        code = bound.to_code(config)
    assert values["block_sizes"] == [256, 256, 64]
    assert values["tcgen05_cluster_m"] == 2
    assert values["tcgen05_ab_stages"] == 6
    assert fits == [True, False]
    assert "cute.nvgpu.CopyR2GOp()" not in code
    assert "row_lookahead" not in code
    assert "pred=tcgen05_aux_rowvec_pred_0" in code


def test_partial_output_rows_load_one_tile_ahead() -> None:
    """On the persistent partial-output store the epilogue role loads its
    staged rows one tile ahead: a lookahead copy of its tile scheduler issues
    the next tile's ``cp.async`` row copies (one group per tile, two rows
    here) into the other of two SMEM slots before this tile waits for its own
    (``cp.async.wait_group 1``); the epilogue barrier then publishes them,
    without a fence. A flat walk (one tile per CTA) has nothing to load
    ahead."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        m, n, k = 2048, 3000, 128
        bound = scale_bias_mm_into._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        config = spec.default_config()
        code = bound.to_code(config)
        flat = helion.Config(
            **{
                **config.config,
                "pid_type": "flat",
                "block_sizes": [128, 256, 64],
                "tcgen05_cluster_m": 1,
            }
        )
        spec.normalize(flat, _fix_invalid=True)
        flat_code = bound.to_code(flat)
    assert config.config["pid_type"] == "persistent_interleaved"
    # two rows, each issued ahead of the loop and at the top of every tile
    assert code.count("cute.nvgpu.cpasync.CopyG2SOp()") == 4
    assert code.count("cute.arch.cp_async_commit_group()") == 2
    assert code.count("cute.arch.cp_async_wait_group(1)") == 1
    assert code.count("make_layout((2, 256), stride=(256, 1))") == 2
    assert code.count("tcgen05_tma_store_role_tile % cutlass.Int32(2), None]") == 2
    assert "fence_acq_rel_cta" not in code
    assert flat.config["pid_type"] == "flat"
    assert "row_lookahead" not in flat_code
    assert "cp_async" not in flat_code


def test_long_n_sweep_defaults_group_by_32() -> None:
    """A multi-wave pair default walks groups of 32 M tiles on a long N sweep
    whatever grouping its seed carried (a plain output-edge GEMM's seed walks
    groups of 4; the staged grouping has its own test,
    ``test_staged_residual_long_sweep_groups_eight_m_tiles``)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        plain = plain_mm_into._bind_isolated(
            (
                torch.empty((16384, 768), dtype=torch.bfloat16),
                torch.empty((768, 30528), dtype=torch.bfloat16),
                torch.empty((16384, 30528), dtype=torch.bfloat16),
            )
        ).config_spec.default_config()
    assert plain.block_sizes == [256, 256, 64]
    assert plain.config["l2_groupings"] == [32]


def test_m_edge_without_a_reachable_pair_keeps_the_flat_search() -> None:
    """An M ragged for every one-CTA tile height runs persistent only
    through the CtaGroup.TWO pairs: bf16 1000x128x2880 has four pairs, under
    the quarter-wave gate, so the search stays flat with its 128-deep K
    tiles (no whole-K cap), and a persistent one-CTA config is repaired to
    the flat pid type. An M of whole 128-row tiles (1152x1792x4096, 35
    pairs) keeps the persistent one-CTA tiles, as do an M of whole 64-row
    tiles (1088x128x2880) and an N edge over whole 64-row tiles
    (64x9000x4096)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound, whole_rows, half_rows, n_edge = (
            plain_mm._bind_isolated(
                (
                    torch.empty((m, k), dtype=torch.bfloat16),
                    torch.empty((k, n), dtype=torch.bfloat16),
                )
            )
            for m, n, k in (
                (1000, 128, 2880),
                (1152, 1792, 4096),
                (1088, 128, 2880),
                (64, 9000, 4096),
            )
        )
        spec = bound.config_spec
        persistent = {
            "block_sizes": [128, 128, 64],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
        }
        spec.normalize(persistent, _fix_invalid=True)
    assert spec.allowed_pid_types == ("flat",)
    assert persistent["pid_type"] == "flat"
    view = facts._matmul_block_fragments(spec.tcgen05)
    assert view is not None
    assert view[2].high == 128
    for other in (whole_rows, half_rows, n_edge):
        assert "persistent_interleaved" in other.config_spec.allowed_pid_types


@pytest.mark.parametrize("column_major", [False, True])
def test_output_wider_than_n_stores_stop_at_the_gemm_extents(
    column_major: bool,
) -> None:
    """A 640-wide output written over N = 576, row- or column-major: the
    edge stores stop at the GEMM's (M, N), not the destination's shape,
    whose columns past N other writers own. The TMA store's TensorMap views
    the GEMM's extents (``d_extents``) on the pair's partial-output store
    and on the one-CTA N-edge tiles, and the SIMT edge predicate reads N."""
    m, n, k = 25600, 576, 192
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty((640, m), dtype=torch.bfloat16).T
        if column_major
        else torch.empty((m, 640), dtype=torch.bfloat16),
    )
    pair = helion.Config(
        block_sizes=[256, 256, 64],
        l2_groupings=[1],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_ab_stages=4,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
    )
    one_cta = helion.Config(
        block_sizes=[128, 256, 64],
        l2_groupings=[1],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_ab_stages=3,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
    )
    flat = helion.Config(block_sizes=[128, 256, 64], l2_groupings=[1])
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm_into._bind_isolated(args)
        for config in (pair, one_cta):
            code = bound.to_code(config)
            assert f"'d_extents': [{m}, {n}]" in code
        code = bound.to_code(flat)
    if not column_major:
        assert f"cute.elem_less(_coord, ({m}, {n}))" in code


def test_output_edge_bias_default_tile_order() -> None:
    """An output-edge bias GEMM walks groups of 32 M tiles on a long N sweep
    and of 4 on a short one (the plain raster where the group is as tall as
    the M tiles). A sampled persistent tile order survives the search
    projection; the swizzle goes back to 1 where nothing runs it: on a flat
    walk, and under ``tcgen05_cluster_n=2``, whose edge schedule refuses
    it."""
    defaults = {}
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        for m, n, k in (
            (32768, 8128, 768),
            (32768, 9536, 768),
            (8192, 9536, 768),
            (8192, 2112, 768),
        ):
            bound = bias_mm_into._bind_isolated(
                (
                    torch.empty((m, k), dtype=torch.bfloat16),
                    torch.empty((k, n), dtype=torch.bfloat16),
                    torch.empty((n,), dtype=torch.bfloat16),
                    torch.empty((m, n), dtype=torch.bfloat16),
                )
            )
            config = bound.config_spec.default_config().config
            defaults[m, n] = (config["l2_groupings"], config["tcgen05_l2_swizzle_size"])
        # an M edge only, where two-cluster N samples stay
        bound = bias_mm_into._bind_isolated(
            (
                torch.empty((8000, 768), dtype=torch.bfloat16),
                torch.empty((768, 8192), dtype=torch.bfloat16),
                torch.empty((8192,), dtype=torch.bfloat16),
                torch.empty((8000, 8192), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        sampled = {}
        configs = {}
        for name, tile in (
            ("persistent", {"pid_type": "persistent_interleaved"}),
            # a one-CTA flat tile (a flat pair goes back to the pair schedule)
            (
                "flat",
                {
                    "pid_type": "flat",
                    "block_sizes": [128, 128, 64],
                    "tcgen05_cluster_m": 1,
                },
            ),
            (
                "cluster_n",
                {"pid_type": "persistent_interleaved", "tcgen05_cluster_n": 2},
            ),
        ):
            config = helion.Config(
                **{
                    **spec.default_config().config,
                    **tile,
                    "l2_groupings": [4],
                    "tcgen05_l2_swizzle_size": 4,
                }
            )
            spec.normalize(config, _fix_invalid=True)
            configs[name] = config
            sampled[name] = (
                config.config["pid_type"],
                config.config.get("tcgen05_cluster_n", 1),
                config.config["l2_groupings"],
                config.config["tcgen05_l2_swizzle_size"],
            )
        # the cluster_n sample compiles (its edge schedule refuses a swizzle),
        # as does a plain GEMM's on the same M edge
        bound.to_code(configs["cluster_n"])
        plain = plain_mm_into._bind_isolated(
            (
                torch.empty((8000, 768), dtype=torch.bfloat16),
                torch.empty((768, 8192), dtype=torch.bfloat16),
                torch.empty((8000, 8192), dtype=torch.bfloat16),
            )
        )
        plain_config = helion.Config(
            **{
                **plain.config_spec.default_config().config,
                "pid_type": "persistent_interleaved",
                "tcgen05_cluster_n": 2,
                "tcgen05_l2_swizzle_size": 4,
            }
        )
        plain.config_spec.normalize(plain_config, _fix_invalid=True)
        plain.to_code(plain_config)
    # 128 M tiles over 32 and 38 N tiles; 32 M tiles (a group of 32 is the
    # plain raster); 9 N tiles (a short sweep) over 32 M tiles
    assert defaults == {
        (32768, 8128): ([32], 1),
        (32768, 9536): ([32], 1),
        (8192, 9536): ([1], 1),
        (8192, 2112): ([4], 1),
    }
    assert sampled == {
        "persistent": ("persistent_interleaved", 1, [4], 4),
        "flat": ("flat", 1, [4], 1),
        "cluster_n": ("persistent_interleaved", 2, [4], 1),
    }
    assert plain_config.config["tcgen05_cluster_n"] == 2
    assert plain_config.config["tcgen05_l2_swizzle_size"] == 1


@pytest.mark.parametrize("kernel", (bias_mm_into, promoted_bias_mm_into))
def test_direct_entry_bias_default_stages_the_row(kernel: helion.Kernel) -> None:
    """Whole tiles keep the direct-entry pair, now with the promoted bias row
    staged per tile (``pre_acc_wait`` at the seed's own rings)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = kernel._bind_isolated(
            (
                torch.empty((8192, 768), dtype=torch.bfloat16),
                torch.empty((768, 2304), dtype=torch.bfloat16),
                torch.empty((2304,), dtype=torch.bfloat16),
                torch.empty((8192, 2304), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
    assert config.config["tcgen05_tvm_ffi_launch"] is True
    assert config.config["tcgen05_ab_stages"] == 6
    assert config.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
    assert "tcgen05_aux_rowvec_smem" in code


def test_direct_entry_drops_a_staged_fp32_row() -> None:
    """The direct-entry projection keeps a sampled ``pre_acc_wait`` only for
    promoted rows: an FP32 bias row (read per subtile, never staged by the
    defaults) comes back without the placement."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = bias_mm_into._bind_isolated(
            (
                torch.empty((8192, 768), dtype=torch.bfloat16),
                torch.empty((768, 2304), dtype=torch.bfloat16),
                torch.empty((2304,), dtype=torch.float32),
                torch.empty((8192, 2304), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        default = spec.default_config()
        sampled = helion.Config(
            **{**default.config, "tcgen05_aux_load_placement": "pre_acc_wait"}
        )
        spec.normalize(sampled, _fix_invalid=True)
    assert default.config["tcgen05_tvm_ffi_launch"] is True
    assert "tcgen05_aux_load_placement" not in default.config
    assert sampled.config["tcgen05_tvm_ffi_launch"] is True
    assert sampled.config.get("tcgen05_aux_load_placement") is None


def test_one_cta_n_edge_samples_stay_persistent_on_the_tma_store() -> None:
    """A one-CTA persistent sample whose partial tiles are N-edge tiles
    (25216x576: 128x256 tiles end 64 columns into the third) stays
    persistent where register auxiliary loads take the partial-output TMA
    store, and goes flat where they do not: ``pre_acc_wait``, the ``direct``
    store (no TMA epilogue), 64-row tiles (1024x576, where they are
    searchable: that body does not lay out their register auxiliary
    fragment), a column-major output (nor its auxiliary reads), and operands
    TMA cannot load (N=580: a 1160-byte row stride)."""
    projected = {}
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        for name, m, n, column_major, overrides in (
            ("post", 25216, 576, False, {}),
            ("pre", 25216, 576, False, {"tcgen05_aux_load_placement": "pre_acc_wait"}),
            ("direct", 25216, 576, False, {"tcgen05_c_store_mode": "direct"}),
            ("64 rows", 1024, 576, False, {"block_sizes": [64, 256, 64]}),
            ("column-major", 25216, 576, True, {}),
            ("no TMA", 8192, 580, False, {}),
        ):
            out = (
                torch.empty((n, m), dtype=torch.bfloat16).T
                if column_major
                else torch.empty((m, n), dtype=torch.bfloat16)
            )
            bound = bias_mm_into._bind_isolated(
                (
                    torch.empty((m, 192), dtype=torch.bfloat16),
                    torch.empty((192, n), dtype=torch.bfloat16),
                    torch.empty((n,), dtype=torch.bfloat16),
                    out,
                )
            )
            spec = bound.config_spec
            config = spec.default_config()
            config.config.update(
                block_sizes=[128, 256, 64],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=1,
                tcgen05_ab_stages=3,
                tcgen05_aux_load_placement="post_acc_wait",
            )
            config.config.update(overrides)
            spec.normalize(config, _fix_invalid=True)
            projected[name] = (config.config["block_sizes"], config.config["pid_type"])
            if name == "post":
                code = bound.to_code(config)
    assert projected == {
        "post": ([128, 256, 64], "persistent_interleaved"),
        "pre": ([128, 256, 64], "flat"),
        "direct": ([128, 256, 64], "flat"),
        "64 rows": ([64, 256, 64], "flat"),
        "column-major": ([128, 256, 64], "flat"),
        "no TMA": ([128, 256, 64], "flat"),
    }
    assert "tcgen05_tma_store_atom" in code
    assert "cute.nvgpu.CopyR2GOp()" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_output_bias_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
) -> None:
    """One biased accumulator stored to two outputs (no store facts)."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        value = acc + bias[cols].to(torch.float32)
        out0[rows, cols] = value.to(out0.dtype)
        out1[rows, cols] = value.to(out1.dtype)


@pytest.mark.parametrize("aux_load_mode", ("simt", "tma", "epi_tma"))
def test_one_cta_n_edge_sample_is_judged_on_its_final_epilogue(
    aux_load_mode: str,
) -> None:
    """A persistent one-CTA N-edge sample keeps its persistence through the
    auxiliary fixups that change its load mode (a sampled ``tma`` load ends
    on ``simt``): the partial-output store is judged on the final config, so
    the three modes leave the same persistent kernel."""
    m, n, k = 25216, 576, 1024
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = rank_two_aux._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        values = {
            **bound.config_spec.default_config().config,
            "block_sizes": [128, 256, 64],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_ab_stages": 3,
            "tcgen05_aux_load_placement": "post_acc_wait",
            "tcgen05_aux_load_mode": aux_load_mode,
        }
        bound.config_spec.normalize(values, _fix_invalid=True)
    assert values["pid_type"] == "persistent_interleaved"
    assert values["block_sizes"] == [128, 256, 64]


def test_small_grid_default_takes_the_128_row_pair() -> None:
    """A GEMM with fewer 128x128 tiles than SMs defaults to the paired-pipeline
    seeds' 128-row pair that fills the most SMs in one wave: 128x9216x4096
    + bias runs 72 128x128 pairs, 144 CTAs (B200: 0.19 of the ATen GEMM on
    the flat 64x16 base default, 1.15 on the pair)."""
    m, n, k = 128, 9216, 4096
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = bias_mm_into._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [128, 128, 64]
    assert config.config["tcgen05_cluster_m"] == 2
    assert config.config["pid_type"] != "flat"


def test_boolean_mask_rides_no_staged_ring() -> None:
    """A boolean mask stays on the direct read (``stages_through_smem_ring``):
    it is an exact-shape operand for the register mechanisms, but no staged
    ring carries it, so neither ``epi_tma`` nor the producer warp's ``tma``
    ring is searched or seeded (their seeds failed codegen), an explicit
    ``tma`` config is refused, and the default keeps SIMT loads."""
    m, n, k = 8192, 1024, 1024
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = dropout_mm_into._bind_isolated(
            inductor_epilogue_args(
                next(e for e in INDUCTOR_EPILOGUES if e.kernel is dropout_mm_into),
                m,
                n,
                k,
                device=CPU_DEVICE,
                random=False,
            )
        )
        spec = bound.config_spec
        state = spec._cute_tcgen05_config
        facts = state.pipeline_smem_facts
        config = spec.default_config()
        explicit = dict(config.config, tcgen05_aux_load_mode="tma")
        with bound.env, pytest.raises(InvalidConfig, match="producer's ring carries"):
            spec.normalize(explicit)
    assert facts is not None
    assert facts.exact_aux_itemsizes == (1,)
    assert facts.ring_aux_itemsizes == ()
    assert not surface.epi_tma_search_enabled(state)
    assert not surface._aux_tma_search_enabled(state)
    assert all(
        seed.config.get("tcgen05_aux_load_mode", "simt") == "simt"
        for seed in spec.compiler_seed_configs
    )
    assert config.config.get("tcgen05_aux_load_mode", "simt") == "simt"


def test_math_bound_residual_pair_default_keeps_simt_loads() -> None:
    """A residual read through a transcendental keeps the SIMT loads: staging
    it buys no latency the math does not already hide (GELU backward,
    16384x4096x1024: 0.33 of the ATen GEMM on SIMT loads, 0.29 staged)."""
    m, n, k = 8192, 1024, 1024
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = gelu_tanh_grad_mm_into._bind_isolated(
            inductor_epilogue_args(
                next(
                    e for e in INDUCTOR_EPILOGUES if e.kernel is gelu_tanh_grad_mm_into
                ),
                m,
                n,
                k,
                device=CPU_DEVICE,
                random=False,
            )
        )
        facts = bound.config_spec._cute_tcgen05_config.pipeline_smem_facts
        config = bound.config_spec.default_config()
    assert facts is not None
    assert facts.epilogue_transcendental
    assert config.config["tcgen05_cluster_m"] == 2
    assert config.config["tcgen05_aux_load_mode"] == "simt"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_mm_narrow_n_blocks(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """A residual GEMM whose N block size is registered at 16-64."""
    m, k = a.shape
    n = b.shape[1]
    block_n = hl.register_block_size(16, 64)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[None, block_n]):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = (acc + residual[rows, cols].to(torch.float32)).to(
            torch.bfloat16
        )
    return out


def test_narrow_n_pair_keeps_a_registered_n_range() -> None:
    """N = 128 runs the 256 x 128 pair, unless the kernel registered an N
    block size range without 128: the two-CTA search stays off and a
    sampled pair keeps its N tile inside the range."""
    m, n, k = 19000, 128, 256
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _residual_mm_narrow_n_blocks._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        assert spec.tcgen05.cluster_m_search_choices == (1,)
        # block 0 is the registered N block size, then M and K
        sample = helion.Config(
            block_sizes=[64, 256, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
        )
        spec.normalize(sample, _fix_invalid=True)
        bound.to_code(sample)
    assert 16 <= sample.config["block_sizes"][0] <= 64


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_mm_m_blocks(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """A residual GEMM whose M block size is registered at 16-128."""
    m, k = a.shape
    n = b.shape[1]
    block_m = hl.register_block_size(16, 128)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[block_m, None]):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = (acc + residual[rows, cols].to(torch.float32)).to(
            torch.bfloat16
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_mm_k_blocks(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """A residual GEMM whose K block size is registered at 64-128."""
    m, k = a.shape
    n = b.shape[1]
    block_k = hl.register_block_size(64, 128)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile([m, n]):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = (acc + residual[rows, cols].to(torch.float32)).to(
            torch.bfloat16
        )
    return out


def _bind_residual(kernel: helion.Kernel, m: int, n: int, k: int):
    return kernel._bind_isolated(
        (
            torch.empty((m, k), dtype=torch.bfloat16),
            torch.empty((k, n), dtype=torch.bfloat16),
            torch.empty((m, n), dtype=torch.bfloat16),
        )
    )


def test_registered_block_sizes_shape_the_pair_defaults() -> None:
    """An M block size registered at 16-128 keeps N = 128 off the 256 x 128
    pair's search; a K block size registered at 64-128 keeps the M-edge
    pair (1000x4096x4096 + residual) on 64-deep k-blocks instead of losing
    it to the flat base tile."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        narrow = _bind_residual(_residual_mm_m_blocks, 19000, 128, 256)
        assert narrow.config_spec.tcgen05.cluster_m_search_choices == (1,)
        edge = _bind_residual(_residual_mm_k_blocks, 1000, 4096, 4096)
        config = edge.config_spec.default_config()
        edge.to_code(config)
    # block 0 is the registered K block size, then M and N
    assert config.block_sizes == [64, 256, 256]
    assert config.config["tcgen05_cluster_m"] == 2


def test_mask_only_producer_warp_keeps_any_ab_depth() -> None:
    """A mask-only kernel's C-input warp stages nothing (a mask rides no
    ring), so the strict pass leaves its AB depth to codegen, as before
    masks counted as exact-shape operands: dropout at 8192x1024x1024 keeps
    three AB stages beside the warp."""
    (dropout,) = [e for e in INDUCTOR_EPILOGUES if e.name == "dropout"]
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        spec = dropout.kernel._bind_isolated(
            inductor_epilogue_args(
                dropout, 8192, 1024, 1024, device=CPU_DEVICE, random=False
            )
        ).config_spec
        config = {
            "block_sizes": [256, 256, 64],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_warp_spec_c_input_warps": 1,
            "tcgen05_ab_stages": 3,
        }
        spec.normalize(config)
    assert config["tcgen05_ab_stages"] == 3
    assert config["tcgen05_warp_spec_c_input_warps"] == 1


def test_m_edge_pair_keeps_two_ab_stages_over_one_k_block() -> None:
    """The M-edge pair default over a single k-block (30000x128x64) keeps a
    second AB stage, which loads the pair's next tile while the MMA reads
    this one."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        config = _plain_bind(
            30000, 128, 64, torch.bfloat16
        ).config_spec.default_config()
    assert config.block_sizes == [256, 128, 64]
    assert config.config["tcgen05_cluster_m"] == 2
    assert config.config["tcgen05_ab_stages"] == 2


def test_narrow_edge_seed_takes_a_two_stage_aux_ring() -> None:
    """The edge family's narrow CLC aux-TMA seed (256x128x256 at two AB
    stages, 4000x4000x4008 + residual) searches two and three aux stages;
    three overflow beside its 256-deep ring, so a three-stage sample takes
    two and keeps the producer warp, TMA loads and CLC persistence."""
    args = (
        torch.empty((4000, 4008), dtype=torch.bfloat16),
        torch.empty((4008, 4000), dtype=torch.bfloat16),
        torch.empty((4000, 4000), dtype=torch.bfloat16),
    )
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        spec = rank_two_aux._bind_isolated(args).config_spec
        (seed,) = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.config["block_sizes"][:3] == [256, 128, 256]
        ]
        sample = {**seed.config, "tcgen05_aux_stages": 3}
        spec.normalize(sample, _fix_invalid=True)
    assert sample["tcgen05_aux_stages"] == 2
    assert sample["block_sizes"][:3] == [256, 128, 256]
    assert sample["tcgen05_warp_spec_c_input_warps"] == 1
    assert sample["tcgen05_aux_load_mode"] == "tma"
    assert sample["tcgen05_persistence_model"] == "clc_persistent"


@pytest.mark.parametrize(
    ("bk", "l2_groupings"),
    ((16, [1]), (64, [4])),
    ids=("k16", "grouped"),
)
def test_n_edge_cluster_n_samples_repair_to_compiling_configs(
    bk: int, l2_groupings: list[int]
) -> None:
    """A persistent 2x2-cluster sample over an N edge (16384x384x768) outside
    the proved N-multicast schedule (a 16-deep K tile; a grouped tile order)
    repairs to one the codegen runs, instead of one it refuses (``tcgen05
    N-edge cluster_n=2 requires ...``)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((16384, 768), dtype=torch.bfloat16),
                torch.empty((768, 384), dtype=torch.bfloat16),
            )
        )
        sample = helion.Config(
            block_sizes=[256, 256, bk],
            l2_groupings=l2_groupings,
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=4,
            tcgen05_c_stages=2,
        )
        bound.config_spec.normalize(sample, _fix_invalid=True)
        bound.to_code(sample)


@pytest.mark.parametrize(
    "move",
    ({"tcgen05_c_store_mode": "direct"}, {"tcgen05_warp_spec_c_input_warps": 1}),
)
def test_epi_tma_default_neighbours_repair_to_valid_configs(
    move: dict[str, object],
) -> None:
    """A search move off the four-stage ``epi_tma`` default repairs to a
    config strict normalization accepts: one that leaves the
    epilogue-issued ring drops its four stages with it, and the ring's AB
    stages are judged on the tile the projections leave (a C-input warp's
    neighbour widened the pair to a tile where no ``epi_tma`` stage fits)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        spec = _three_residual_mm._bind_isolated(
            (
                torch.empty((16384, 256), dtype=torch.bfloat16),
                torch.empty((256, 256), dtype=torch.bfloat16),
                *(torch.empty((16384, 256), dtype=torch.bfloat16) for _ in range(3)),
            )
        ).config_spec
        default = spec.default_config()
        assert default.config["tcgen05_aux_load_mode"] == "epi_tma"
        assert default.config["tcgen05_aux_stages"] == 4
        repaired = helion.Config(**{**default.config, **move})
        spec.normalize(repaired, _fix_invalid=True)
        spec.normalize(helion.Config(**repaired.config))
        # The repair is a fixed point: a second pass (a flat round trip's
        # decode) leaves the benchmarked kernel unchanged.
        again = helion.Config(**repaired.config)
        spec.normalize(again, _fix_invalid=True)
    assert again.config == repaired.config
    if repaired.config["tcgen05_aux_load_mode"] != "epi_tma":
        assert repaired.config.get("tcgen05_aux_stages", 2) != 4


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _six_residual_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    r4: torch.Tensor,
    r5: torch.Tensor,
) -> torch.Tensor:
    """Six exact-shape operands added to the accumulator."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty_like(r0)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        acc = acc + r0[rows, cols].to(torch.float32)
        acc = acc + r1[rows, cols].to(torch.float32)
        acc = acc + r2[rows, cols].to(torch.float32)
        acc = acc + r3[rows, cols].to(torch.float32)
        acc = acc + r4[rows, cols].to(torch.float32)
        out[rows, cols] = (acc + r5[rows, cols].to(torch.float32)).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _three_residual_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
) -> torch.Tensor:
    """Three exact-shape operands added to the accumulator."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty_like(r0)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        acc = acc + r0[rows, cols].to(torch.float32)
        acc = acc + r1[rows, cols].to(torch.float32)
        acc = acc + r2[rows, cols].to(torch.float32)
        out[rows, cols] = acc.to(out.dtype)
    return out


def test_whole_tile_residual_pair_default_stages_through_the_ring() -> None:
    """A whole-tile residual pair's default loads its exact-shape operands
    through the epilogue-issued TMA ring, on the widest N tile whose AB ring
    keeps four stages beside it, the ring deepened where the AB ring still
    holds a whole tile's K (B200, of the ATen GEMM: 8192x1024x1024 +
    residual 0.83 on SIMT loads -> 1.05; 32768x256x256 + three operands
    0.50 -> 0.97, where 256x256 keeps three AB stages and 256x128 runs, its
    four k-blocks beside a four-stage ring)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        one = rank_two_aux._bind_isolated(
            (
                torch.empty((8192, 1024), dtype=torch.bfloat16),
                torch.empty((1024, 1024), dtype=torch.bfloat16),
                torch.empty((8192, 1024), dtype=torch.bfloat16),
            )
        ).config_spec.default_config()
        # 16384 rows: two waves of the narrower tile's pairs, one of 256x256
        m, n, k = 16384, 256, 256
        three_spec = _three_residual_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                *(torch.empty((m, n), dtype=torch.bfloat16) for _ in range(3)),
            )
        ).config_spec
        three = three_spec.default_config()
        # the flat search space encodes the promoted ring depth
        generation = ConfigGeneration(three_spec)
        round_trip = generation.unflatten(generation.flatten(three))
    assert one.block_sizes == [256, 256, 64]
    assert one.config["tcgen05_aux_load_mode"] == "epi_tma"
    # sixteen k-blocks: the ring keeps its two stages beside five AB stages
    assert one.config.get("tcgen05_aux_stages") is None
    assert one.config["tcgen05_ab_stages"] == 5
    assert three.block_sizes == [256, 128, 64]
    assert three.config["tcgen05_aux_load_mode"] == "epi_tma"
    # four k-blocks: the AB ring holds them beside a four-stage ring
    assert three.config["tcgen05_aux_stages"] == 4
    assert three.config["tcgen05_ab_stages"] == 4
    assert round_trip.config["tcgen05_aux_stages"] == 4
    # the narrower tile's two waves of pairs walk the grouped raster
    assert three.config["l2_groupings"] == [TCGEN05_TWO_CTA_SEED_L2_GROUPING]


def test_staged_residual_long_sweep_groups_eight_m_tiles() -> None:
    """A long N sweep of a pair staging its exact-shape operands groups 8 M
    tiles, not the plain sweep's 32: the staged operands stream through L2
    beside the group's A panels (eleven residuals, 4096x16384x4096: 0.80 of
    the ATen GEMM grouped by 32, 0.83 by 8)."""
    m, n, k = 4096, 16384, 4096
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        config = _three_residual_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                *(torch.empty((m, n), dtype=torch.bfloat16) for _ in range(3)),
            )
        ).config_spec.default_config()
    assert config.config["tcgen05_aux_load_mode"] == "epi_tma"
    assert config.config["l2_groupings"] == [8]


@pytest.mark.parametrize(
    ("m", "n", "k", "residual_dtype", "block_k", "ab_stages"),
    (
        # a wide N: the shallowest k-blocks, ten 32-deep stages
        (1000, 4096, 4096, torch.bfloat16, 32, 10),
        # 512 32-deep K tiles would pass the pair's cap: 64-deep ones
        (1000, 4096, 16384, torch.bfloat16, 64, 5),
        # one 256-wide N tile, an FP32 residual
        (19000, 256, 256, torch.float32, 32, 10),
    ),
)
def test_m_edge_residual_default_takes_the_producer_pair(
    m: int,
    n: int,
    k: int,
    residual_dtype: torch.dtype,
    block_k: int,
    ab_stages: int,
) -> None:
    """Rows that one-CTA 128-row tiles do not cover, under a wide N or one
    256-wide N tile, take the persistent 256x256 pair with the residual's TMA
    producer warp, at the shallowest k-blocks within the pair's K-tile cap
    (B200, of the ATen GEMM: 1000x4096x4096 + residual 0.09 on the flat base
    default -> 0.93; FP32 residual: 30000x256x4096 616 -> 91 us, 19000x256x256
    47 -> 21 us)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = rank_two_aux._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=residual_dtype),
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [256, 256, block_k]
    assert config.config["tcgen05_ab_stages"] == ab_stages
    assert config.config["tcgen05_cluster_m"] == 2
    assert config.config["tcgen05_aux_load_mode"] == "tma"
    assert config.config["tcgen05_strategy"] == "role_local_with_scheduler"


@pytest.mark.parametrize(
    ("residuals", "block_sizes", "cluster_m", "ab_stages", "keeps_producer"),
    (
        # beside one residual's ring: the bytes of two 128-deep stages of the
        # BF16 pair, four 64-deep and eight 32-deep ones
        (1, [256, 256, 64], 2, 4, True),
        (1, [256, 256, 32], 2, 8, True),
        # three 128-deep stages overshoot the cap
        (1, [256, 256, 128], 2, 3, False),
        # one CTA holds both operands: four 64-deep 128x256 stages are 192 KB
        (1, [128, 256, 64], 1, 4, False),
        # three residuals' rings leave room for six 32-deep stages, not eight
        (3, [256, 256, 32], 2, 8, False),
        (3, [256, 256, 32], 2, 6, True),
    ),
)
def test_aux_producer_warp_admits_the_ring_the_arena_leaves(
    residuals: int,
    block_sizes: list[int],
    cluster_m: int,
    ab_stages: int,
    keeps_producer: bool,
) -> None:
    """The residuals' producer warp keeps an AB ring the arena model fits
    beside its rings (one per residual) and the C ring; a deeper one drops
    the warp in the search and is refused by codegen."""
    m, n, k = 4096, 4096, 4096
    lhs = torch.empty((m, k), dtype=torch.bfloat16)
    rhs = torch.empty((k, n), dtype=torch.bfloat16)
    res = [torch.empty((m, n), dtype=torch.bfloat16) for _ in range(residuals)]
    kernel = rank_two_aux if residuals == 1 else _three_residual_mm
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = kernel._bind_isolated((lhs, rhs, *res))
        config = {
            "block_sizes": block_sizes,
            "tcgen05_cluster_m": cluster_m,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_c_stages": 2,
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_warp_spec_c_input_warps": 1,
        }
        repair._fix_with_scheduler_search_config(bound.config_spec.tcgen05, config)
        assert (config["tcgen05_warp_spec_c_input_warps"] == 1) is keeps_producer
        producer = helion.Config(
            block_sizes=block_sizes,
            l2_groupings=[1],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=cluster_m,
            tcgen05_ab_stages=ab_stages,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_warp_spec_c_input_warps=1,
            tcgen05_aux_load_mode="tma",
        )
        if not keeps_producer:
            # The strict normalizer refuses the overflowing rings ahead of
            # codegen (``_normalize_aux_producer_ab_stages``).
            with pytest.raises(exc.InvalidConfig, match="incompatible"):
                bound.to_code(producer)
            return
        code = bound.to_code(producer)
    assert "tcgen05_aux_pipeline.producer_acquire(" in code
    assert f"make_tma_umma_pipeline(num_stages={ab_stages}" in code


@pytest.mark.parametrize(
    ("kernel", "residuals", "c_stages"),
    ((two_residual_aux, 2, 4), (_three_residual_mm, 3, 2)),
)
def test_direct_store_producer_prices_no_c_ring(
    kernel: Kernel, residuals: int, c_stages: int
) -> None:
    """A direct store allocates no C ring and is priced without one: two
    128-deep pair stages fit beside two residuals' rings where the TMA
    store's four C stages overflow (262 KB of 232; NVVM refuses it), and
    beside three residuals' rings where even two C stages overflow."""
    m, n, k = 4096, 4096, 4096
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        *(torch.empty((m, n), dtype=torch.bfloat16) for _ in range(residuals)),
    )
    values = {
        "block_sizes": [256, 256, 128],
        "l2_groupings": [1],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": c_stages,
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
        "tcgen05_warp_spec_c_input_warps": 1,
        "tcgen05_aux_load_mode": "tma",
    }
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = kernel._bind_isolated(args)
        with pytest.raises(exc.InvalidConfig, match="incompatible"):
            bound.to_code(helion.Config(**values))
        code = bound.to_code(helion.Config(**values, tcgen05_c_store_mode="direct"))
    assert "tcgen05_aux_pipeline.producer_acquire(" in code


def test_m_edge_producer_pair_default_prices_every_residual_ring() -> None:
    """Over an M edge with two residuals, the producer pair's default takes
    the deepest AB ring the arena model fits beside both rings: eight 32-deep
    stages at two C stages (beside four C stages only six fit; eight are 262
    KB of 232, which NVVM refused). B200, bf16 1000x4096x4096 + two
    residuals: 32.2 us at six stages and four C stages, 31.3 at eight and
    two."""
    m, n, k = 1000, 4096, 1088
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = two_residual_aux._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes == [256, 256, 32]
    assert config.config["tcgen05_aux_load_mode"] == "tma"
    assert config.config["tcgen05_ab_stages"] == 8
    assert config.config["tcgen05_c_stages"] == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_power_mm(
    a: torch.Tensor, b: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """``acc + r ** 2**24`` as 24 nested squares: every level reads the
    previous one twice, so the chain is a DAG of 24 shared subtrees with no
    transcendental, which a classifier must walk to its leaves."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty_like(r)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        t = r[rows, cols].to(torch.float32)
        for _ in hl.static_range(24):
            t = t * t
        out[rows, cols] = (acc + t).to(out.dtype)
    return out


def test_transcendental_fact_walks_shared_subtrees_once() -> None:
    """The transcendental fact walks the whole chain once per node, by
    identity: a walk without the identity set visits 2**24 nodes here, and
    hashing the value-compared nodes rehashes each shared subtree. The bind
    expands each node's children a few dozen times in all (once per walk),
    not millions."""
    m, n, k = 8192, 1024, 1024
    calls = 0

    def counted(expr: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return children(expr)

    children = cute_epilogue._tensor_expr_children
    with (
        patch(
            "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
            return_value=148,
        ),
        patch.object(cute_epilogue, "_tensor_expr_children", counted),
    ):
        bound = _residual_power_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        facts = bound.config_spec._cute_tcgen05_config.pipeline_smem_facts
    assert facts is not None
    assert not facts.epilogue_transcendental
    assert 0 < calls < 10_000


def test_narrow_n_m_edge_default_takes_the_256xn_pair() -> None:
    """A GEMM one 128-wide N tile wide over an M edge defaults to the
    persistent CtaGroup.TWO 256x128 pair: the proved-schedule seeds all ask
    for whole tiles, and the 256x256 projection would run half of each MMA
    past N (30000x128x4096 + residual on B200: the flat 128x16 base default
    0.19 of the ATen GEMM, the 256x128 pair 0.86)."""
    m, n, k = 19000, 128, 256
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = rank_two_aux._bind_isolated(
            (
                torch.empty((k, m), dtype=torch.bfloat16).T,
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.float32),
            )
        )
        config = bound.config_spec.default_config()
        assert config.block_sizes == [256, 128, 64]
        assert config.config["tcgen05_cluster_m"] == 2
        assert config.config["pid_type"] == "persistent_interleaved"
        # Every two-CTA sample runs the 256 x 128 pair too, whatever its
        # sampled M and N tiles.
        for block_sizes in ([128, 128, 64], [128, 64, 64], [512, 128, 64]):
            sampled = {
                **config.config,
                "block_sizes": block_sizes,
                "tcgen05_ab_stages": 4,
            }
            bound.config_spec.normalize(sampled, _fix_invalid=True)
            assert sampled["block_sizes"] == [256, 128, 64]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_atomic_mm(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor, counter: torch.Tensor
) -> torch.Tensor:
    """``acc + residual`` beside an atomic counter."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty_like(residual)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = (acc + residual[rows, cols].to(torch.float32)).to(out.dtype)
        hl.atomic_add(counter, [0], 1)
    return out


def test_m_edge_pair_default_keeps_64_deep_blocks_without_a_producer_ring() -> None:
    """Six residuals' rings leave no room for two AB stages beside them even
    at 32-deep k-blocks (one ring each: 32 + 32 + 6 x 32 KB of 232), so the
    projection keeps the pair's register loads, and with them the M-edge
    default's 64-deep k-blocks (the shallow ones only serve the producer's
    ring)."""
    m, n, k = 1000, 4096, 1024
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        *(torch.empty((m, n), dtype=torch.bfloat16) for _ in range(6)),
    )
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _six_residual_mm._bind_isolated(args)
        config = bound.config_spec.default_config()
        bound.to_code(config)
    assert config.block_sizes == [256, 256, 64]
    assert config.config.get("tcgen05_aux_load_mode") != "tma"


def test_m_edge_pair_default_renders_beside_masks_atomics_and_offsets() -> None:
    """Over an M edge the pair's producer warp takes only an operand its ring
    carries and a TensorMap proves, and no default runs an atomic beside the
    warp roles: a boolean mask alone keeps the pair's register loads, an
    atomic the flat base default, and a residual off its 16-byte alignment
    the register loads at 64-deep k-blocks. Each default renders its source
    (CPU codegen; ``test_cute_tcgen05_flat_edge_gpu`` runs the residual
    defaults)."""
    m, n, k = 1000, 4096, 1024
    a = torch.empty((m, k), dtype=torch.bfloat16)
    b = torch.empty((k, n), dtype=torch.bfloat16)
    shifted = torch.empty((m * n + 1,), dtype=torch.bfloat16)[1:].view(m, n)
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        configs = []
        for bound in (
            dropout_mm_into._bind_isolated(
                (
                    a,
                    b,
                    torch.empty((m, n), dtype=torch.bool),
                    torch.empty((m, n), dtype=torch.bfloat16),
                )
            ),
            _residual_atomic_mm._bind_isolated(
                (
                    a,
                    b,
                    torch.empty((m, n), dtype=torch.bfloat16),
                    torch.zeros((1,), dtype=torch.int32),
                )
            ),
            rank_two_aux._bind_isolated((a, b, shifted)),
        ):
            config = bound.config_spec.default_config()
            bound.to_code(config)
            configs.append(config.config)
    mask, atomic, offset = configs
    assert mask["tcgen05_cluster_m"] == 2
    assert mask.get("tcgen05_aux_load_mode", "simt") == "simt"
    assert atomic.get("tcgen05_cluster_m", 1) == 1
    assert atomic.get("pid_type", "flat") == "flat"
    assert offset.get("tcgen05_aux_load_mode", "simt") == "simt"
    assert offset["block_sizes"][-1] == 64


def test_column_major_output_keeps_pre_acc_wait_off_the_128_row_pair() -> None:
    """Beside a column-major output the 128-row CTA pair loads its rows after
    the accumulator wait (the whole-row load before it fails in CuTe):
    normalization repairs or refuses ``pre_acc_wait`` there and the seeds
    offer only ``post_acc_wait``, while the 256-row pairs keep both."""
    m, n, k = 512, 2048, 512
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty((n,), dtype=torch.bfloat16),
        torch.empty((n, m), dtype=torch.bfloat16).T,
    )
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        spec = bias_mm_into._bind_isolated(args).config_spec
    placements = {
        (seed.block_sizes[0], seed.config.get("tcgen05_aux_load_placement"))
        for seed in spec.compiler_seed_configs
        if seed.config.get("tcgen05_cluster_m") == 2
    }
    assert (128, "pre_acc_wait") not in placements
    assert {(128, "post_acc_wait"), (256, "pre_acc_wait")} <= placements
    pair = next(
        seed
        for seed in spec.compiler_seed_configs
        if seed.block_sizes[0] == 128 and seed.config.get("tcgen05_cluster_m") == 2
    )
    sampled = helion.Config(
        **{**pair.config, "tcgen05_aux_load_placement": "pre_acc_wait"}
    )
    with pytest.raises(helion.exc.InvalidConfig, match="128-row CTA pair"):
        spec.normalize(helion.Config(**sampled.config))
    spec.normalize(sampled, _fix_invalid=True)
    assert sampled.config["tcgen05_aux_load_placement"] == "post_acc_wait"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``0.125 * bmm(a, b)``."""
    batch, m, k = a.shape
    n = b.shape[2]
    out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    for tile_b, tile_m, tile_n in hl.tile((batch, m, n)):
        acc = hl.zeros((tile_b, tile_m, tile_n), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.baddbmm(acc, a[tile_b, tile_m, inner], b[tile_b, inner, tile_n])
        out[tile_b, tile_m, tile_n] = (acc * 0.125).to(out.dtype)
    return out


@pytest.mark.parametrize(
    ("shape", "block_sizes", "pid_type", "ab_stages"),
    (
        # four k-blocks: flat, two AB stages
        ((1024, 256, 64, 256), [1, 128, 64, 64], "flat", 2),
        # 64 rows by the widest dividing N tile; eight k-blocks: persistent
        ((384, 64, 512, 512), [1, 64, 256, 64], "persistent_blocked", 4),
    ),
)
def test_narrow_batched_default_takes_the_one_cta_tile(
    shape: tuple[int, int, int, int],
    block_sizes: list[int],
    pid_type: str,
    ab_stages: int,
) -> None:
    """Per-batch matrices narrower than 128 rows or columns default to the
    one-CTA tile of their extents (B200, of the ATen GEMM: 1024 x
    256x64x256 + scale 1.09, 384 x 64x512x512 1.03), not the 64x16 base."""
    batch, m, n, k = shape
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _scaled_bmm._bind_isolated(
            (
                torch.empty((batch, m, k), dtype=torch.bfloat16),
                torch.empty((batch, k, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        bound.to_code(config)
    assert config.block_sizes == block_sizes
    assert config.config["pid_type"] == pid_type
    assert config.config["tcgen05_ab_stages"] == ab_stages
    assert config.config["tcgen05_cluster_m"] == 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _five_row_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    r4: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``acc + r0[cols] + ... + r4[cols]``: five unpromoted bf16 rows."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = (
            acc + r0[cols] + r1[cols] + r2[cols] + r3[cols] + r4[cols]
        ).to(out.dtype)
    return out


def test_many_tile_bias_defaults_pick_the_row_placement() -> None:
    """Many tiles per pair: static full 256x128 tiles stage a promoted bias
    row before the accumulator wait (two K tiles: the epilogue bounds the
    pair), and five unpromoted rows the partial-output store cannot hold
    leave ``pre_acc_wait`` for the full tiles' TMA store (B200: promoted
    bias 65536x1152x128 44.3 -> 42.0 us, five rows 16384x30528x768 3284 ->
    2539 us)."""
    bf16 = torch.bfloat16
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bias = promoted_bias_mm_into._bind_isolated(
            (
                torch.empty((65536, 128), dtype=bf16),
                torch.empty((128, 1152), dtype=bf16),
                torch.empty((1152,), dtype=bf16),
                torch.empty((65536, 1152), dtype=bf16),
            )
        ).config_spec.default_config()
        rows_spec = _five_row_mm._bind_isolated(
            (
                torch.empty((16384, 768), dtype=bf16),
                torch.empty((768, 30528), dtype=bf16),
                *(torch.empty((30528,), dtype=bf16) for _ in range(5)),
                torch.empty((16384, 30528), dtype=bf16),
            )
        ).config_spec
        rows = rows_spec.default_config()
    assert bias.block_sizes == [256, 128, 64]
    assert bias.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
    assert bias.config["tcgen05_ab_stages"] == 7
    # the partial-output store does not hold the five rows: the full tiles'
    # TMA store, the edge tiles' SIMT store
    assert rows.config["tcgen05_aux_load_placement"] == "post_acc_wait"
    assert not smem.simt_aux_partial_output_fits(
        rows_spec._cute_tcgen05_config,
        dict(rows.config),
        bm=256,
        bn=256,
        bk=64,
        cluster_m=2,
        ab_stages=rows.config["tcgen05_ab_stages"],
        c_stages=rows.config["tcgen05_c_stages"],
    )


def test_one_cta_n_edge_admission_needs_known_store_facts() -> None:
    """Two column-major outputs of a biased accumulator leave the search no
    store facts: their persistent one-CTA N-edge sample goes flat instead of
    running persistent on the SIMT store codegen gives a column-major
    auxiliary epilogue."""
    m, n, k = 8192, 576, 192
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _two_output_bias_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((n, m), dtype=torch.bfloat16).T,
                torch.empty((n, m), dtype=torch.bfloat16).T,
            )
        )
        spec = bound.config_spec
        config = spec.default_config()
        config.config.update(
            block_sizes=[128, 256, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_ab_stages=3,
            tcgen05_aux_load_placement="post_acc_wait",
        )
        spec.normalize(config, _fix_invalid=True)
    assert spec._cute_tcgen05_config.pipeline_smem_facts is None
    assert config.config["pid_type"] == "flat"


@pytest.mark.parametrize(
    ("m", "n", "block_sizes", "ab_stages"),
    (
        # one row of M tiles: 394 128x128 tiles (2.66 waves of 148) over 197
        # 128x256 ones (1.33 waves)
        (128, 50400, [128, 128, 64], 6),
        # two rows: the 256x256 pair keeps the MMA width
        (256, 50400, [256, 256, 64], 6),
        # five rows of N-edge tiles keep the 128x256 tile (the 128-wide one
        # is a single-row candidate only)
        (640, 9000, [128, 256, 64], 3),
    ),
)
def test_decode_bias_default_tile(
    m: int, n: int, block_sizes: list[int], ab_stages: int
) -> None:
    """A decode GEMM (one row of M tiles) streams B: the MMA width buys
    nothing and the N tiles' wave quantization decides, so 128x50400x4096 +
    bias defaults to the persistent one-CTA 128x128 tile at the deepest ring
    (0.99 of the ATen GEMM, against 0.83 on 128x256), storing every tile
    through TMA; more rows of M tiles keep the wider tiles."""
    k = 4096
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = bias_mm_into._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
    assert config.block_sizes == block_sizes
    assert config.config["pid_type"] == "persistent_interleaved"
    assert config.config["tcgen05_ab_stages"] == ab_stages
    assert "tcgen05_tma_store_atom" in code
    assert "cute.nvgpu.CopyR2GOp()" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bf16_residual_f32_mm(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor, out: torch.Tensor
) -> None:
    """An FP32 output plus a bf16 exact-shape residual (no dtype match)."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = acc + residual[rows, cols].to(torch.float32)


def test_one_cta_n_edge_residual_default_needs_k_tiles() -> None:
    """An exact-shape residual bounds the 256-wide tile's epilogue on few K
    tiles: the one-CTA 128x256 N-edge default takes it from 8 K tiles
    (25216x576x1024 + residual: 100 us against the flat default's 260), and
    25216x576x192 keeps the flat default (78 us against 88 at K=256)."""
    defaults = {}
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        for k in (192, 1024):
            bound = rank_two_aux._bind_isolated(
                (
                    torch.empty((25216, k), dtype=torch.bfloat16),
                    torch.empty((k, 576), dtype=torch.bfloat16),
                    torch.empty((25216, 576), dtype=torch.bfloat16),
                )
            )
            config = bound.config_spec.default_config().config
            defaults[k] = (config["block_sizes"], config["pid_type"])
    assert defaults[192][1] == "flat"
    assert defaults[1024] == ([128, 256, 64], "persistent_interleaved")


def test_one_cta_n_edge_residual_gate_ignores_the_residual_dtype() -> None:
    """A bf16 residual of an FP32 output bounds the 256-wide tile's epilogue
    as much as one of the output's dtype: 25216x576x192 keeps the flat
    default too."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _bf16_residual_f32_mm._bind_isolated(
            (
                torch.empty((25216, 192), dtype=torch.bfloat16),
                torch.empty((192, 576), dtype=torch.bfloat16),
                torch.empty((25216, 576), dtype=torch.bfloat16),
                torch.empty((25216, 576), dtype=torch.float32),
            )
        )
        state = bound.config_spec._cute_tcgen05_config
        config = bound.config_spec.default_config().config
    assert state.exact_shape_aux_kernel_detected
    assert config["pid_type"] == "flat"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bmm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """A batched GEMM: partial tiles only on the flat one-CTA pipeline."""
    bsz, m, k = a.shape
    n = b.shape[2]
    for bi, rows, cols in hl.tile((bsz, m, n), block_size=[1, None, None]):
        acc = hl.zeros((bi, rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.baddbmm(acc, a[bi, rows, inner], b[bi, inner, cols])
        out[bi, rows, cols] = acc.to(out.dtype)


def test_one_cta_n_edge_default_survives_the_search_fixups() -> None:
    """The one-CTA 128x256 N-edge default is the deepest ring the search
    fixups keep persistent (an auxiliary edge epilogue's C ring goes to 4
    stages: K=768 takes ab=3), a sampled deeper AB ring shortens to it
    instead of going flat, and batched GEMMs keep their flat default (a
    persistent batched partial tile falls back to scalar code)."""
    bf16 = torch.bfloat16
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bias = bias_mm_into._bind_isolated(
            (
                torch.empty((25216, 768), dtype=bf16),
                torch.empty((768, 576), dtype=bf16),
                torch.empty((576,), dtype=bf16),
                torch.empty((25216, 576), dtype=bf16),
            )
        )
        bias_config = bias.config_spec.default_config()
        bias_code = bias.to_code(bias_config)
        sampled = helion.Config(**{**bias_config.config, "tcgen05_ab_stages": 6})
        bias.config_spec.normalize(sampled, _fix_invalid=True)
        batched = _bmm_into._bind_isolated(
            (
                torch.empty((4, 1024, 192), dtype=bf16),
                torch.empty((4, 192, 576), dtype=bf16),
                torch.empty((4, 1024, 576), dtype=bf16),
            )
        )
        batched_config = batched.config_spec.default_config()
    assert bias_config.config["pid_type"] == "persistent_interleaved"
    assert bias_config.config["tcgen05_ab_stages"] == 3
    assert sampled.config["pid_type"] == "persistent_interleaved"
    assert sampled.config["tcgen05_ab_stages"] == 3
    assert "tcgen05_tma_store_atom" in bias_code
    assert "cute.nvgpu.CopyR2GOp()" not in bias_code
    assert batched_config.config["pid_type"] == "flat"


def test_one_k_tile_n_edge_default_runs_one_ab_stage() -> None:
    """With one K tile the one-CTA N-edge candidate runs one AB stage, and
    the ring fixup keeps it persistent there (bf16 25216x576x64: 8.2 us
    persistent at ab=1, 26.9 on the flat 128x16 default)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm_into._bind_isolated(
            (
                torch.empty((25216, 64), dtype=torch.bfloat16),
                torch.empty((64, 576), dtype=torch.bfloat16),
                torch.empty((25216, 576), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config().config
    assert config["block_sizes"] == [128, 256, 64]
    assert config["pid_type"] == "persistent_interleaved"
    assert config["tcgen05_ab_stages"] == 1


def test_one_cta_n_edge_rings_are_judged_after_the_ring_fixups() -> None:
    """The persistence of a one-CTA N-edge sample follows its final rings:
    samples the ring fixups leave identical (a sampled C ring of 4 goes to
    2 on this plain one-CTA tile) project to the same persistent kernel on
    the TMA store."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm_into._bind_isolated(
            (
                torch.empty((25216, 768), dtype=torch.bfloat16),
                torch.empty((768, 576), dtype=torch.bfloat16),
                torch.empty((25216, 576), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        projected = []
        for c_stages in (4, 2):
            config = helion.Config(
                **{
                    **spec.default_config().config,
                    "block_sizes": [128, 256, 64],
                    "pid_type": "persistent_interleaved",
                    "tcgen05_cluster_m": 1,
                    "tcgen05_ab_stages": 4,
                    "tcgen05_c_stages": c_stages,
                }
            )
            spec.normalize(config, _fix_invalid=True)
            projected.append(config)
        code = bound.to_code(projected[0])
    assert projected[0].config == projected[1].config
    assert projected[0].config["pid_type"] == "persistent_interleaved"
    assert "tcgen05_tma_store_atom" in code
    assert "cute.nvgpu.CopyR2GOp()" not in code


def test_hybrid_store_edge_reads_a_staged_row_from_global_memory() -> None:
    """A ``pre_acc_wait`` row stage lives in the full-tile branch of the
    hybrid TMA/SIMT store; the SIMT edge branch reads the row from global
    memory instead of the stage it never allocated (``NameError`` in the DSL
    when the edge built its view over the stage)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = promoted_bias_mm_into._bind_isolated(
            (
                torch.empty((4096, 128), dtype=torch.bfloat16),
                torch.empty((128, 30000), dtype=torch.bfloat16),
                torch.empty((30000,), dtype=torch.bfloat16),
                torch.empty((4096, 30000), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        # ab5 leaves SMEM for the C ring and the row stage: the hybrid store
        config.config.update(
            tcgen05_aux_load_placement="pre_acc_wait", tcgen05_ab_stages=5
        )
        code = bound.to_code(config)
    full, edge = full_and_edge_store_bodies(code)
    assert "tcgen05_tma_store_atom" in full
    assert "make_tensor(tcgen05_aux_rowvec_smem_0.iterator" in full
    assert "tcgen05_aux_rowvec_smem" not in edge
    assert "make_tensor(bias.iterator" in edge


def _k_tail_bind(m: int, n: int, k: int) -> BoundKernel:
    return plain_mm_into._bind_isolated(
        (
            torch.empty((m, k), dtype=torch.bfloat16),
            torch.empty((n, k), dtype=torch.bfloat16).T,
            torch.empty((m, n), dtype=torch.float32),
        )
    )


@pytest.mark.parametrize(
    ("shape", "family"),
    (
        # whole M and N, a K no 64-deep tile divides (Blenderbot's 8008)
        ((2048, 2560, 8008), True),
        # ragged in both M and N with a K tail: the family's own shapes
        ((3000, 3000, 3000), True),
        # a single ragged edge with a K tail: outside the validated runtime
        ((1000, 2560, 8008), False),
        # whole K: the dense families
        ((2048, 2560, 8000), False),
    ),
)
def test_k_tail_shapes_take_the_edge_k_tail_family(
    shape: tuple[int, int, int], family: bool
) -> None:
    """A K tail on whole M and N tiles runs the edge + K-tail CtaGroup.TWO
    family (bf16 2048x2560x8008: 0.92 of cuBLAS on its 256x256x64 seed, 0.75
    on the flat winner), and the autotune-off default promotes that seed
    instead of the flat 128x16 kernel (0.08)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _k_tail_bind(*shape)
        state = bound.config_spec._cute_tcgen05_config
        constraints = state.cluster_m2_search_constraints
        config = bound.config_spec.default_config()
    assert (constraints is not None and constraints.allow_edge_k_tail_family) == family
    if family:
        assert config.block_sizes == [256, 256, 64]
        assert config.config["tcgen05_persistence_model"] == "clc_persistent"
        # the family's deep bk=64 ring, as normalization keeps it
        assert config.config["tcgen05_ab_stages"] == 5


def test_edge_k_tail_default_needs_the_dense_gemm_body() -> None:
    """The family's default promotes only a plain dense GEMM body: an atomic
    beside the warp roles keeps the flat default."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = atomic_output_mm._bind_isolated(
            (
                torch.empty((3000, 3000), dtype=torch.bfloat16),
                torch.empty((3000, 3000), dtype=torch.bfloat16),
                torch.zeros((1,), dtype=torch.int32),
            )
        )
        config = bound.config_spec.default_config()
    assert config.config.get("tcgen05_persistence_model") != "clc_persistent"


@pytest.mark.parametrize(
    ("m", "n", "k"),
    (
        # the small-grid pair, an N-edge 256x256 pair, a persistent one-CTA
        # 128x256 tile on N-edge tiles, the FFI direct-entry pair of whole
        # 256x256 tiles
        (128, 4096, 9216),
        (4096, 2000, 768),
        (25216, 576, 192),
        (4096, 4096, 4096),
    ),
)
def test_default_keeps_atomics_on_the_flat_base_default(m: int, n: int, k: int) -> None:
    """An atomic beside the warp roles keeps every grid on the flat base
    default: codegen refuses atomics on two-CTA pairs and on persistent CTAs
    that own several tiles (``reject_atomics_beside_warp_roles``), the FFI
    direct-entry base default of whole 256x256 tiles included."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = atomic_output_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.zeros((1,), dtype=torch.int32),
            )
        )
        config = bound.config_spec.default_config()
        bound.to_code(config)
    assert config.config.get("tcgen05_cluster_m", 1) == 1
    assert config.config.get("pid_type", "flat") == "flat"


def test_small_grid_default_keeps_the_search_bounds() -> None:
    """Two 128x128 output tiles on 148 SMs: the search admits the 64-row
    one-CTA tile for grids smaller than the SM count (the 256-row operands
    alone would pin the floor at 128 rows), and the register-MMA family
    admitted on this latency-bound problem widens the M floor to its 16-row
    atom; K tiles reach min(128, K). The small-N problem cannot form the
    256x256 CtaGroup.TWO tile, so the cluster search stays at one CTA, and
    the default is the paired-pipeline 128-row pair seed, which the
    allocation proof validates outside that search."""
    args = (
        torch.empty([256, 64], dtype=torch.float16),
        torch.empty([64, 128], dtype=torch.float16),
    )
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        spec = _addmm_mm._bind_isolated(args).config_spec
        default = spec.default_config().config
    assert [x.min_size for x in spec.block_sizes] == [16, 8, 16]
    assert [x.max_size for x in spec.block_sizes] == [256, 128, 64]
    assert default["l2_groupings"] == [1]
    assert spec.tcgen05.cluster_m_search_choices == (1,)
    assert default["block_sizes"] == [128, 64, 64]
    assert default["tcgen05_cluster_m"] == 2
    assert "persistent_blocked" in spec.allowed_pid_types
    assert "persistent_interleaved" in spec.allowed_pid_types


def test_small_grid_relu_default_is_the_pair_not_a_thread_local_tile() -> None:
    """A regular (ReLU) epilogue does not take the thread-local epilogue
    heuristic; the paired-pipeline seeds' 128-row pair is the default."""
    args = (
        torch.empty([128, 128], dtype=torch.bfloat16),
        torch.empty([128, 128], dtype=torch.bfloat16),
    )
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        spec = _relu_mm._bind_isolated(args).config_spec
        default = spec.compiler_default_config
    assert CuteTcgen05ThreadLocalEpilogueHeuristic.name not in spec.autotuner_heuristics
    assert default is not None
    assert default.config["tcgen05_cluster_m"] == 2
    assert default.config["block_sizes"][0] == 128


@pytest.mark.parametrize(("m", "n", "k"), ((128, 9600, 4096), (1152, 1152, 1024)))
def test_small_grid_default_takes_the_fewest_waves(m: int, n: int, k: int) -> None:
    """Where no 128-row pair fits one wave the small-grid default takes the
    one with the fewest (the 128x128 pair's two), not the flat 64x16 base
    tile (B200: 128x9600x4096 137 -> 29 us, 1152x1152x1024 37 -> 12 us; ATen
    27 and 10.8 us)."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        bound.to_code(config)
    assert config.block_sizes[:2] == [128, 128]
    assert config.config["tcgen05_cluster_m"] == 2


# Small grids no 128-row pair serves: a 64-row M (64x4096x4096), K tails
# (K=1000, 16 k-blocks: unsplit, at three per-SM CTA counts), a K loop
# past the pairs' 256 k-blocks, the 32 k-block minimum's edge (31 unsplit,
# 32 in four slices) and three CTAs per SM capping 216 tiles at two slices.
@pytest.mark.parametrize(
    ("m", "n", "k", "split"),
    (
        (64, 4096, 4096, 4),
        (32, 1280, 1000, 1),
        (128, 4096, 1000, 1),
        (320, 2048, 1000, 1),
        (256, 512, 32768, 8),
        (64, 4096, 1984, 1),
        (64, 4096, 2048, 4),
        (64, 13824, 2048, 2),
    ),
)
def test_small_grid_without_a_pair_splits_k(m: int, n: int, k: int, split: int) -> None:
    """The flat one-CTA 64x64x64 tile, not the flat 64x16 base tile, over a
    K loop of 32 k-blocks or more splitting K into the most slices of at
    least four k-blocks whose CTAs three per SM hold (64 tiles of
    64x4096x4096: four slices, 256 CTAs); unsplit over 16 k-blocks."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = plain_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
    assert config.block_sizes == [64, 64, 64]
    assert config.config["pid_type"] == "flat"
    assert config.config["tcgen05_split_k"] == split
    assert config.config["tcgen05_ab_stages"] == 4
    assert "cute.gemm(" in source


# Short K loops keep the 64x64 tile unsplit (under the 32 k-block minimum:
# five k-blocks, three at three AB stages, and a 16 k-block K tail);
# without the flat-edge TMA path a K tail stays unsplit too, and a sampled
# split is dropped there.
@pytest.mark.parametrize(
    ("m", "n", "k", "ab_stages", "flat_edge_tma"),
    (
        (64, 4096, 320, 4, True),
        (64, 4096, 192, 3, True),
        (1280, 1600, 1000, 4, True),
        (64, 4096, 4000, 4, False),
    ),
)
def test_small_grid_split_k_default_edges(
    m: int, n: int, k: int, ab_stages: int, flat_edge_tma: bool
) -> None:
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = helion.kernel(
            plain_mm.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            cute_flat_edge_tma=flat_edge_tma,
        )._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
        assert "cute.gemm(" in bound.to_code(config)
        # a sampled split stays off a ragged grid without the flat-edge path
        sample = {**config.config, "tcgen05_split_k": 4}
        bound.config_spec.normalize(sample, _fix_invalid=True)
        if not flat_edge_tma:
            # and an explicit one is refused before codegen
            with pytest.raises(exc.InvalidConfig, match="splits whole tiles only"):
                bound.config_spec.normalize(
                    helion.Config(**{**config.config, "tcgen05_split_k": 4})
                )
    assert config.block_sizes == [64, 64, 64]
    assert config.config["pid_type"] == "flat"
    assert config.config["tcgen05_split_k"] == 1
    assert config.config["tcgen05_ab_stages"] == ab_stages
    # (four slices need four k-blocks)
    assert sample["tcgen05_split_k"] == (4 if flat_edge_tma and k >= 4 * 64 else 1)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pinned_k_block_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    block_k = hl.register_block_size(128, 128)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


def test_split_k_default_counts_slices_on_the_tile_that_runs() -> None:
    """A K block pinned to 128 runs 128-deep k-blocks: 4096 deep holds 32,
    so eight slices of four (the 64-deep count would split in sixteen, two
    128-deep k-blocks each)."""
    m, n, k = 128, 512, 4096
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _pinned_k_block_mm._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    # the registered K block comes first: K, M, N
    assert config.block_sizes == [128, 64, 64]
    assert config.config["tcgen05_split_k"] == 8


# A batch of one: codegen splits no leading passthrough axis, so split-K is
# off its search surface (and its default keeps the base default's tile).
def test_batch_of_one_takes_no_split_k_search() -> None:
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _scaled_bmm._bind_isolated(
            (
                torch.empty((1, 64, 1000), dtype=torch.bfloat16),
                torch.empty((1, 1000, 4096), dtype=torch.bfloat16),
            )
        )
        state = bound.config_spec._cute_tcgen05_config
        config = bound.config_spec.default_config()
        # an explicit split is refused by the strict pass, not by codegen
        explicit = helion.Config(
            block_sizes=[1, 64, 64, 64], pid_type="flat", tcgen05_split_k=4
        )
        with pytest.raises(helion.exc.InvalidConfig, match="tcgen05_split_k=4"):
            bound.config_spec.normalize(explicit)
    assert state.matmul_has_leading_passthrough
    assert surface.split_k_search_choices(state) == ()
    assert config.config.get("tcgen05_split_k", 1) == 1


def test_small_grid_default_loads_the_row_before_the_accumulator_wait() -> None:
    """On the small-grid pair the bias row loaded before the accumulator wait
    (``pre_acc_wait``) wins the tie, and the AB ring stops at the K loop's
    one 64-deep block."""
    m, n, k = 128, 9216, 64
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = bias_mm_into._bind_isolated(
            (
                torch.empty((m, k), dtype=torch.bfloat16),
                torch.empty((k, n), dtype=torch.bfloat16),
                torch.empty((n,), dtype=torch.bfloat16),
                torch.empty((m, n), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    assert config.block_sizes[0] == 128
    assert config.config["tcgen05_cluster_m"] == 2
    assert config.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
    assert config.config["tcgen05_ab_stages"] == 2


def test_edge_k_tail_cluster_n2_seed_needs_even_n_tiles() -> None:
    """The family's 4-CTA seed pairs N tiles: 1100 columns are 5 tiles of
    256, which the cluster refuses."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _k_tail_bind(1100, 1100, 3000)
        state = bound.config_spec._cute_tcgen05_config
        with bound.env:
            seed_configs = seeds.autotune_seed_configs(state)
    assert state.cluster_m2_search_constraints is not None
    assert state.cluster_m2_search_constraints.allow_edge_k_tail_family
    assert seed_configs
    assert all(seed.config.get("tcgen05_cluster_n") != 2 for seed in seed_configs)


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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mm_n128(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_n = hl.register_block_size(16, 128)
    for rows, cols in hl.tile((m, n), block_size=[None, block_n]):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


def test_default_tile_stays_inside_registered_block_sizes() -> None:
    """An N tile registered at most 128 wide keeps the promoted default
    inside it: the one-CTA 128x256 candidate the M quantization prefers at
    25216x768x3072 would leave the declared range."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        bound = _mm_n128._bind_isolated(
            (
                torch.empty((25216, 3072), dtype=torch.bfloat16),
                torch.empty((3072, 768), dtype=torch.bfloat16),
            )
        )
        config = bound.config_spec.default_config()
    view = facts._matmul_config_view(bound.config_spec.tcgen05, dict(config.config))
    assert view is not None
    blocks, _m_index, n_index, _k_index = view
    assert blocks[n_index] <= 128


def test_direct_entry_seed_deepens_64_deep_k_tiles() -> None:
    """K=576 takes only 64-deep tiles: six stages instead of three."""
    seed = seeds.full_tile_direct_entry_seed_config(
        _plain_bind(8192, 4096, 576).config_spec.tcgen05
    )
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
    seed = seeds.full_tile_direct_entry_seed_config(bound.config_spec.tcgen05)
    assert seed is not None
    assert seed.block_sizes == [256, 256, 64]
    assert seed.config["tcgen05_ab_stages"] == 3


def test_direct_entry_seed_takes_three_64_deep_stages_past_the_128_deep_ring() -> None:
    """K=1024 with three outputs: the three 128-deep stages hold the deep
    ring's 192 KiB of AB, beside which three C rings overflow the arena as
    well, so the seed runs three 64-deep stages."""
    bound = _mm_three_outputs._bind_isolated(
        (
            torch.empty((8192, 1024), dtype=torch.bfloat16),
            torch.empty((4096, 1024), dtype=torch.bfloat16).T,
        )
    )
    state = bound.config_spec.tcgen05
    seed = seeds.full_tile_direct_entry_seed_config(state)
    assert seed is not None
    assert seed.block_sizes == [256, 256, 64]
    assert seed.config["tcgen05_ab_stages"] == 3
    assert seeds.full_tile_direct_entry_seed_eligible(state)


def test_direct_entry_seed_keeps_a_declared_k_block_size() -> None:
    """A K fragment pinned to 128 keeps the three-stage 128-deep seed."""
    bound = _mm_k128._bind_isolated(
        (
            torch.empty((4096, 1024), dtype=torch.bfloat16),
            torch.empty((4096, 1024), dtype=torch.bfloat16).T,
        )
    )
    seed = seeds.full_tile_direct_entry_seed_config(bound.config_spec.tcgen05)
    assert seed is not None
    # The registered K block size is block 0; the M/N tile follows.
    assert seed.block_sizes == [128, 256, 256]
    assert seed.config["tcgen05_ab_stages"] == 3


def test_default_tile_one_cta_tile_keeps_a_registered_k_block() -> None:
    """A K block size registered at 32 keeps the default inside it: the
    one-CTA 128x256x64 candidate is left out at 25216x768x3072, where it is
    the pick otherwise."""
    with patch(
        "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
        return_value=148,
    ):
        config = user_bound_mm._bind_isolated(
            (
                torch.empty((25216, 3072), dtype=torch.bfloat16),
                torch.empty((3072, 768), dtype=torch.bfloat16),
                32,
            )
        ).config_spec.default_config()
    # block 0 is the registered K block size
    assert config.block_sizes[0] == 32


def test_default_tile_promotion_survives_ffi_normalization() -> None:
    """A one-CTA pick over an FFI-eligible shape keeps its tile: the promoted
    values turn the FFI controls off instead of relying on the winning seed.
    On 147 SMs the 147 FFI pair tiles take three waves of 73 pairs, where the
    294 one-CTA tiles take two; on 148 the pair would stay."""
    config = _default_with_sms((5376, 1792, 4096), 147)
    assert config.block_sizes == [128, 256, 64]
    assert not config.config.get("tcgen05_tvm_ffi_launch")


def test_default_tile_gives_small_grids_the_pair_that_fills_one_wave() -> None:
    """Fewer 128x128 tiles than SMs: the default is the paired-pipeline seeds'
    128-row pair with the fewest waves, then the most CTAs (1024x1024x4096:
    64 128x128 pairs, 128 CTAs in one wave; 128x64 pairs would need two)."""
    bound = _plain_bind(1024, 1024, 4096)
    with bound.env:
        config = CuteTcgen05DefaultTileHeuristic._pick(
            bound.env, bound.host_function.device_ir
        )
    assert config is not None
    assert config.block_sizes == [128, 128, 64]
    assert config.config["tcgen05_cluster_m"] == 2


def test_plain_pair_prices_the_narrow_store_subtile() -> None:
    """A plain 16-bit 256x256 pair stores through (128, 32) C subtiles, so a
    six-deep 64-wide AB ring keeps a four-deep C ring (192 + 32 KiB).  Pricing
    the C ring at CuTe's with-source (128, 64) subtile capped the ring at five
    stages and rejected the config."""
    bound = _plain_bind(16384, 768, 768)
    values = {
        "block_sizes": [256, 256, 64],
        "pid_type": "persistent_blocked",
        "tcgen05_cta_group": "two",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 6,
        "tcgen05_c_stages": 4,
        "tcgen05_tvm_ffi_launch": False,
    }
    state = bound.config_spec._cute_tcgen05_config
    assert smem._paired_pipeline_stage_limit(state, values) == 6
    # Each epilogue warp group owns its own ring: two groups of four (128, 32)
    # stages take 64 KiB, which leaves room for five AB stages, not six.
    assert (
        smem._paired_pipeline_stage_limit(
            state, values | {"tcgen05_epi_warp_groups": 2}
        )
        == 5
    )
    # A K tail leaves the static-full-tile rule: codegen keeps the with-source
    # (128, 64) subtile, so four C stages again cap the AB ring at five.
    k_tail = _plain_bind(16384, 768, 800).config_spec._cute_tcgen05_config
    assert smem._paired_pipeline_stage_limit(k_tail, values) == 5
    config = helion.Config(**values)
    bound.config_spec.normalize(config)
    assert config["tcgen05_ab_stages"] == 6
    assert config["tcgen05_c_stages"] == 4
    source = bound.to_code(config)
    assert "PipelineTmaStore.create(num_stages=4" in source
    assert "(cute.make_layout(128), cute.make_layout(32))" in source
