"""CPU proofs for an unsplit N fringe shared by a complete 2x2 cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import blackwell_capability
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import extra_store_mm
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import vector_scaled_mm

import helion
from helion._compiler.cute.tcgen05_config import _TWO_CTA_PERSISTENT_PID_VETO
from helion._compiler.cute.tcgen05_schedule_facts import Tcgen05FullTileScheduleFacts
from helion._compiler.cute.tcgen05_schedule_facts import (
    n_edge_cluster_schedule_defaults,
)
from helion._compiler.cute.tcgen05_schedule_facts import (
    n_edge_cluster_schedule_supported,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _args(
    *, dtype: torch.dtype = torch.float8_e4m3fn, n: int = 832
) -> tuple[torch.Tensor, ...]:
    return scaled_args(512, n, 256, dtype=dtype)


def _config(**overrides: object) -> helion.Config:
    values = {
        **n_edge_cluster_schedule_defaults(),
        "block_sizes": [256, 256, 128],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 2,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
    }
    values.update(overrides)
    return helion.Config(**values)


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float16, torch.bfloat16))
def test_even_fringe_roundtrip_seeds_and_unsplit_codegen(dtype: torch.dtype) -> None:
    bound = vector_scaled_mm._bind_isolated(_args(dtype=dtype))
    state = bound.config_spec._cute_tcgen05_config
    assert state.n_edge_cluster_schedule_facts is not None
    config = _config()
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        flat, canonical = generator.canonicalize_flat(generator.flatten(config))
        assert generator.unflatten(flat) == canonical
        assert canonical.block_sizes == [256, 256, 128]
        assert state.n_edge_cluster_schedule_config_supported(canonical.config)
        seeds = state.n_edge_cluster_schedule_seed_configs()
        assert seeds
        for seed in seeds:
            _flat, normalized = generator.canonicalize_flat(generator.flatten(seed))
            assert normalized.block_sizes == seed.block_sizes
            assert state.n_edge_cluster_schedule_config_supported(normalized.config)
    source = bound.to_code(config)
    assert "cute.make_layout((2, 2, 1))" in source
    assert "CtaGroup.TWO" in source
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source
    assert "tile_offset_1 < cutlass.Int32(832)" in source
    assert "if tcgen05_full_tile:" in source
    assert "cute.elem_less(_coord, (" in source
    assert "tcgen05_scheduler_warp_full_tile" not in source
    assert "tcgen05_sched_pipeline =" not in source
    assert "_full_work_tile" not in source


def test_sm103_keeps_even_n_cluster_schedule() -> None:
    capability = (10, 3)
    with blackwell_capability(capability):
        bound = vector_scaled_mm._bind_isolated(_args())
        assert bound.config_spec.target_device_capability == capability
        state = bound.config_spec._cute_tcgen05_config
        assert state.n_edge_cluster_schedule_facts is not None
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            _flat, canonical = generation.canonicalize_flat(
                generation.flatten(_config())
            )
            assert state.n_edge_cluster_schedule_config_supported(canonical.config)
            assert state.n_edge_cluster_schedule_seed_configs()
        source = bound.to_code(canonical)
        assert "cute.make_layout((2, 2, 1))" in source
        assert "PipelineTmaUmma.create" in source
        assert "PipelineTmaStore.create" in source
        assert "tcgen05_sched_pipeline =" not in source


@pytest.mark.parametrize(
    "override",
    (
        {
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
        },
        {"tcgen05_l2_swizzle_size": 4},
        {"l2_groupings": [2]},
        {"block_sizes": [256, 128, 128]},  # Seven N tiles: padded peer.
    ),
)
def test_unsafe_explicit_schedule_is_rejected(override: dict[str, object]) -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    config = _config(**override)
    assert not bound.config_spec._cute_tcgen05_config.n_edge_cluster_schedule_config_supported(
        config.config
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="N-edge cluster_n=2"):
        bound.to_code(config)


@pytest.mark.parametrize("unproved", ("unaligned", "mn_major", "extra_store"))
def test_native_graph_proof_does_not_admit_unproved_inputs(unproved: str) -> None:
    args = list(_args())
    kernel = vector_scaled_mm
    if unproved == "unaligned":
        args[0] = torch.empty(512 * 256 + 1, dtype=args[0].dtype)[1:].view(512, 256)
    elif unproved == "mn_major":
        args[1] = args[1].contiguous()
    else:
        kernel = extra_store_mm
        args = args[:2]
    bound = kernel._bind_isolated(tuple(args))
    assert bound.config_spec._cute_tcgen05_config.n_edge_cluster_schedule_facts is None


@pytest.mark.parametrize(
    ("shape", "tile", "override"),
    (
        ((512, 832, 256), (256, 256, 128), {"tcgen05_cluster_m": True}),
        ((511, 832, 256), (256, 256, 128), {}),
        ((512, 832, 288), (256, 256, 128), {}),
        ((512, 832, 256), (256, 128, 128), {}),
        ((512, 768, 256), (256, 256, 128), {}),
        ((512, 832, 256), (256, 256, 128), {"tcgen05_warp_spec_c_input_warps": 1}),
        (
            (512, 832, 256),
            (256, 256, 128),
            {"tcgen05_persistence_model": "clc_persistent"},
        ),
    ),
)
def test_shared_admission_rejects_unsafe_geometry_or_protocol(
    shape: tuple[int, int, int], tile: tuple[int, int, int], override: dict[str, object]
) -> None:
    facts = Tcgen05FullTileScheduleFacts(
        (0, 1, 2), shape, torch.float8_e4m3fn, rhs_k_major=True
    )
    assert not n_edge_cluster_schedule_supported(
        facts, tile, _config(**override).config
    )


def test_search_projects_phase_split_onto_the_proved_unsplit_protocol() -> None:
    bound = vector_scaled_mm._bind_isolated(_args())
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        candidate = _config(
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            l2_groupings=[2],
        )
        _flat, canonical = generator.canonicalize_flat(generator.flatten(candidate))
        assert bound.config_spec._cute_tcgen05_config.n_edge_cluster_schedule_config_supported(
            canonical.config
        )
        assert canonical.config["tcgen05_cluster_n"] == 2


def _unregistered_n_edge(
    *reasons: str,
) -> tuple[BoundKernel, Tcgen05FullTileScheduleFacts]:
    """Undo the N-edge registration, then veto both persistent pid types."""
    bound = vector_scaled_mm._bind_isolated(_args())
    state = bound.config_spec._cute_tcgen05_config
    facts = state.n_edge_cluster_schedule_facts
    assert facts is not None
    state.n_edge_cluster_schedule_facts = None
    state.n_edge_cluster_restored_pid_types = ()
    for pid in ("persistent_interleaved", "persistent_blocked"):
        for reason in reasons:
            bound.config_spec.disallow_pid_type(pid, reason=reason)
    return bound, facts


@pytest.mark.parametrize("first_owned_reason", (False, True))
def test_unrelated_persistence_restriction_is_preserved(
    first_owned_reason: bool,
) -> None:
    # Positive control: registration lifts the CuTe-owned veto alone.
    bound, facts = _unregistered_n_edge(_TWO_CTA_PERSISTENT_PID_VETO)
    state = bound.config_spec._cute_tcgen05_config
    state.register_n_edge_cluster_schedule_facts(facts)
    assert state.n_edge_cluster_schedule_facts == facts
    assert "persistent_interleaved" in bound.config_spec.allowed_pid_types
    owned = (_TWO_CTA_PERSISTENT_PID_VETO,) if first_owned_reason else ()
    bound, facts = _unregistered_n_edge(*owned, "another loop owns this restriction")
    state = bound.config_spec._cute_tcgen05_config
    state.register_n_edge_cluster_schedule_facts(facts)
    assert state.n_edge_cluster_schedule_facts is None
    assert "persistent_interleaved" not in bound.config_spec.allowed_pid_types
