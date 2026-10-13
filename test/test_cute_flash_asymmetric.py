"""Asymmetric head dims on the fused CuTe flash path (MLA prefill: 192/128).

Q and K are 192 wide, V and O 128 wide. The detector reads the Q/K width from
K's host tensor and the V/O width from the PV output; only the plain FA4
family hosts the pair (sQ and the aliased K/V ring follow the Q/K width, the
V TMA box, TMEM output accumulators and epilogue follow the V width), so the
resolver pins every request to it and the search surface collapses onto it.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import FLASH_HEAD_DIM_PAIRS
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import flash_head_dims_supported
from helion._testing import patch_cute_mma_support
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

FAMILY_KEY = cute_flash.FLASH_PIPELINE_FAMILY_KEY
HEAD_DIM = 192
HEAD_DIM_V = 128


@helion.kernel(backend="cute", static_shapes=True)
def cute_causal_attention_dv(q_in, k_in, v_in):
    """Causal flash envelope whose V/O head dim differs from the Q/K head dim."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    v_dim = hl.specialize(v_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, v_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty(
        [q_view.size(0), m_dim, v_dim], dtype=q_in.dtype, device=q_in.device
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, v_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view([q_in.size(0), q_in.size(1), m_dim, v_dim])


def _cpu_surface(
    head_dim: int, head_dim_v: int, seq: int
) -> tuple[bool, int | None, int | None]:
    """Bind the kernel on CPU tensors and report the flash search surface."""
    q = torch.randn(2, 4, seq, head_dim, dtype=torch.bfloat16)
    k = torch.randn(2, 4, seq, head_dim, dtype=torch.bfloat16)
    v = torch.randn(2, 4, seq, head_dim_v, dtype=torch.bfloat16)
    cute_causal_attention_dv.reset()
    with patch_cute_mma_support():
        bound = cute_causal_attention_dv.bind((q, k, v))
    spec = bound.config_spec
    return (
        spec.cute_flash_search_enabled,
        spec._cute_flash_head_dim,
        spec._cute_flash_head_dim_v,
    )


def _resolve(
    config: dict[str, object] | None,
    *,
    head_dim: int = HEAD_DIM,
    head_dim_v: int | None = HEAD_DIM_V,
    num_kv: int = 32,
    is_causal: bool = True,
) -> cute_flash.FlashAttentionConfig:
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            config,
            head_dim_v=head_dim_v,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
        )


def _fragments(
    *,
    head_dim: int = HEAD_DIM,
    head_dim_v: int | None = HEAD_DIM_V,
    num_kv: int = 32,
    is_causal: bool = True,
) -> dict[str, ConfigSpecFragment]:
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            head_dim,
            num_kv,
            head_dim_v=head_dim_v,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_head_dim_pairs() -> None:
    assert {(64, 64), (128, 128), (192, 128)} == FLASH_HEAD_DIM_PAIRS
    assert flash_head_dims_supported(192, 128)
    assert not flash_head_dims_supported(192, 192)
    assert not flash_head_dims_supported(128, 64)
    assert not flash_head_dims_supported(256, 128)


def test_score_plan_value_head_dim() -> None:
    symmetric = AttentionScorePlan(head_dim=128, qk_scale_log2=0.125)
    assert symmetric.value_head_dim == 128
    assert not symmetric.asymmetric_head_dims
    asymmetric = AttentionScorePlan(head_dim=192, qk_scale_log2=0.1, head_dim_v=128)
    assert asymmetric.value_head_dim == 128
    assert asymmetric.asymmetric_head_dims
    # ``head_dim_v == head_dim`` is the symmetric reading, not a distinct plan.
    assert not AttentionScorePlan(
        head_dim=128, qk_scale_log2=0.125, head_dim_v=128
    ).asymmetric_head_dims


def test_detector_admits_the_mla_pair_on_cpu() -> None:
    assert _cpu_surface(192, 128, 512) == (True, 192, 128)


def test_detector_keeps_the_symmetric_surface_unchanged() -> None:
    assert _cpu_surface(128, 128, 512) == (True, 128, None)


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v", "seq"),
    (
        pytest.param(192, 192, 512, id="192-192-no-tmem-room"),
        pytest.param(128, 64, 512, id="128-64-unsupported-pair"),
        pytest.param(192, 128, 384, id="192-128-odd-kv-tiles"),
    ),
)
def test_detector_rejects_other_pairs(head_dim: int, head_dim_v: int, seq: int) -> None:
    assert _cpu_surface(head_dim, head_dim_v, seq) == (False, None, None)


# --------------------------------------------------------------------------
# resolver and search surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested",
    (
        None,
        "fa4",
        "fa4_alt",
        "fa4_deep_1cta",
        "fa4_2cta",
        "fa4_2cta_causal",
        "fa4_clc",
        "ws_overlap",
        "row_mma",
    ),
)
def test_resolver_pins_the_plain_fa4_family(requested: str | None) -> None:
    config = None if requested is None else {FAMILY_KEY: requested}
    resolved = _resolve(config)
    assert resolved.pipeline_family == "fa4"
    assert resolved.topology == "fa4"
    assert resolved.q_tile_count == 2
    assert not resolved.use_2cta_instrs
    assert not resolved.use_cga2_local_cta
    assert not resolved.separate_kv_rings
    assert not resolved.alternating_warpgroups
    assert not resolved.use_clc_scheduler
    # No head-dim keyed compound packet exists for the pair.
    assert resolved.exp2_packet == "1x1"


def test_resolver_pins_fa4_for_the_dense_pair_too() -> None:
    resolved = _resolve({FAMILY_KEY: "fa4_alt"}, is_causal=False)
    assert resolved.pipeline_family == "fa4"
    assert not resolved.alternating_warpgroups


def test_symmetric_resolution_is_unchanged() -> None:
    resolved = _resolve(
        {FAMILY_KEY: "fa4_alt"}, head_dim=128, head_dim_v=None, is_causal=False
    )
    assert resolved.pipeline_family == "fa4_alt"
    assert resolved.alternating_warpgroups
    # A redundant ``head_dim_v == head_dim`` is the symmetric reading.
    same = _resolve(
        {FAMILY_KEY: "fa4_alt"}, head_dim=128, head_dim_v=128, is_causal=False
    )
    assert same == resolved


def test_surface_searches_only_the_fa4_family() -> None:
    fragments = _fragments()
    assert _active(fragments[FAMILY_KEY]) == ("fa4",)
    # The causal resident packet is keyed on the 128-wide value tiles (its
    # body never touches the Q/K width); the plain cadence packets stay off.
    assert _active(fragments[cute_flash.FLASH_EXP2_PACKET_KEY]) == (
        "1x1",
        "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
    )
    assert _active(fragments[cute_flash.FLASH_Q_TILE_M_KEY]) == (128,)
    assert fragments[FAMILY_KEY].default() == "fa4"


def test_seeds_resolve_to_the_fa4_family() -> None:
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            HEAD_DIM,
            32,
            head_dim_v=HEAD_DIM_V,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
        )
    assert seeds
    assert len(seeds) == len({seed.to_json() for seed in seeds})
    resolved_seeds = [_resolve(seed.config) for seed in seeds]
    for resolved in resolved_seeds:
        assert resolved.pipeline_family == "fa4"
        # Either the direct-store two-slot ring or FA4's structure for the
        # pair: the uneven ring with the TMA output staged in the Q stages.
        if resolved.uneven_kv_ring:
            assert resolved.kv_stage == 3
            assert resolved.o_over_q and resolved.epi_tma
        else:
            assert resolved.kv_stage == 2
            assert not resolved.epi_tma and not resolved.epi_stg
    staged = [resolved for resolved in resolved_seeds if resolved.uneven_kv_ring]
    assert len(staged) == 3
    resident = "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"
    staged_packets = sorted(
        (resolved.exp2_packet, resolved.e2e_schedule, resolved.softmax_regs)
        for resolved in staged
    )
    # The measured best (hd128 winner registers), the packet on the defaults,
    # and the XU cadence without the packet.
    assert staged_packets == [
        ("1x1", "xu", 200),
        (resident, "16/6", 160),
        (resident, "16/6", 200),
    ]
    for resolved in staged:
        assert resolved.causal_kv_order == "descending"
        assert resolved.causal_loop_split
        assert resolved.corr_tile_size == 16
    # The direct-store fallbacks: the descending loop-split schedule with the
    # default and the XU cadence (plus the packet's generic seed).
    direct_schedules = sorted(
        resolved.e2e_schedule
        for resolved in resolved_seeds
        if not resolved.uneven_kv_ring
        and resolved.causal_kv_order == "descending"
        and resolved.causal_loop_split
    )
    assert direct_schedules == ["16/6", "8/2", "xu"]


# --------------------------------------------------------------------------
# shared-memory budget: two-deep aliased ring of K-wide slots, direct output
# --------------------------------------------------------------------------


def test_resolver_caps_the_ring_and_drops_the_staged_output() -> None:
    resolved = _resolve(
        {
            cute_flash.FLASH_KV_STAGE_KEY: 3,
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_EPI_STG_KEY: True,
            cute_flash.FLASH_EPI_STG_STORE_KEY: "whole",
        }
    )
    # 2 x 128 x 192 Q + 2 x 128 x 192 ring slots = 192 KiB; a third slot or two
    # staged 128 x 128 O tiles would exceed the 227 KiB budget.
    assert resolved.kv_stage == 2
    assert not resolved.epi_tma
    assert not resolved.epi_stg
    assert resolved.epi_stg_store == "slice"
    assert resolved.corr_tile_size == 16
    assert resolved.rescale_chunk_cols == 16


def test_surface_pins_the_ring_depth_and_the_direct_output() -> None:
    fragments = _fragments()
    # The ring depth is pinned (the uneven ring is its own knob); a staged
    # output is searched only because the Q-stage staging knob can host it,
    # and resolves to the direct store without that knob.
    assert _active(fragments[cute_flash.FLASH_KV_STAGE_KEY]) == (2,)
    assert _active(fragments[cute_flash.FLASH_O_OVER_Q_KEY]) == (False, True)
    resolved = _resolve({cute_flash.FLASH_EPI_TMA_KEY: True})
    assert not resolved.epi_tma
    assert not resolved.epi_stg
    assert resolved.kv_stage == 2


def test_schedule_byte_model_matches_the_runtime_storage() -> None:
    import dataclasses
    import importlib

    from helion._compiler.cute.flash_schedule import FlashScheduleError
    from helion._compiler.cute.flash_schedule import FlashScheduleSpec
    from helion._compiler.cute.flash_schedule import build_fa4_schedule
    from helion._compiler.cute.flash_schedule import max_fa4_kv_depth
    from helion._compiler.cute.flash_schedule import verify_flash_schedule

    cutlass = importlib.import_module("cutlass")
    flash_fa4_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_fa4_shared_storage

    spec = FlashScheduleSpec(
        HEAD_DIM, 2, stage_output=False, head_dim_v=HEAD_DIM_V, causal=True
    )
    schedule = verify_flash_schedule(build_fa4_schedule(spec)).schedule
    q_bytes = 2 * 128 * HEAD_DIM * 2
    ring_bytes = 2 * 128 * HEAD_DIM * 2
    assert schedule.shared_memory_bytes == q_bytes + ring_bytes + 3072
    assert schedule.tmem_columns == 512
    regions = {region.name: region for region in schedule.memory_regions}
    assert (regions["K_r0"].offset, regions["K_r0"].extent) == (q_bytes, ring_bytes)
    assert (regions["V_r0"].offset, regions["V_r0"].extent) == (q_bytes, ring_bytes)
    assert (regions["O_r0_q1"].offset, regions["O_r0_q1"].extent) == (384, 128)
    assert max_fa4_kv_depth(spec) == 2
    assert max_fa4_kv_depth(dataclasses.replace(spec, stage_output=True)) == 1
    storage = flash_fa4_shared_storage(
        HEAD_DIM,
        2,
        2,
        2,
        cutlass.BFloat16,
        False,
        False,
        1,
        False,
        128,
        head_dim_v=HEAD_DIM_V,
    )
    assert storage.size_in_bytes() == schedule.shared_memory_bytes
    with pytest.raises(FlashScheduleError, match="shared-memory capacity"):
        verify_flash_schedule(build_fa4_schedule(dataclasses.replace(spec, kv_depth=3)))
    with pytest.raises(FlashScheduleError, match="192/128"):
        build_fa4_schedule(FlashScheduleSpec(HEAD_DIM, 2, head_dim_v=192))
    with pytest.raises(FlashScheduleError, match="aliased"):
        build_fa4_schedule(
            FlashScheduleSpec(HEAD_DIM, 2, head_dim_v=HEAD_DIM_V, separate_kv=True)
        )
    # The symmetric model is unchanged.
    assert (
        build_fa4_schedule(FlashScheduleSpec(128, 2)).shared_memory_bytes
        == build_fa4_schedule(
            FlashScheduleSpec(128, 2, head_dim_v=128)
        ).shared_memory_bytes
    )


# --------------------------------------------------------------------------
# FA4's uneven [K-wide, V-wide, K-wide] three-slot ring
# --------------------------------------------------------------------------


def test_uneven_ring_knob_sets_the_three_slot_depth() -> None:
    resolved = _resolve({cute_flash.FLASH_UNEVEN_KV_RING_KEY: True})
    assert resolved.uneven_kv_ring
    assert resolved.kv_stage == 3
    assert resolved.pipeline_family == "fa4"
    assert not resolved.epi_tma
    # The knob is the depth's only route to three: a requested depth of two
    # (or anything else) is overridden while the ring is uneven.
    resolved = _resolve(
        {
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_KV_STAGE_KEY: 2,
        }
    )
    assert resolved.uneven_kv_ring
    assert resolved.kv_stage == 3
    # Without it the pair keeps the capped two-slot ring.
    resolved = _resolve({cute_flash.FLASH_KV_STAGE_KEY: 3})
    assert not resolved.uneven_kv_ring
    assert resolved.kv_stage == 2


def test_uneven_ring_drops_when_the_staged_output_does_not_fit() -> None:
    # Two staged 128 x 128 output tiles next to the 96 KiB of Q leave no room
    # for the 128 KiB uneven ring; the staged output is dropped first (the
    # existing asymmetric rule) and the uneven ring then fits.
    resolved = _resolve(
        {
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
        }
    )
    assert not resolved.epi_tma
    assert resolved.uneven_kv_ring
    assert resolved.kv_stage == 3


def test_uneven_ring_is_rejected_off_the_pair_and_off_the_ptx_mma() -> None:
    symmetric = _resolve(
        {cute_flash.FLASH_UNEVEN_KV_RING_KEY: True}, head_dim=128, head_dim_v=None
    )
    assert not symmetric.uneven_kv_ring
    assert symmetric.kv_stage == _resolve({}, head_dim=128, head_dim_v=None).kv_stage
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            HEAD_DIM,
            32,
            {cute_flash.FLASH_UNEVEN_KV_RING_KEY: True},
            head_dim_v=HEAD_DIM_V,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
        )
    assert not resolved.mma_ptx
    assert not resolved.uneven_kv_ring
    assert resolved.kv_stage == 2


def test_uneven_ring_is_searched_only_for_the_pair() -> None:
    fragments = _fragments()
    assert _active(fragments[cute_flash.FLASH_UNEVEN_KV_RING_KEY]) == (False, True)
    # The ring depth itself stays pinned; three is a legal resolved value.
    kv_stage = fragments[cute_flash.FLASH_KV_STAGE_KEY]
    assert _active(kv_stage) == (2,)
    assert tuple(kv_stage.choices) == (2, 3)  # pyrefly: ignore[missing-attribute]
    symmetric = _fragments(head_dim=128, head_dim_v=None)
    assert _active(symmetric[cute_flash.FLASH_UNEVEN_KV_RING_KEY]) == (False,)
    assert tuple(symmetric[cute_flash.FLASH_KV_STAGE_KEY].choices) == tuple(  # pyrefly: ignore[missing-attribute]
        _fragments(head_dim=128, head_dim_v=128)[cute_flash.FLASH_KV_STAGE_KEY].choices  # pyrefly: ignore[missing-attribute]
    )


def test_uneven_ring_byte_model_matches_the_runtime_storage() -> None:
    import dataclasses
    import importlib

    from helion._compiler.cute.flash_schedule import FlashScheduleError
    from helion._compiler.cute.flash_schedule import FlashScheduleSpec
    from helion._compiler.cute.flash_schedule import build_fa4_schedule
    from helion._compiler.cute.flash_schedule import max_fa4_kv_depth
    from helion._compiler.cute.flash_schedule import uneven_kv_ring_shift_bytes
    from helion._compiler.cute.flash_schedule import uneven_kv_ring_stage_stride_bytes
    from helion._compiler.cute.flash_schedule import verify_flash_schedule

    cutlass = importlib.import_module("cutlass")
    flash_fa4_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_fa4_shared_storage

    spec = FlashScheduleSpec(
        HEAD_DIM,
        3,
        stage_output=False,
        head_dim_v=HEAD_DIM_V,
        causal=True,
        uneven_kv_ring=True,
    )
    schedule = verify_flash_schedule(build_fa4_schedule(spec)).schedule
    q_bytes = 2 * 128 * HEAD_DIM * 2
    # [48, 32, 48] KiB: two stage strides of (48 + 32) / 2 KiB plus one K slot.
    assert uneven_kv_ring_stage_stride_bytes(spec) == 40 * 1024
    assert uneven_kv_ring_shift_bytes(spec) == 8 * 1024
    ring_bytes = 2 * 40 * 1024 + 48 * 1024
    assert ring_bytes == 128 * 1024
    assert schedule.shared_memory_bytes == q_bytes + ring_bytes + 3072 == 232448
    regions = {region.name: region for region in schedule.memory_regions}
    assert (regions["K_r0"].offset, regions["K_r0"].extent) == (q_bytes, ring_bytes)
    assert (regions["V_r0"].offset, regions["V_r0"].extent) == (q_bytes, ring_bytes)
    assert max_fa4_kv_depth(spec) == 3
    # With staged output the uneven ring does not fit and the depth is the
    # plain ring's; without the pair or with the wrong depth it is rejected.
    assert max_fa4_kv_depth(dataclasses.replace(spec, stage_output=True)) == 1
    storage = flash_fa4_shared_storage(
        HEAD_DIM,
        3,
        2,
        2,
        cutlass.BFloat16,
        False,
        False,
        1,
        False,
        128,
        head_dim_v=HEAD_DIM_V,
        uneven_kv_ring=True,
    )
    assert storage.size_in_bytes() == schedule.shared_memory_bytes
    with pytest.raises(FlashScheduleError, match="three slots"):
        build_fa4_schedule(dataclasses.replace(spec, kv_depth=2))
    with pytest.raises(FlashScheduleError, match="asymmetric"):
        build_fa4_schedule(FlashScheduleSpec(128, 3, uneven_kv_ring=True))
    with pytest.raises(FlashScheduleError, match="aliased"):
        build_fa4_schedule(dataclasses.replace(spec, separate_kv=True))
    with pytest.raises(FlashScheduleError, match="shared-memory capacity"):
        verify_flash_schedule(
            build_fa4_schedule(dataclasses.replace(spec, stage_output=True))
        )


# --------------------------------------------------------------------------
# FA4's overlap_sO_sQ: the staged output tiles live in the dead Q stages
# --------------------------------------------------------------------------


def test_o_over_q_makes_the_staged_output_legal_for_the_pair() -> None:
    resolved = _resolve(
        {cute_flash.FLASH_O_OVER_Q_KEY: True, cute_flash.FLASH_EPI_TMA_KEY: True}
    )
    assert resolved.o_over_q
    assert resolved.epi_tma
    assert resolved.kv_stage == 2
    # Without it the pair still drops the staged output (no room for sO).
    resolved = _resolve({cute_flash.FLASH_EPI_TMA_KEY: True})
    assert not resolved.o_over_q
    assert not resolved.epi_tma
    # It needs a staged output to alias.
    resolved = _resolve({cute_flash.FLASH_O_OVER_Q_KEY: True})
    assert not resolved.o_over_q
    assert not resolved.epi_tma
    assert not resolved.epi_stg


def test_o_over_q_composes_with_the_uneven_ring() -> None:
    resolved = _resolve(
        {
            cute_flash.FLASH_O_OVER_Q_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
        }
    )
    assert resolved.o_over_q
    assert resolved.epi_tma
    assert resolved.uneven_kv_ring
    assert resolved.kv_stage == 3
    resolved = _resolve(
        {
            cute_flash.FLASH_O_OVER_Q_KEY: True,
            cute_flash.FLASH_EPI_STG_KEY: True,
            cute_flash.FLASH_EPI_STG_STORE_KEY: "whole",
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
        }
    )
    assert resolved.o_over_q
    assert resolved.epi_stg
    assert resolved.epi_stg_store == "whole"
    assert resolved.uneven_kv_ring


def test_o_over_q_needs_a_non_persistent_one_cta_fa4_body() -> None:
    # The dense pair defaults to the persistent scheduler: the next item's Q
    # tile would land while the output drains.
    dense = _resolve(
        {cute_flash.FLASH_O_OVER_Q_KEY: True, cute_flash.FLASH_EPI_TMA_KEY: True},
        is_causal=False,
    )
    assert dense.persistent
    assert not dense.o_over_q
    dense = _resolve(
        {
            cute_flash.FLASH_O_OVER_Q_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_PERSISTENT_KEY: False,
        },
        is_causal=False,
    )
    assert not dense.persistent
    assert dense.o_over_q
    assert dense.epi_tma
    # Symmetric shapes accept the value (a non-persistent fa4 body with a
    # staged output) but do not search it.
    symmetric = _resolve(
        {cute_flash.FLASH_O_OVER_Q_KEY: True, cute_flash.FLASH_EPI_TMA_KEY: True},
        head_dim=128,
        head_dim_v=None,
    )
    assert symmetric.o_over_q
    assert symmetric.epi_tma
    assert not _resolve(
        {cute_flash.FLASH_EPI_TMA_KEY: True}, head_dim=128, head_dim_v=None
    ).o_over_q


def test_o_over_q_opens_the_staged_epilogue_search_for_the_pair() -> None:
    fragments = _fragments()
    assert _active(fragments[cute_flash.FLASH_O_OVER_Q_KEY]) == (False, True)
    assert set(_active(fragments[cute_flash.FLASH_EPI_TMA_KEY])) == {False, True}
    assert set(_active(fragments[cute_flash.FLASH_EPI_STG_KEY])) == {False, True}
    symmetric = _fragments(head_dim=128, head_dim_v=None)
    assert _active(symmetric[cute_flash.FLASH_O_OVER_Q_KEY]) == (False,)


def test_o_over_q_schedule_aliases_the_output_tiles_on_the_q_stages() -> None:
    import dataclasses
    import importlib

    from helion._compiler.cute.flash_schedule import FlashScheduleError
    from helion._compiler.cute.flash_schedule import FlashScheduleSpec
    from helion._compiler.cute.flash_schedule import build_fa4_schedule
    from helion._compiler.cute.flash_schedule import max_fa4_kv_depth
    from helion._compiler.cute.flash_schedule import verify_flash_schedule

    cutlass = importlib.import_module("cutlass")
    flash_fa4_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_fa4_shared_storage

    spec = FlashScheduleSpec(
        HEAD_DIM,
        3,
        stage_output=True,
        output_over_query=True,
        head_dim_v=HEAD_DIM_V,
        causal=True,
        uneven_kv_ring=True,
    )
    schedule = verify_flash_schedule(build_fa4_schedule(spec)).schedule
    # The staged tiles cost nothing: the uneven ring still fits exactly.
    assert schedule.shared_memory_bytes == 232448
    regions = {region.name: region for region in schedule.memory_regions}
    q_tile = 128 * HEAD_DIM * 2
    o_tile = 128 * HEAD_DIM_V * 2
    assert (regions["Q_r0_q1"].offset, regions["Q_r0_q1"].extent) == (q_tile, q_tile)
    assert (regions["O_stage_r0_q0"].offset, regions["O_stage_r0_q0"].extent) == (
        0,
        o_tile,
    )
    assert (regions["O_stage_r0_q1"].offset, regions["O_stage_r0_q1"].extent) == (
        o_tile,
        o_tile,
    )
    assert regions["Q_r0_q0"].alias_group == regions["O_stage_r0_q1"].alias_group
    assert max_fa4_kv_depth(spec) == 3
    assert max_fa4_kv_depth(dataclasses.replace(spec, uneven_kv_ring=False)) == 2
    # The runtime storage has no separate sO for this layout.
    storage = flash_fa4_shared_storage(
        HEAD_DIM,
        3,
        2,
        2,
        cutlass.BFloat16,
        False,
        False,
        1,
        False,
        128,
        head_dim_v=HEAD_DIM_V,
        uneven_kv_ring=True,
    )
    assert storage.size_in_bytes() == schedule.shared_memory_bytes
    with pytest.raises(FlashScheduleError, match="non-persistent"):
        build_fa4_schedule(
            FlashScheduleSpec(
                128,
                2,
                stage_output=True,
                output_over_query=True,
                persistent=True,
                kv_iterations=8,
            )
        )
    with pytest.raises(FlashScheduleError, match="staged output"):
        build_fa4_schedule(dataclasses.replace(spec, stage_output=False))
    with pytest.raises(FlashScheduleError, match="one-CTA"):
        build_fa4_schedule(
            FlashScheduleSpec(
                128,
                2,
                stage_output=True,
                output_over_query=True,
                cta_count=2,
                multicast_kv=True,
                cooperative_mma=True,
            )
        )


# --------------------------------------------------------------------------
# the causal resident packet, keyed on the 128-wide value tiles
# --------------------------------------------------------------------------

RESIDENT_PACKET = "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"


def test_resident_packet_resolves_for_the_pair() -> None:
    resolved = _resolve({cute_flash.FLASH_EXP2_PACKET_KEY: RESIDENT_PACKET})
    assert resolved.exp2_packet == RESIDENT_PACKET
    # The packet pins its measured structure (the hd128 compound overrides).
    assert resolved.e2e_schedule == "16/6"
    assert resolved.disc_pipe_depth == 2
    assert resolved.causal_kv_order == "descending"
    assert resolved.causal_loop_split
    assert resolved.pipeline_family == "fa4"
    # It composes with the uneven ring and the Q-staged output.
    resolved = _resolve(
        {
            cute_flash.FLASH_EXP2_PACKET_KEY: RESIDENT_PACKET,
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_O_OVER_Q_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
        }
    )
    assert resolved.exp2_packet == RESIDENT_PACKET
    assert resolved.uneven_kv_ring and resolved.o_over_q and resolved.epi_tma
    # The plain cadence packets stay off the pair; the dense pair has none.
    assert _resolve({cute_flash.FLASH_EXP2_PACKET_KEY: "8x1"}).exp2_packet == "1x1"
    assert (
        _resolve(
            {cute_flash.FLASH_EXP2_PACKET_KEY: RESIDENT_PACKET}, is_causal=False
        ).exp2_packet
        == "1x1"
    )
    assert _active(_fragments(is_causal=False)[cute_flash.FLASH_EXP2_PACKET_KEY]) == (
        "1x1",
    )


def test_resident_packet_overrides_key_on_the_value_head_dim() -> None:
    overrides = cute_flash._flash_compound_exp2_packet_overrides(
        HEAD_DIM,
        32,
        {cute_flash.FLASH_EXP2_PACKET_KEY: RESIDENT_PACKET},
        dtype=torch.bfloat16,
        is_causal=True,
        standard_causal_output=True,
        head_dim_v=HEAD_DIM_V,
    )
    assert overrides[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
    assert overrides[cute_flash.FLASH_E2E_SCHEDULE_KEY] == "16/6"
    # Without the value width the 192-wide request is not a resident shape.
    assert (
        cute_flash._flash_compound_exp2_packet_overrides(
            HEAD_DIM,
            32,
            {cute_flash.FLASH_EXP2_PACKET_KEY: RESIDENT_PACKET},
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
        == {}
    )


def test_resident_packet_gate_follows_the_hd128_causal_one() -> None:
    # The pair hosts the causal resident packet exactly where the hd128 causal
    # bf16 surface does: same eligibility (output shape, pruning, the small
    # biased path, dtype, family) and the same search standing, in the
    # fragments and in the resolver. A change to the hd128 gate (a
    # fast_math-only packet, say) has to move the pair's gate with it.
    key = cute_flash.FLASH_EXP2_PACKET_KEY
    variants: tuple[dict[str, object], ...] = (
        {},
        {"has_kv_tile_pruning": True},
        {"small_biased_candidate": True},
        {"dtype": torch.float16},
        {"standard_causal_output": False},
    )
    standings: set[bool] = set()
    for extra in variants:
        kwargs: dict[str, object] = {
            "dtype": torch.bfloat16,
            "num_bh": 64,
            "is_causal": True,
            "standard_causal_output": True,
            **extra,
        }
        with patch.dict(os.environ, {}, clear=True):
            hd128 = cute_flash.flash_autotune_fragments(128, 32, **kwargs)[key]
            pair = cute_flash.flash_autotune_fragments(
                HEAD_DIM, 32, head_dim_v=HEAD_DIM_V, **kwargs
            )[key]
            hd128_resolved = cute_flash.resolve_flash_config(
                128, 32, {key: RESIDENT_PACKET}, **kwargs
            ).exp2_packet
            pair_resolved = cute_flash.resolve_flash_config(
                HEAD_DIM, 32, {key: RESIDENT_PACKET}, head_dim_v=HEAD_DIM_V, **kwargs
            ).exp2_packet
        hd128_choices = tuple(hd128.choices)  # pyrefly: ignore[missing-attribute]
        pair_choices = tuple(pair.choices)  # pyrefly: ignore[missing-attribute]
        legal = RESIDENT_PACKET in hd128_choices
        searched = RESIDENT_PACKET in _active(hd128)
        assert (RESIDENT_PACKET in pair_choices) == legal, extra
        assert (RESIDENT_PACKET in _active(pair)) == searched, extra
        assert (pair_resolved == RESIDENT_PACKET) == (
            hd128_resolved == RESIDENT_PACKET
        ), extra
        standings.add(searched)
    # The variants exercise both standings, so the pins are not vacuous.
    assert standings == {False, True}


# --------------------------------------------------------------------------
# structural coverage design: the pinned-fa4 surface has no uncoverable rows
# --------------------------------------------------------------------------


def _config_spec(num_kv: int = 32, *, is_causal: bool = True):
    from helion._compiler.backend import CuteBackend
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec

    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.target_device_capability = (10, 0)
    spec.enable_cute_flash_search(
        head_dim=HEAD_DIM,
        head_dim_v=HEAD_DIM_V,
        num_kv=num_kv,
        num_bh=64,
        dtype=torch.bfloat16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=is_causal,
        standard_causal_output=is_causal,
        standard_dense_output=not is_causal,
    )
    return spec


def test_surface_drops_values_only_other_bodies_realize() -> None:
    fragments = _fragments()
    assert _active(fragments[cute_flash.FLASH_S_STAGE_KEY]) == (2,)
    assert _active(fragments[cute_flash.FLASH_PACKED_REDUCE_KEY]) == (True,)


def test_dense_pair_keeps_the_direct_output_surface() -> None:
    # The dense pair's default body is persistent, so the Q-stage output
    # staging (and with it the staged epilogue) is not searched there.
    fragments = _fragments(is_causal=False)
    assert _active(fragments[cute_flash.FLASH_O_OVER_Q_KEY]) == (False,)
    assert _active(fragments[cute_flash.FLASH_EPI_TMA_KEY]) == (False,)
    assert _active(fragments[cute_flash.FLASH_EPI_STG_KEY]) == (False,)
    assert _active(fragments[cute_flash.FLASH_UNEVEN_KV_RING_KEY]) == (False, True)


@pytest.mark.parametrize("is_causal", (True, False), ids=("causal", "dense"))
def test_structural_coverage_design_is_complete(is_causal: bool) -> None:
    from helion.autotuner.config_generation import ConfigGeneration

    spec = _config_spec(is_causal=is_causal)
    with patch.dict(os.environ, {}, clear=True):
        generation = ConfigGeneration(spec)
        generation.validate_flash_structural_coverage()
        configs = generation.flash_deterministic_population_configs()
        assert configs
        staged = 0
        for config in configs:
            resolved = spec._resolve_cute_flash_config(config.config)
            assert resolved.pipeline_family == "fa4"
            # The plain two-slot ring, or FA4's uneven three-slot ring.
            assert resolved.kv_stage == (3 if resolved.uneven_kv_ring else 2)
            # A staged output only through the dead Q stages.
            assert resolved.o_over_q == (resolved.epi_tma or resolved.epi_stg)
            staged += resolved.o_over_q
        assert (staged > 0) == is_causal
