"""The alternating-warpgroup flash pipeline family (``fa4_alt``).

One 128-row query tile per persistent work item. The two softmax warpgroups own
alternating KV steps on their own score and probability buffers, so the QK of
step i+2 is issued as soon as the scores of step i were read into registers
instead of after the PV of step i; a QK warp and a PV warp issue the two MMA
streams; the output tile is staged through the finished item's dead Q stage.
The numerics follow the fa4 body per element; only the row-sum association
differs (each warpgroup keeps a partial sum rescaled by every alpha).
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.cute_flash_alt import FA4_ALT_V_STAGE
from helion._compiler.cute.cute_flash_alt import emit_flash_fa4_alt_device_body
from helion._compiler.cute.cute_flash_alt import fa4_alt_supported
from helion._compiler.cute.flash_schedule import FlashBarrier
from helion._compiler.cute.flash_schedule import FlashScheduleError
from helion._compiler.cute.flash_schedule import FlashScheduleSpec
from helion._compiler.cute.flash_schedule import build_fa4_alt_schedule
from helion._compiler.cute.flash_schedule import verify_flash_schedule
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

FAMILY_KEY = cute_flash.FLASH_PIPELINE_FAMILY_KEY

# The (2, 16, 2048, 128) bf16 campaign winner (fa4 one-CTA, chunked softmax,
# 16/6 schedule, 8x2 packet), reduced to the flash keys that select its body.
_WINNER: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_kv_tile_n": 128,
    "cute_flash_s_stage": 2,
    "cute_flash_kv_stage": 3,
    "cute_flash_persistent": True,
    "cute_flash_e2e_schedule": "16/6",
    "cute_flash_e2e_offset": 3,
    "cute_flash_e2e_offset0": 11,
    "cute_flash_exp2_packet": "8x2",
    "cute_flash_mma_interleave": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_pipeline_family": "fa4",
    "cute_flash_softmax_disc": True,
    "cute_flash_disc_pipe": 2,
    "cute_flash_split_p_arrive": True,
    "cute_flash_p_store_rep": 16,
    "cute_flash_s_load_rep": 32,
    "cute_flash_first_load_order": 1,
    "cute_flash_epi_tma": True,
    "cute_flash_epi_stg": False,
    "cute_flash_rescale_threshold": 12.0,
    "cute_flash_softmax_regs": 192,
    "cute_flash_corr_regs": 64,
    "cute_flash_other_regs": 48,
    "cute_flash_corr_tile_size": 8,
}
# The same knobs on the alternating family (one emulation phase for both
# warpgroups: they process the same rows).
_ALT: dict[str, object] = {
    **_WINNER,
    FAMILY_KEY: "fa4_alt",
    "cute_flash_e2e_offset": 11,
    "cute_flash_e2e_offset0": 11,
}
_ALT_DEFAULT: dict[str, object] = {
    **_ALT,
    "cute_flash_kv_stage": 2,
    "cute_flash_e2e_schedule": "8/2",
    "cute_flash_e2e_offset": 0,
    "cute_flash_e2e_offset0": 0,
    "cute_flash_exp2_packet": "1x1",
    "cute_flash_rescale_threshold": 8.0,
    "cute_flash_corr_tile_size": 16,
}


@helion.kernel(backend="cute", static_shapes=True)
def _attention_with_lse(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty(
        [q_view.size(0), m_dim, 1], device=q_in.device, dtype=torch.float32
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m, :] = (m_i + torch.log2(l_i))[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.reshape(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def _attention_output_only(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def _fragments(
    head_dim: int = 128,
    *,
    dtype: torch.dtype = torch.bfloat16,
    is_causal: bool = False,
    num_kv: int = 16,
) -> dict[str, ConfigSpecFragment]:
    return cute_flash.flash_autotune_fragments(
        head_dim,
        num_kv,
        num_bh=32,
        dtype=dtype,
        is_causal=is_causal,
        standard_dense_output=False,
        standard_causal_output=False,
        plain_row_body=True,
    )


def _resolve(
    config: dict[str, object],
    *,
    head_dim: int = 128,
    num_kv: int = 16,
    dtype: torch.dtype | None = None,
    is_causal: bool = False,
    plain_row_body: bool = True,
) -> cute_flash.FlashAttentionConfig:
    if dtype is None:
        dtype = torch.bfloat16 if head_dim == 128 else torch.float16
    with patch.dict(os.environ, {}, clear=True):
        return cast(
            "cute_flash.FlashAttentionConfig",
            cute_flash.resolve_flash_config(
                head_dim,
                num_kv,
                config,
                dtype=dtype,
                num_bh=32,
                is_causal=is_causal,
                standard_dense_output=False,
                standard_causal_output=False,
                plain_row_body=plain_row_body,
            ),
        )


def _emit(config: dict[str, object], *, num_kv: int = 16, has_lse: bool = True) -> str:
    cfg = _resolve(config, num_kv=num_kv)
    assert cfg.alternating_warpgroups
    exp2 = cute_flash._flash_disc_exp2_codegen_params(
        cfg.exp2_packet, cfg.e2e_freq, cfg.e2e_res
    )
    body = emit_flash_fa4_alt_device_body(
        head_dim=128,
        num_kv=num_kv,
        sequence_extent=128 * num_kv,
        total_tiles=32 * num_kv,
        cfg=cfg,
        has_lse=has_lse,
        io_dtype="cutlass.BFloat16",
        lse_scale=1.0,
        exp2_pair_batch=exp2.pair_batch,
        exp2_emu_batch=exp2.emu_batch,
        exp2_degree2=exp2.degree2,
        exp2_degree1=exp2.degree1_unmasked,
        e2e_freq=exp2.e2e_freq,
        e2e_res=exp2.e2e_res,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _search_families(fragments: dict[str, ConfigSpecFragment]) -> tuple[str, ...]:
    fragment = cast("object", fragments[FAMILY_KEY])
    return tuple(fragment.search_choices or ())  # pyrefly: ignore[missing-attribute]


# --------------------------------------------------------------------------
# surface and resolver
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_dense_hd128_surface_searches_the_family(dtype: torch.dtype) -> None:
    assert "fa4_alt" in _search_families(_fragments(dtype=dtype))
    assert "fa4_alt" in cute_flash.FLASH_AUTOTUNE_PIPELINE_FAMILIES


def _seed_configs(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool = False,
) -> tuple[helion.Config, ...]:
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_attention_seed_configs(
            head_dim,
            num_kv,
            num_bh=32,
            dtype=dtype,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
            target_device_capability=(10, 0),
        )


@pytest.mark.parametrize("num_kv", (2, 16, 64))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_dense_hd128_seed_list_carries_the_family(
    dtype: torch.dtype, num_kv: int
) -> None:
    """Generation zero of a cold search holds one seed of the family on every
    length of the surface, from fragment defaults rather than a winner table:
    the deep K ring and the pinned structure, with the pass knobs at their
    searchable defaults."""
    seeds = _seed_configs(128, num_kv, dtype=dtype)
    family_seeds = [seed for seed in seeds if seed.config.get(FAMILY_KEY) == "fa4_alt"]
    assert len(family_seeds) == 1
    (seed,) = family_seeds
    assert seed.config["cute_flash_kv_stage"] == 3
    assert seed.config["cute_flash_e2e_offset"] == 0
    assert seed.config["cute_flash_exp2_packet"] == "1x1"
    resolved = _resolve(dict(seed.config), num_kv=num_kv, dtype=dtype)
    assert resolved.pipeline_family == "fa4_alt"
    assert resolved.alternating_warpgroups
    assert resolved.kv_stage == 3
    assert resolved.q_tile_count == 1
    assert resolved.persistent


@pytest.mark.parametrize(
    ("head_dim", "dtype", "is_causal"),
    ((64, torch.float16, False), (128, torch.bfloat16, True)),
    ids=("hd64", "causal_hd128"),
)
def test_other_seed_lists_do_not_carry_the_family(
    head_dim: int, dtype: torch.dtype, is_causal: bool
) -> None:
    seeds = _seed_configs(head_dim, 16, dtype=dtype, is_causal=is_causal)
    assert seeds
    assert all(seed.config.get(FAMILY_KEY) != "fa4_alt" for seed in seeds)


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "is_causal", "plain_row_body"),
    (
        (64, 16, False, True),
        (128, 16, True, True),
        (128, 15, False, True),
        (128, 1, False, True),
        (128, 16, False, False),
    ),
    ids=("hd64", "causal", "odd_kv", "one_kv", "modified_rows"),
)
def test_family_is_not_legal_outside_dense_hd128_plain_rows(
    head_dim: int, num_kv: int, is_causal: bool, plain_row_body: bool
) -> None:
    """Requests for the family on other classes normalize to another family,
    which is what removes it from those search surfaces."""
    resolved = _resolve(
        _ALT,
        head_dim=head_dim,
        num_kv=num_kv,
        is_causal=is_causal,
        plain_row_body=plain_row_body,
    )
    assert resolved.pipeline_family != "fa4_alt"
    assert not resolved.alternating_warpgroups
    assert not fa4_alt_supported(
        head_dim=head_dim,
        num_kv=num_kv,
        is_causal=is_causal,
        plain_row_body=plain_row_body,
        has_row_epilogue=False,
        kv_tile_n=128,
    )
    if head_dim == 64 or is_causal:
        assert "fa4_alt" not in _search_families(
            _fragments(head_dim, dtype=torch.float16, is_causal=is_causal)
        )


def test_direct_epilogue_requests_are_not_the_family() -> None:
    """The TMA store through the dead Q stage is the family's only output
    path; a request without the TMA epilogue resolves to the fa4 body."""
    resolved = _resolve({**_ALT, "cute_flash_epi_tma": False})
    assert resolved.pipeline_family == "fa4"
    assert not resolved.alternating_warpgroups
    assert not resolved.epi_tma


def test_resolver_pins_the_structure() -> None:
    cfg = _resolve(_ALT)
    assert cfg.pipeline_family == "fa4_alt"
    assert cfg.alternating_warpgroups
    assert cfg.topology == "fa4"
    assert cfg.q_tile_count == 1
    assert cfg.persistent and cfg.persistent_loop == "while"
    assert cfg.epi_tma and not cfg.epi_stg
    assert cfg.softmax_disc and cfg.split_p_arrive and not cfg.p_chunk_arrive
    assert cfg.disc_pipe_depth == 1
    assert (cfg.p_store_repetition, cfg.s_load_repetition) == (16, 32)
    assert cfg.first_load_order == 0 and not cfg.precompute_qk_desc
    assert (cfg.softmax_regs, cfg.corr_regs, cfg.other_regs) == (192, 64, 64)
    assert cfg.kv_stage == 3
    assert cfg.e2e_offset0 == cfg.e2e_offset == 11
    assert cfg.exp2_packet == "8x2" and cfg.e2e_schedule == "16/6"
    assert cfg.rescale_threshold == 12.0 and cfg.corr_tile_size == 8
    assert cute_flash.flash_effective_config_values(cfg)[FAMILY_KEY] == "fa4_alt"


@pytest.mark.parametrize(("requested", "expected"), ((2, 2), (3, 3), (5, 3), (1, 2)))
def test_k_ring_depth_is_two_or_three(requested: int, expected: int) -> None:
    cfg = _resolve({**_ALT, "cute_flash_kv_stage": requested})
    assert cfg.kv_stage == expected
    assert FA4_ALT_V_STAGE == 2


def test_fa4_structural_knobs_normalize_to_one_effective_config() -> None:
    """Knobs of the fa4 body that have no meaning here collapse, so the search
    does not measure the same program under several names."""
    base = cute_flash.flash_effective_config_values(_resolve(_ALT))
    for overrides in (
        {"cute_flash_disc_pipe": 3},
        {"cute_flash_first_load_order": 4},
        {"cute_flash_p_chunk_arrive": True},
        {"cute_flash_precompute_qk_desc": True},
        {"cute_flash_persistent_loop": "counted"},
        {"cute_flash_other_regs": 48, "cute_flash_softmax_regs": 200},
        {"cute_flash_e2e_offset0": 3},
    ):
        assert (
            cute_flash.flash_effective_config_values(_resolve({**_ALT, **overrides}))
            == base
        ), overrides


def _search_spec(**overrides: object) -> ConfigSpec:
    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    values: dict[str, object] = {
        "head_dim": 128,
        "num_kv": 16,
        "num_bh": 32,
        "dtype": torch.bfloat16,
        "block_size_targets": {0: 1, 1: 128, 2: 128},
        "is_causal": False,
        "standard_dense_output": True,
        "standard_causal_output": False,
    }
    values.update(overrides)
    spec.enable_cute_flash_search(**values)  # pyrefly: ignore[bad-argument-type]
    return spec


def test_search_identity_folds_the_second_emulation_offset() -> None:
    """The autotuner's identity is the normalized config, which restores the
    requested emulation offsets of a fa4-topology body after folding the
    effective values. The family has one emulation phase, so the second offset
    must fold into the first there (and only there): otherwise the population
    and the pattern search's neighbors name the same program several times."""
    generation = _search_spec().create_config_generation()

    def identity(config: dict[str, object]) -> helion.Config:
        with patch.dict(os.environ, {}, clear=True):
            return generation.canonicalize_flat(
                generation.flatten(helion.Config.from_dict(config))
            )[1]

    alt = identity(_ALT)
    assert alt.config[FAMILY_KEY] == "fa4_alt"
    assert alt.config["cute_flash_e2e_offset0"] == alt.config["cute_flash_e2e_offset"]
    assert identity({**_ALT, "cute_flash_e2e_offset0": 3}) == alt
    assert identity({**_ALT, "cute_flash_e2e_offset0": 0}) == alt
    assert identity({**_ALT, "cute_flash_e2e_offset": 3}) != alt
    # the two-tile fa4 body keeps both phases apart
    fa4 = identity(_WINNER)
    assert fa4.config[FAMILY_KEY] == "fa4"
    assert identity({**_WINNER, "cute_flash_e2e_offset0": 3}) != fa4


def _assert_partial_sum_consumption_guard(src: str) -> None:
    """The even warpgroup rewrites the partial-sum slot and arrives on named
    barrier 13 only after the odd warpgroup's ``l_consumed`` arrival for the
    previous item (made after its read of the slot). Without it, one KV pair per
    item lets the even warpgroup arrive twice before one odd sync: the 256-thread
    barrier completes without the reader and the row sums mix items."""
    assert "cute.arch.mbarrier_init(flash_l_consumed_ptr, 128)" in src
    even_wait = src.index("mbar_spin_wait(flash_l_consumed_ptr, flash_lc_phase")
    slot_write = src.index("flash_l_t[flash_local_tidx] = flash_part_sum")
    even_arrive = src.index("named_barrier_arrive_unaligned(13, 256)")
    assert even_wait < slot_write < even_arrive
    odd_sync = src.index("named_barrier_wait_unaligned(13, 256)")
    slot_read = src.index("flash_l_t[flash_local_tidx] * flash_alpha")
    odd_arrive = src.index("mbarrier_arrive(flash_l_consumed_ptr)")
    assert odd_sync < slot_read < odd_arrive
    assert src.count("mbarrier_arrive(flash_l_consumed_ptr)") == 1
    # parity 1 passes the fresh barrier for the first item, so no branch is needed
    assert "flash_lc_phase = cutlass.Int32(1)" in src


def _assert_row_sum_consumption_guard(src: str) -> None:
    """The odd warpgroup's alpha slot carries the item's row sum to the
    correction warps. Its next item's first alpha is written after a P-buffer
    wait the row-sum publish already passed, so without a guard it could
    overwrite the slot before the correction warps read it and arrive on the
    64-thread handoff barrier twice before one correction sync. The warpgroup
    waits for their ``rowsum_consumed`` arrival at the top of every item."""
    assert "cute.arch.mbarrier_init(flash_rowsum_consumed_ptr, 128)" in src
    assert "flash_rsc_phase = cutlass.Int32(1)" in src
    odd_wait = src.index("mbar_spin_wait(flash_rowsum_consumed_ptr, flash_rsc_phase")
    odd_first_alpha = src.index("flash_pfor_ptr + 1, flash_pfor2_ptr + 1")
    assert odd_wait < odd_first_alpha
    assert src.count("mbar_spin_wait(flash_rowsum_consumed_ptr") == 1
    # the correction warps arrive after the row-sum read, before the epilogue
    corr_sync = src.rindex("named_barrier_wait_unaligned(7 + warp_idx % 4, 64)")
    slot_read = src.index("rcp_approx_ftz(flash_scale_t[1, flash_local_tidx])")
    corr_arrive = src.index("mbarrier_arrive(flash_rowsum_consumed_ptr)")
    epilogue = src.index("fa4_correction_epilogue_to_smem_scoped(")
    assert corr_sync < slot_read < corr_arrive < epilogue
    assert src.count("mbarrier_arrive(flash_rowsum_consumed_ptr)") == 1


# --------------------------------------------------------------------------
# emitted structure
# --------------------------------------------------------------------------


def test_emitted_body_has_the_alternating_structure() -> None:
    src = _emit(_ALT)
    assert "flash_fa4_alt_shared_storage(128, 3, 2, cutlass.BFloat16)" in src
    # the two MMA streams on their own warps
    qk_warp = src.index("if warp_idx == 12:")
    pv_warp = src.index("if warp_idx == 13:")
    assert "gemm_ptx_dyn_qk" in src[qk_warp:pv_warp]
    assert "gemm_ptx_precomputed_pv_ts" not in src[qk_warp:pv_warp]
    assert "gemm_ptx_precomputed_pv_ts" in src[pv_warp:]
    assert src.count("gemm_ptx_dyn_qk(") == 2
    assert src.count("gemm_ptx_precomputed_pv_ts(") == 2
    # the softmax warpgroups: one score read, then the buffer is released
    assert src.count("fa4_alt_rowmax_hold(") == 2
    assert src.count("fa4_alt_exp_pass_held(") == 2
    assert "flash_s_empty_ptr" in src and "flash_pv_done_ptr" in src
    # running max A -> B and B -> A over named barriers, the partial sum once per item
    assert "named_barrier_arrive_unaligned(11, 256)" in src
    assert "named_barrier_wait_unaligned(11, 256)" in src
    assert "named_barrier_arrive_unaligned(12, 256)" in src
    assert "named_barrier_wait_unaligned(12, 256)" in src
    assert "named_barrier_arrive_unaligned(13, 256)" in src
    assert "named_barrier_wait_unaligned(13, 256)" in src
    _assert_partial_sum_consumption_guard(src)
    _assert_row_sum_consumption_guard(src)
    # the knobs reach the pass
    assert "pair_batch=8, emu_batch=2" in src
    assert ", 16, 6, 11, " in src
    assert "-12.0" in src
    # the output goes through the dead Q stage and the TMA store warp
    assert "recast_ptr(sQ.iterator, _flash_osl.inner)" in src
    assert "cute.copy(_flash_tma_o, tOsO_tma[None, flash_q_stage]" in src
    assert (
        "fa4_correction_epilogue_to_smem_scoped(flash_pvt, tOtO, sO[None, None, flash_q_stage]"
        in src
    )
    assert "_flash_mLSE[" in src
    assert "_flash_mLSE[" not in _emit(_ALT, has_lse=False)


def test_emitted_body_with_default_knobs_and_the_shallow_k_ring() -> None:
    src = _emit(_ALT_DEFAULT)
    assert "flash_fa4_alt_shared_storage(128, 2, 2, cutlass.BFloat16)" in src
    assert "pair_batch=" not in src
    assert ", 8, 2, 0, " in src
    assert "-8.0" in src
    assert "rescale_o_tmem(tOtO, flash_a, flash_local_tidx, 128, 16)" in src


def test_emitted_body_adapts_to_the_kv_count() -> None:
    """The loader prefetches the next item's Q after the fourth K/V pair, or
    after the last pair of a shorter item."""
    long_src = _emit(_ALT, num_kv=16)
    assert "cutlass.range(0, 4, unroll=1)" in long_src
    assert "cutlass.range(4, 7, unroll=1)" in long_src
    short_src = _emit(_ALT, num_kv=6)
    assert "cutlass.range(0, 2, unroll=1)" in short_src
    assert "cutlass.range(2, 2, unroll=1)" not in short_src
    assert "tVgV[None, 4]" in short_src and "tVgV[None, 5]" in short_src
    # longer items prefetch the next item's first K tiles ahead of their last
    # two V tiles; a two-step item has nothing to overlap and issues its V
    # tiles first, then the next item's Q and K tiles
    assert long_src.index("tKgKn[None, 1]") < long_src.index("tVgV[None, 14]")
    two_src = _emit(_ALT, num_kv=2)
    assert "cutlass.range(0, 0, unroll=1)" not in two_src
    assert "flash_kv0" not in two_src
    assert two_src.index("tVgV[None, 1]") < two_src.index("tQgQn[None, flash_m_next]")
    assert two_src.index("tQgQn[None, flash_m_next]") < two_src.index("tKgKn[None, 0]")
    assert two_src.count("fa4_alt_exp_pass_held(") == 2
    _assert_partial_sum_consumption_guard(two_src)
    _assert_row_sum_consumption_guard(two_src)


def test_coverage_design_keeps_the_64_row_tile_witness() -> None:
    """The family makes six families on the hd128 dense surface, which moved
    the rotation design's rows away from the one combination that carries the
    64-row query tile (flat, staged ws_overlap body: three parents). The
    declared parent context keeps that value covered."""
    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=128,
        num_kv=48,
        num_bh=64,
        dtype=torch.float16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=False,
        standard_dense_output=True,
        standard_causal_output=False,
    )
    generation = spec.create_config_generation(
        overrides={cute_flash.FLASH_EPI_TMA_KEY: False}
    )
    with patch.dict(os.environ, {}, clear=True):
        coverage = generation.flash_deterministic_population_configs()
    # the family's only output path is the TMA store: under the direct-epilogue
    # override it is unreachable, which the design reports but does not reject
    assert generation.flash_structural_coverage_uncovered_values() == [
        (FAMILY_KEY, "fa4_alt")
    ]
    witnesses = [
        config.config
        for config in coverage
        if config.config[cute_flash.FLASH_Q_TILE_M_KEY] == 64
    ]
    assert witnesses
    for witness in witnesses:
        assert witness[FAMILY_KEY] == "ws_overlap"
        assert witness[cute_flash.FLASH_PERSISTENT_KEY] is False
        assert witness[cute_flash.FLASH_EPI_STG_KEY] is True
    # the family's only output path is the TMA store: it is absent from a
    # design pinned to the direct epilogue, and the pin holds in every row
    assert all(
        config.config[FAMILY_KEY] != "fa4_alt"
        and config.config[cute_flash.FLASH_EPI_TMA_KEY] is False
        for config in coverage
    )


# --------------------------------------------------------------------------
# schedule model
# --------------------------------------------------------------------------


def _spec(**overrides: object) -> FlashScheduleSpec:
    values: dict[str, object] = {
        "head_dim": 128,
        "kv_depth": 3,
        "v_depth": 2,
        "query_slots_per_cta": 1,
        "alternating_warpgroups": True,
        "persistent": True,
        "kv_iterations": 16,
    }
    values.update(overrides)
    return FlashScheduleSpec(**values)  # pyrefly: ignore[bad-argument-type]


def test_schedule_models_the_alternating_pipeline() -> None:
    verified = verify_flash_schedule(build_fa4_alt_schedule(_spec()))
    schedule = verified.schedule
    assert schedule.shared_memory_bytes == 232448
    assert schedule.tmem_columns == 512
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    assert barriers["s_empty_a"].expected_arrivals == 4
    assert barriers["pfor_a"].expected_arrivals == 256
    assert barriers["pfor2_b"].expected_arrivals == 128
    assert barriers["pv_done_a"].expected_arrivals == 1
    assert barriers["k_reuse_r0"].expected_arrivals == 2
    assert barriers["max_a"].expected_arrivals == 128
    assert barriers["part_sum_consumed_r0"].expected_arrivals == 128
    guard = [edge for edge in schedule.edges if edge.barrier == "part_sum_consumed_r0"]
    assert [(e.source, e.target, e.iteration_delta) for e in guard] == [
        ("softmax_b", "softmax_a", 1)
    ]
    assert barriers["row_sum_consumed_r0"].expected_arrivals == 128
    row_guard = [
        edge for edge in schedule.edges if edge.barrier == "row_sum_consumed_r0"
    ]
    assert [(e.source, e.target, e.iteration_delta) for e in row_guard] == [
        ("correction_b", "stat_b", 1)
    ]
    cycles = {cycle.barrier: cycle for cycle in schedule.phase_cycles}
    assert cycles["s_full_a"].uses_per_work == 8
    assert cycles["k_ready_r0"].uses_per_work == 16
    assert cycles["alpha_a"].uses_per_work == 7
    assert cycles["alpha_b"].uses_per_work == 9
    assert cycles["part_sum_r0"].uses_per_work == 1
    assert cycles["part_sum_consumed_r0"].uses_per_work == 1
    assert cycles["row_sum_consumed_r0"].uses_per_work == 1
    regions = {region.name: region for region in schedule.memory_regions}
    assert regions["Q_r0_q0"].offset == regions["O_stage_r0_q0"].offset == 0
    assert regions["P_a"].offset == 384 and regions["P_b"].offset == 448
    shallow = verify_flash_schedule(build_fa4_alt_schedule(_spec(kv_depth=2)))
    assert shallow.schedule.shared_memory_bytes == 199680
    # a two-step item has one even step: no even alpha handoff at all
    two_step = verify_flash_schedule(
        build_fa4_alt_schedule(_spec(kv_iterations=2))
    ).schedule
    two_barriers = {barrier.name for barrier in two_step.barriers}
    assert "alpha_a" not in two_barriers and "alpha_b" in two_barriers
    assert "STAT_a" not in {region.name for region in two_step.memory_regions}
    two_cycles = {cycle.barrier: cycle for cycle in two_step.phase_cycles}
    assert two_cycles["alpha_b"].uses_per_work == 2
    assert two_cycles["s_full_a"].uses_per_work == 1
    assert two_cycles["k_ready_r0"].uses_per_work == 2


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"alternating_warpgroups": False}, "spec flag"),
        ({"kv_iterations": 15}, "even"),
        ({"v_depth": None}, "V ring"),
        ({"causal": True}, "dense"),
        ({"split_p_arrive": False}, "split"),
        ({"query_slots_per_cta": 2}, "one query slot"),
    ),
)
def test_schedule_rejects_unsupported_shapes(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(FlashScheduleError, match=match):
        build_fa4_alt_schedule(_spec(**overrides))


@pytest.mark.parametrize(
    "guard,match",
    [
        ("part_sum_consumed_r0", "PARTSUM_r0 needs a consumption guard"),
        ("row_sum_consumed_r0", "STAT_b needs a consumption guard"),
    ],
)
def test_schedule_verifier_requires_the_consumption_guards(
    guard: str, match: str
) -> None:
    """The slot handed over named barrier 13 and the odd warpgroup's alpha slot
    (which carries the row sum across the item boundary) need their reverse
    barrier edges; reachability through the max handoff or the PV chain is not
    enough because the reader publishes other values before it reads them."""
    schedule = build_fa4_alt_schedule(_spec())
    without_guard = dataclasses.replace(
        schedule,
        edges=tuple(edge for edge in schedule.edges if edge.barrier != guard),
        barriers=tuple(
            barrier for barrier in schedule.barriers if barrier.name != guard
        ),
        phase_cycles=tuple(
            cycle for cycle in schedule.phase_cycles if cycle.barrier != guard
        ),
    )
    with pytest.raises(FlashScheduleError, match=match):
        verify_flash_schedule(without_guard)
    # ... and a guard that not every reader thread arrives on is no guard
    thin_guard = dataclasses.replace(
        schedule,
        barriers=tuple(
            FlashBarrier(barrier.name, 1, barrier.scope)
            if barrier.name == guard
            else barrier
            for barrier in schedule.barriers
        ),
        edges=tuple(
            dataclasses.replace(edge, arrival_count=1)
            if edge.barrier == guard
            else edge
            for edge in schedule.edges
        ),
    )
    with pytest.raises(FlashScheduleError, match=match):
        verify_flash_schedule(thin_guard)


def test_schedule_verifier_rejects_a_corrupted_release_count() -> None:
    schedule = build_fa4_alt_schedule(_spec())
    corrupted = dataclasses.replace(
        schedule,
        barriers=tuple(
            FlashBarrier(barrier.name, 128, barrier.scope)
            if barrier.name == "s_empty_a"
            else barrier
            for barrier in schedule.barriers
        ),
    )
    with pytest.raises(FlashScheduleError, match="s_empty_a"):
        verify_flash_schedule(corrupted)


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


def _kernel(config: dict[str, object]) -> object:
    return helion.kernel(
        _attention_with_lse.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict(config),
    )


def _run(
    config: dict[str, object], *args: torch.Tensor, launches: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = cast("Any", _kernel(config))
    out, lse = kernel(*args)
    for _ in range(launches - 1):
        out, lse = kernel(*args)
    torch.cuda.synchronize()
    return out, lse


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    return out, lse


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("config", "shape", "dtype"),
    (
        (_ALT, (2, 16, 2048, 128), torch.bfloat16),
        (_ALT, (1, 2, 512, 128), torch.float16),
        (_ALT_DEFAULT, (1, 2, 1280, 128), torch.bfloat16),
        (_ALT_DEFAULT, (1, 2, 256, 128), torch.bfloat16),
        # 192 work items on a 148-CTA grid with three pairs each: the per-pair
        # barrier parities carry across items with an odd count
        (_ALT_DEFAULT, (2, 16, 768, 128), torch.bfloat16),
        # 256 one-pair items: the two-step loader order across items
        (_ALT_DEFAULT, (8, 16, 256, 128), torch.bfloat16),
    ),
    ids=(
        "winner_knobs_bf16",
        "four_kv_fp16",
        "default_knobs_ten_kv",
        "default_knobs_two_kv",
        "odd_pairs_across_items",
        "one_pair_across_items",
    ),
)
def test_family_matches_the_fa4_body_and_sdpa(
    config: dict[str, object], shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    """Same per-element arithmetic as the fa4 body: outputs agree to the output
    dtype's rounding and the LSE to fp32 rounding; both match SDPA."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(*shape, dtype=dtype, device=DEVICE) for _ in range(3))
    out_ref, lse_ref = _reference(q, k, v)
    out_fa4, lse_fa4 = _run({**config, FAMILY_KEY: "fa4"}, q, k, v)
    out_alt, lse_alt = _run(config, q, k, v, launches=4)
    ulp = 2.0**-7 if dtype is torch.bfloat16 else 2.0**-10
    assert (out_alt.float() - out_fa4.float()).abs().max().item() <= 2 * ulp
    assert (lse_alt - lse_fa4).abs().max().item() < 1e-5
    assert (out_alt.float() - out_ref.float()).abs().max().item() < 1e-2
    assert (lse_alt - lse_ref).abs().max().item() < 1e-3


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("config", "shape"),
    ((_ALT, (2, 16, 2048, 128)), (_ALT_DEFAULT, (2, 16, 768, 128))),
    ids=("winner_knobs", "default_knobs_odd_pairs"),
)
def test_output_only_body_matches_the_fa4_body_and_sdpa(
    config: dict[str, object], shape: tuple[int, int, int, int]
) -> None:
    """Without an LSE output the odd warpgroup skips the LSE store only."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
    )
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def run(cfg: dict[str, object]) -> torch.Tensor:
        kernel = helion.kernel(
            _attention_output_only.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict(cfg),
        )
        out = cast("Any", kernel)(q, k, v)
        torch.cuda.synchronize()
        return out

    out_alt = run(config)
    out_fa4 = run({**config, FAMILY_KEY: "fa4"})
    assert (out_alt.float() - out_fa4.float()).abs().max().item() <= 2 * 2.0**-7
    assert (out_alt.float() - out_ref.float()).abs().max().item() < 1e-2
    assert torch.equal(run(config), out_alt)


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_only_alternating_modules_import_the_family_runtime() -> None:
    """The family's runtime import is not part of the shared flash preamble
    (which is also the CuTe disk-cache key of every flash render): fa4 and
    hd64 modules render byte for byte as before the family existed."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 2, 512, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )

    def render(config: dict[str, object]) -> str:
        bound = cast("Any", _kernel(config)).bind((q, k, v))
        return bound.to_code(
            bound._normalized_config_copy(helion.Config.from_dict(config))
        )

    alt_src = render(_ALT)
    fa4_src = render({**_ALT, FAMILY_KEY: "fa4"})
    alt_import = (
        "import helion._compiler.cute._flash_alt_runtime as _helion_flash_alt_rt"
    )
    assert alt_import in alt_src
    assert "_helion_flash_alt_rt.flash_fa4_alt_shared_storage(" in alt_src
    assert "_helion_flash_alt_rt" not in fa4_src
    assert "import helion._compiler.cute._flash_runtime as _helion_flash_rt" in fa4_src


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_family_launches_are_deterministic_under_repetition() -> None:
    """Repeated launches cover the barrier parities carried across KV steps and
    work items (three or four items per CTA on this shape); the handoff
    protocol must neither hang nor change a value."""
    torch.manual_seed(1)
    q, k, v = (
        torch.randn(2, 16, 2048, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    kernel = cast("Any", _kernel(_ALT))
    first_out, first_lse = kernel(q, k, v)
    for _ in range(200):
        out, lse = kernel(q, k, v)
    torch.cuda.synchronize()
    assert torch.equal(out, first_out)
    assert torch.equal(lse, first_lse)
