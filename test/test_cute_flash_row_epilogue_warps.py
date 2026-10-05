"""Fused row epilogues on the FA4 flash topology.

At (2, 16, 2048, 128) bf16 the XSA kernel (``examples/xsa.py``) ran 2.2x slower
on the persistent FA4 families than on the flat one.  Three general changes
cover it:

- the loop-invariant reciprocal hoist now descends into ``while`` bodies, so a
  persistent role loop no longer keeps one IEEE divide (a slow-path CALL) per
  element (covered in ``test_cute_hoist_loop_invariant_recip.py``);
- ``cute_flash_row_epilogue_warps`` moves the row program from the correction
  warpgroup (64 registers, both Q tiles in sequence, the next tile's rescales
  queued behind it) to the softmax warpgroups (200 registers, both Q tiles in
  parallel); with a staged output the epilogue warp TMA-loads the aux tile
  into the free ``sO`` stage so the program reads aux rows from shared memory;
- the chunk loop evaluates element pairs with packed ``f32x2`` FMA-pipe ops.
"""

from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.backend import CuteBackend
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import cute_flash

_FA4_2CTA = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_persistent": True,
    "cute_flash_kv_stage": 3,
    "cute_flash_s_stage": 2,
    "cute_flash_softmax_disc": True,
    "cute_flash_epi_tma": True,
}
_FA4 = {**_FA4_2CTA, "cute_flash_pipeline_family": "fa4"}
_FA4_STG = {**_FA4_2CTA, "cute_flash_epi_tma": False, "cute_flash_epi_stg": True}
_FA4_DIRECT = {**_FA4, "cute_flash_epi_tma": False, "cute_flash_epi_stg": False}
_DEEP_FLAT = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "fa4_deep_1cta",
    "cute_flash_persistent": False,
    "cute_flash_kv_stage": 2,
    "cute_flash_s_stage": 2,
    "cute_flash_softmax_disc": True,
}
_WS = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "ws_overlap",
    "cute_flash_persistent": False,
    "cute_flash_kv_stage": 2,
    "cute_flash_s_stage": 2,
}
_CORRECTION = {"cute_flash_row_epilogue_warps": "correction"}


def _xsa_kernel() -> helion.Kernel:
    from examples.xsa import xsa_kernel

    return helion.kernel(
        xsa_kernel.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


def _attention_kernel() -> helion.Kernel:
    from examples.attention import attention

    return helion.kernel(
        attention.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


def _qkv(
    batch: int, heads: int, seq: int, head_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=DEVICE)
        for _ in range(3)
    )
    return q, k, v


def _code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    bound = kernel.bind(args)
    return bound.to_code(bound._normalized_config_copy(helion.Config(**config)))


def _role(code: str, guard: str) -> str:
    """Source of one FA4 role block (from its ``if warp_idx`` guard to the next)."""
    start = code.index(f"    if {guard}:\n")
    end = code.find("\n    if ", start + 1)
    return code[start : end if end > 0 else len(code)]


# --------------------------------------------------------------------------
# Config resolution and search surface
# --------------------------------------------------------------------------


def test_row_epilogue_warps_resolution() -> None:
    fa4 = cute_flash.resolve_flash_config(
        128, 16, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    )
    assert fa4.row_epilogue_warps == "softmax"
    corr = cute_flash.resolve_flash_config(
        128,
        16,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY: "correction",
        },
    )
    assert corr.row_epilogue_warps == "correction"
    bogus = cute_flash.resolve_flash_config(
        128,
        16,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY: "epilogue",
        },
    )
    assert bogus.row_epilogue_warps == "correction"
    # The single consumer warpgroup of ws_overlap ignores the choice.
    ws = cute_flash.resolve_flash_config(
        64,
        2,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY: "softmax",
        },
    )
    assert ws.row_epilogue_warps == "correction"
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_ROW_EPILOGUE_WARPS": "correction"}):
        env = cute_flash.resolve_flash_config(
            128, 16, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
        )
    assert env.row_epilogue_warps == "correction"
    assert (
        cute_flash.flash_effective_config_values(fa4)[
            cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY
        ]
        == "softmax"
    )


def test_fused_row_epilogues_keep_the_chunked_softmax_body() -> None:
    # The whole-row FA4 body's two-slot statistics handoff can lap when a
    # fused row epilogue stalls a warpgroup at the end of a work item (the
    # kernel deadlocks intermittently), so a row program resolves to the
    # chunked body and the knob is not searched.
    config = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
        cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
    }
    assert cute_flash.resolve_flash_config(128, 16, config).softmax_disc is False
    assert (
        cute_flash.resolve_flash_config(
            128, 16, config, has_row_epilogue=True
        ).softmax_disc
        is True
    )
    common = {"num_bh": 32, "dtype": torch.bfloat16, "standard_dense_output": True}
    plain = cute_flash.flash_autotune_fragments(128, 16, **common)
    assert set(plain[cute_flash.FLASH_SOFTMAX_DISC_KEY].search_choices or ()) == {
        True,
        False,
    }
    fused = cute_flash.flash_autotune_fragments(
        128, 16, has_row_epilogue=True, plain_row_body=False, **common
    )
    assert tuple(fused[cute_flash.FLASH_SOFTMAX_DISC_KEY].search_choices or ()) == (
        True,
    )
    # The children that need the whole-row body go with it: the single_final
    # statistics transport and the partial-tile KV width of the hd64 class.
    small = {"num_bh": 4, "dtype": torch.bfloat16, "standard_dense_output": True}
    plain64 = cute_flash.flash_autotune_fragments(64, 2, **small)
    assert set(plain64[cute_flash.FLASH_STAT_TRANSPORT_KEY].search_choices or ()) == {
        "ring2",
        "single",
        "single_final",
    }
    assert set(plain64[cute_flash.FLASH_KV_TILE_N_KEY].search_choices or ()) == {
        128,
        160,
    }
    fused64 = cute_flash.flash_autotune_fragments(
        64, 2, has_row_epilogue=True, plain_row_body=False, **small
    )
    assert set(fused64[cute_flash.FLASH_STAT_TRANSPORT_KEY].search_choices or ()) == {
        "ring2",
        "single",
    }
    assert tuple(fused64[cute_flash.FLASH_KV_TILE_N_KEY].search_choices or ()) == (128,)
    # fp16 hd64 also searches the whole-row-only row-sum form on the plain body.
    half = {**small, "dtype": torch.float16}
    plain16 = cute_flash.flash_autotune_fragments(64, 2, **half)
    assert set(plain16[cute_flash.FLASH_SP_ROW_SUM_KEY].search_choices or ()) == {
        "fragment",
        "whole",
    }
    fused16 = cute_flash.flash_autotune_fragments(
        64, 2, has_row_epilogue=True, plain_row_body=False, **half
    )
    assert tuple(fused16[cute_flash.FLASH_SP_ROW_SUM_KEY].search_choices or ()) == (
        "fragment",
    )


def _fused_row_spec(
    head_dim: int, num_kv: int, dtype: torch.dtype, **flags: object
) -> ConfigSpec:
    """A flash search surface for a kernel with a fused row epilogue."""
    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=64,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=False,
        standard_dense_output=False,
        standard_causal_output=False,
        has_row_epilogue=True,
        plain_row_body=False,
        **flags,  # pyrefly: ignore [bad-argument-type]
    )
    return spec


def _active(spec: ConfigSpec, key: str) -> tuple[object, ...]:
    fragment = spec._flat_fields()[key]
    choices = (
        fragment.choices  # pyrefly: ignore [missing-attribute]
        if fragment.search_choices is None  # pyrefly: ignore [missing-attribute]
        else fragment.search_choices  # pyrefly: ignore [missing-attribute]
    )
    return tuple(choices)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "head_dim, num_kv", ((64, 2), (64, 4), (64, 8), (64, 16), (128, 16))
)
def test_fused_row_epilogue_surfaces_have_complete_structural_coverage(
    head_dim: int, num_kv: int, dtype: torch.dtype
) -> None:
    # The paired hd64 classes offer the whole-row-only children (the
    # ``single_final`` transport, the 160-column KV tile, fp16's whole row
    # sum) on the plain body; with a fused row epilogue the chunked body is
    # pinned, so none of them may be offered, otherwise the coverage design
    # has unreachable values and the autotuner refuses to start.
    spec = _fused_row_spec(head_dim, num_kv, dtype)
    generation = spec.create_config_generation()
    generation.validate_flash_structural_coverage()
    configs = generation.flash_deterministic_population_configs()
    assert configs
    assert all(
        generation.canonicalize_flat(generation.flatten(config))[1] == config
        for config in configs
    )
    assert _active(spec, cute_flash.FLASH_SOFTMAX_DISC_KEY) == (True,)
    assert "single_final" not in _active(spec, cute_flash.FLASH_STAT_TRANSPORT_KEY)
    assert _active(spec, cute_flash.FLASH_KV_TILE_N_KEY) == (128,)
    assert _active(spec, cute_flash.FLASH_SP_ROW_SUM_KEY) == ("fragment",)
    assert _active(spec, cute_flash.FLASH_Q_TILE_M_KEY) == (128,)


def test_target_seed_round_trips_under_the_fused_row_body() -> None:
    # The GB300 hd64 fp16 tuning policy carries whole-row-only values.  The
    # plain body seeds them verbatim; a fused row epilogue cannot take them,
    # so its seed must already equal its normalized form (the seed round-trip
    # check runs under the same flags as normalization).
    common = {
        "dtype": torch.float16,
        "num_bh": 8,
        "standard_dense_output": True,
        "target_device_capability": (10, 3),
    }
    plain = cute_flash.flash_attention_seed_config(64, 256, **common)
    assert plain is not None
    assert plain.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is False
    assert plain.config[cute_flash.FLASH_SP_ROW_SUM_KEY] == "whole"
    assert plain.config[cute_flash.FLASH_KV_TILE_N_KEY] == 160
    fused = cute_flash.flash_attention_seed_config(
        64, 256, has_row_epilogue=True, plain_row_body=False, **common
    )
    assert fused is not None
    assert fused.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is True
    assert fused.config[cute_flash.FLASH_SP_ROW_SUM_KEY] == "fragment"
    assert fused.config[cute_flash.FLASH_KV_TILE_N_KEY] == 128
    # The measured structure (family, schedule, transport) is kept.
    for key in (
        cute_flash.FLASH_PIPELINE_FAMILY_KEY,
        cute_flash.FLASH_E2E_SCHEDULE_KEY,
        cute_flash.FLASH_STAT_TRANSPORT_KEY,
        cute_flash.FLASH_KV_STAGE_KEY,
    ):
        assert fused.config[key] == plain.config[key]
    effective = cute_flash.flash_effective_config_values(
        cute_flash.resolve_flash_config(
            64,
            256,
            fused.config,
            dtype=torch.float16,
            num_bh=8,
            standard_dense_output=True,
            has_row_epilogue=True,
            plain_row_body=False,
        )
    )
    assert all(
        effective[key] == value
        for key, value in fused.config.items()
        if key in effective
    )
    # The fused seeds of the search population are canonical as well.
    seeds = cute_flash.flash_attention_seed_configs(
        64, 256, has_row_epilogue=True, plain_row_body=False, **common
    )
    assert fused in seeds


@onlyBackends(["cute"])
@pytest.mark.parametrize("shape", ((1, 4, 256, 64), (2, 16, 2048, 128)))
def test_xsa_search_surface_has_complete_structural_coverage(
    shape: tuple[int, int, int, int],
) -> None:
    # Every value the fused-epilogue surface offers must be reachable by the
    # autotuner's structural coverage design (the cold autotune refuses to
    # start otherwise).
    q, k, v = _qkv(*shape, torch.bfloat16)
    spec = _xsa_kernel().bind((q, k, v)).config_spec
    assert spec._cute_flash_has_row_epilogue and not spec._cute_flash_plain_row_body
    generation = spec.create_config_generation()
    generation.validate_flash_structural_coverage()
    # The autotuner's first act is to build this population.
    assert generation.flash_deterministic_population_configs()


@onlyBackends(["cute"])
def test_xsa_normalizes_the_whole_row_body_away() -> None:
    q, k, v = _qkv(2, 16, 2048, 128, torch.bfloat16)
    bound = _xsa_kernel().bind((q, k, v))
    normalized = bound._normalized_config_copy(
        helion.Config(**{**_FA4_2CTA, cute_flash.FLASH_SOFTMAX_DISC_KEY: False})
    )
    assert normalized.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is True
    plain = _attention_kernel().bind((q, k, v))
    normalized = plain._normalized_config_copy(
        helion.Config(**{**_FA4_2CTA, cute_flash.FLASH_SOFTMAX_DISC_KEY: False})
    )
    assert normalized.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is False


@onlyBackends(["cute"])
def test_row_sum_publish_waits_for_the_last_alpha_slot() -> None:
    # The two-slot statistics ring lets the row-sum publish follow the last
    # alpha before the correction warp consumed it; two bar.arrive from one
    # softmax warp then complete the named barrier alone and the correction
    # warp's bar.sync never returns. The row-sum acquire therefore also waits
    # for the last alpha's slot (parity ``phase ^ index``), on both routes.
    q, k, v = _qkv(1, 4, 512, 128, torch.bfloat16)
    for kernel, config in (
        (_attention_kernel(), _FA4_2CTA),
        (_xsa_kernel(), _FA4_2CTA),
        (_xsa_kernel(), {**_FA4_2CTA, **_CORRECTION}),
    ):
        code = _code(kernel, (q, k, v), **config)
        for stage in ("0", "1"):
            guard = (
                "warp_idx < 4" if stage == "0" else "(warp_idx >= 4) & (warp_idx < 8)"
            )
            softmax = _role(code, guard)
            acquire = (
                f"_helion_flash_rt.mbar_spin_wait(flash_s{stage}_corr_empty_ptr + "
                "(flash_s_corr_prod_index ^ 1), "
                "flash_s_corr_prod_phase ^ flash_s_corr_prod_index"
            )
            assert softmax.count(acquire) == 1
            publish = softmax.index("flash_local_tidx] = flash_row_sum")
            assert softmax.index(acquire) < publish
    # The acknowledged single-slot transport (hd64 classes) cannot run ahead
    # and keeps its own handoff.
    q, k, v = _qkv(1, 4, 512, 64, torch.bfloat16)
    single = _code(
        _attention_kernel(),
        (q, k, v),
        **{**_FA4, "cute_flash_stat_transport": "single"},
    )
    assert "flash_s_corr_prod_phase ^ flash_s_corr_prod_index" not in single
    assert "flash_s_corr_prod_index ^ 1" not in single


def test_row_epilogue_warps_is_searched_only_with_a_row_program() -> None:
    common = {"num_bh": 32, "dtype": torch.bfloat16, "standard_dense_output": True}
    plain = cute_flash.flash_autotune_fragments(128, 16, **common)
    knob = plain[cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY]
    assert len(tuple(knob.search_choices or knob.choices)) == 1
    fused = cute_flash.flash_autotune_fragments(
        128, 16, has_row_epilogue=True, **common
    )
    knob = fused[cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY]
    assert set(knob.search_choices or ()) == {"correction", "softmax"}
    # ws-only classes keep a single value even with a program.
    ws = cute_flash.flash_autotune_fragments(
        64, 3, num_bh=4, dtype=torch.bfloat16, has_row_epilogue=True
    )
    knob = ws[cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY]
    assert tuple(knob.search_choices or knob.choices) == ("correction",)
    # Seeds: the fa4 family seeds carry the softmax default and the correction
    # evaluation is seeded once; without a program neither variant appears.
    seeds = cute_flash.flash_attention_seed_configs(
        128, 16, has_row_epilogue=True, **common
    )
    values = {
        seed.config.get(cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY)
        for seed in seeds
        if seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY, "").startswith("fa4")
    }
    assert values == {"correction", "softmax"}
    # (The ws_overlap family seed carries ``correction``: ws pins it.)
    plain_fa4_seeds = [
        seed
        for seed in cute_flash.flash_attention_seed_configs(128, 16, **common)
        if str(seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY, "")).startswith(
            "fa4"
        )
    ]
    assert plain_fa4_seeds
    assert all(
        seed.config.get(cute_flash.FLASH_ROW_EPILOGUE_WARPS_KEY) != "correction"
        for seed in plain_fa4_seeds
    )


# --------------------------------------------------------------------------
# Codegen structure
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
def test_softmax_route_stages_aux_through_smem_on_the_2cta_persistent_body() -> None:
    q, k, v = _qkv(1, 2, 512, 128, torch.bfloat16)
    code = _code(_xsa_kernel(), (q, k, v), **_FA4_2CTA)
    softmax0 = _role(code, "warp_idx < 4")
    softmax1 = _role(code, "(warp_idx >= 4) & (warp_idx < 8)")
    correction = _role(code, "(warp_idx >= 8) & (warp_idx < 12)")
    epilogue_warp = _role(code, "warp_idx == 13")
    # The program runs on the softmax warpgroups, one Q tile each.
    assert "_ep0_t2r, _ep0_r2s" in softmax0 and "_ep1_t2r, _ep1_r2s" in softmax1
    assert "_ep0_" not in correction and "_ep1_" not in correction
    # The correction warps still order their next-tile ``pfor`` pre-arrive
    # behind the last PV (the softmax warpgroup's final P store must have
    # completed the current ``pfor`` phase first) but read nothing.
    assert correction.count("mbar_spin_wait(flash_o_full_ptr") == 2
    assert "mbar_spin_wait(flash_corr_epi_empty_ptr" not in correction
    for stage, block in (("0", softmax0), ("1", softmax1)):
        assert (
            f"flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz(flash_row_sum)"
            in block
        )
        assert (
            f"mbar_spin_wait(flash_o_full_ptr + {stage}, flash_sm_o_full_phase" in block
        )
        assert (
            f"mbar_spin_wait(flash_corr_epi_empty_ptr + {stage}, flash_sm_epi_empty_phase"
            in block
        )
        assert (
            f"mbar_spin_wait(flash_aux_full_ptr + {stage}, flash_sm_aux_full_phase"
            in block
        )
        assert f"cute.arch.mbarrier_arrive(flash_corr_epi_full_ptr + {stage})" in block
        # The TMEM reads are complete before the next tile's P store.
        assert block.index("cute.arch.fence_view_async_tmem_load()") < block.index(
            f"cute.arch.mbarrier_arrive(flash_corr_epi_full_ptr + {stage})"
        )
        # Aux rows come from this thread's own sO chunks, not from gmem.
        assert (
            f"cute.autovec_copy(_ep{stage}_tOsO[None, 0, 0, 0], _ep{stage}_a0_0)"
            in block
        )
        assert f"_ep_gaux0{stage} = cute.flat_divide" not in block
        # Every pass, including the aux-only norm, runs after the aux wait.
        assert block.index("mbar_spin_wait(flash_aux_full_ptr") < block.index(
            f"_ep{stage}_v7 = cute.math.max"
        )
    # The epilogue warp TMA-loads the aux tile into both sO stages at the top of
    # each work item, the first one's after the first QK. That pacing wait is
    # bounded: a plain parity-0 wait on ``s_full[0]`` would alias after two QK
    # completions and never return for a work item with two KV steps.
    assert "cute_cpasync_flash.prefetch_descriptor(_flash_tma_aux0)" in epilogue_warp
    assert "mbar_spin_wait(flash_s_full_ptr" not in epilogue_warp
    assert epilogue_warp.index(
        "mbar_spin_wait_bounded(flash_s_full_ptr + 0, cutlass.Int32(0), 256, 1000)"
    ) < epilogue_warp.index("while flash_tile_id")
    assert (
        epilogue_warp.count(
            "cute.arch.mbarrier_arrive_and_expect_tx(flash_aux_full_ptr"
        )
        == 2
    )
    assert (
        epilogue_warp.count(
            "cute.copy(_flash_tma_aux0, tAgA_aux_tma[None, flash_q_mma_tile"
        )
        == 2
    )
    assert epilogue_warp.index("cute.copy(_flash_tma_aux0") < epilogue_warp.index(
        "mbar_spin_wait(flash_corr_epi_full_ptr + 0"
    )
    assert "cute.arch.mbarrier_init(flash_aux_full_ptr + flash_st, 1)" in code
    assert "_flash_tma_aux0, _flash_mEpiAux0t" in code
    # The reciprocal hoist reaches into the persistent while loop and the chunk
    # loop walks element pairs with packed FMA-pipe ops.
    assert "_helion_inv_div_0 = 1.0 / _ep0_v7" in softmax0
    divides = [line for line in softmax0.splitlines() if "/ _ep0_v7" in line]
    assert divides and all(
        re.match(r"\s*_helion_inv_div_\d+ = 1\.0 / _ep0_v7$", line) for line in divides
    )
    # One reciprocal per stage: sibling chunk loops reuse the first hoist.
    assert len(divides) == 1
    assert "range_constexpr(0, 16, 2)" in softmax0
    assert "cute.arch.fma_packed_f32x2(" in softmax0
    # The softmax roles decode the tile coordinates for the aux rows.
    assert "flash_m_tile0 = " in softmax0


@onlyBackends(["cute"])
def test_correction_route_keeps_the_program_in_the_correction_warps() -> None:
    q, k, v = _qkv(1, 2, 512, 128, torch.bfloat16)
    code = _code(_xsa_kernel(), (q, k, v), **_FA4_2CTA, **_CORRECTION)
    correction = _role(code, "(warp_idx >= 8) & (warp_idx < 12)")
    softmax0 = _role(code, "warp_idx < 4")
    assert "_ep0_t2r, _ep0_r2s" in correction and "_ep1_t2r, _ep1_r2s" in correction
    assert "_ep0_" not in softmax0
    assert "flash_sm_o_full_phase" not in code
    assert (
        "_flash_tma_aux0" not in code and "flash_aux_full_ptr + 0, flash_sm" not in code
    )
    assert "_ep_gaux00 = cute.flat_divide(_flash_mEpiAux0" in correction
    # The hoist still applies inside the persistent loop.
    assert "_helion_inv_div_0 = 1.0 / _ep0_v7" in correction


@onlyBackends(["cute"])
def test_softmax_route_direct_store_reads_aux_from_gmem() -> None:
    q, k, v = _qkv(1, 2, 512, 128, torch.bfloat16)
    code = _code(_xsa_kernel(), (q, k, v), **_FA4_DIRECT)
    softmax0 = _role(code, "warp_idx < 4")
    assert "_flash_tma_aux0" not in code
    assert "flash_sm_o_full_phase = cutlass.Int32(0)" in softmax0
    assert "flash_sm_epi_empty_phase" not in code
    assert "tDgO0 = flash_thr_o_ld0.partition_D(gO_epi0)" in softmax0
    assert "cute.autovec_copy(_ep0_out0, tDgO0[None, 0, 0])" in softmax0
    assert "_ep_gaux00 = cute.flat_divide(_flash_mEpiAux0" in softmax0
    # The aux-only norm pass is hoisted ahead of the KV loop on this route.
    assert softmax0.index("_ep0_v7 = cute.math.max") < softmax0.index(
        "flash_row_max = cutlass.Float32(-cutlass.Float32.inf)"
    )


@onlyBackends(["cute"])
def test_staged_stg_route_and_flat_grid_also_stage_aux() -> None:
    q, k, v = _qkv(1, 2, 512, 128, torch.bfloat16)
    stg = _code(_xsa_kernel(), (q, k, v), **_FA4_STG)
    assert "cute.copy(_flash_tma_aux0" in stg and "fa4_store_o_smem_to_gmem" in stg
    flat = _code(_xsa_kernel(), (q, k, v), **{**_FA4, "cute_flash_persistent": False})
    assert "while flash_tile_id" not in flat
    softmax0 = _role(flat, "warp_idx < 4")
    assert "cute.autovec_copy(_ep0_tOsO[None, 0, 0, 0], _ep0_a0_0)" in softmax0
    epilogue_warp = _role(flat, "warp_idx == 13")
    assert epilogue_warp.count("cute.copy(_flash_tma_aux0") == 2


@onlyBackends(["cute"])
def test_plain_attention_codegen_ignores_the_knob() -> None:
    q, k, v = _qkv(1, 2, 512, 128, torch.bfloat16)
    default = _code(_attention_kernel(), (q, k, v), **_FA4_2CTA)
    corr = _code(_attention_kernel(), (q, k, v), **_FA4_2CTA, **_CORRECTION)
    assert default == corr
    assert "flash_sm_o_full_phase" not in default and "_flash_tma_aux0" not in default


@onlyBackends(["cute"])
def test_ws_staged_route_stages_aux_and_ignores_the_knob() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_xsa_kernel(), (q, k, v), **_WS)
    assert code == _code(_xsa_kernel(), (q, k, v), **_WS, **_CORRECTION)
    assert "flash_sm_o_full_phase" not in code
    # Warp 0 issues the aux TMA with the prologue loads; the consumer waits
    # ``aux_full`` before the aux-only pass and reads aux rows from sO.
    assert "cute.arch.mbarrier_init(flash_aux_full_ptr, 1)" in code
    assert (
        code.count(
            "cute.copy(_flash_tma_aux0, tAgA_aux_tma[None, flash_m_tile, 0, flash_bh]"
        )
        == 1
    )
    assert code.index("cute.copy(_flash_tma_aux0") < code.index(
        "flash_tmem.allocate(512)"
    )
    assert "mbar_spin_wait(flash_aux_full_ptr, flash_aux_full_phase" in code
    assert "cute.autovec_copy(_ep_tOsO[None, 0, 0, 0], _ep_a0_0)" in code
    assert "_ep_gaux0 = cute.flat_divide(_flash_mEpiAux0" not in code
    assert "range_constexpr(0, 64, 2)" in code
    # The direct-store ws route keeps its gmem aux loads.
    direct = _code(_xsa_kernel(), (q, k, v), **_WS, cute_flash_epi_stg=False)
    assert "_flash_tma_aux0" not in direct
    assert "_ep_gaux0 = cute.flat_divide(_flash_mEpiAux0" in direct
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_ROW_EPILOGUE_PACKED": "0"}):
        scalar = _code(_xsa_kernel(), (q, k, v), **_WS)
    # The scalar form walks the same element pairs with scalar fma.
    assert "range_constexpr(0, 64, 2)" in scalar
    assert "cute.math.fma(" in scalar
    assert "cute.arch.fma_packed_f32x2(" not in scalar
    assert "cute.arch.mul_packed_f32x2(" not in scalar


# --------------------------------------------------------------------------
# GPU numerics
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
@pytest.mark.parametrize("seq", (256, 1024, 2048))
@pytest.mark.parametrize(
    "config",
    (
        pytest.param(_FA4_2CTA, id="fa4_2cta-persistent-epi_tma"),
        pytest.param(_FA4, id="fa4-persistent-epi_tma"),
        pytest.param(_DEEP_FLAT, id="fa4_deep_1cta-flat"),
        pytest.param({**_WS, "cute_flash_persistent": True}, id="ws-persistent"),
    ),
)
def test_xsa_matches_reference_on_every_family(
    config: dict[str, object], seq: int
) -> None:
    from examples.xsa import ref_xsa

    if config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4_2cta" and seq < 512:
        pytest.skip("fa4_2cta needs at least four KV tiles")
    q, k, v = _qkv(1, 4, seq, 128, torch.bfloat16)
    code, out = code_and_output(_xsa_kernel(), (q, k, v), **config)
    if config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] != "ws_overlap":
        assert "flash_sm_o_full_phase" in code
    torch.testing.assert_close(out, ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "config",
    (
        pytest.param({**_FA4_2CTA, **_CORRECTION}, id="correction-route"),
        pytest.param(_FA4_STG, id="softmax-epi_stg"),
        pytest.param(_FA4_DIRECT, id="softmax-direct"),
        pytest.param({**_FA4_2CTA, "cute_flash_corr_tile_size": 32}, id="softmax-c32"),
    ),
)
def test_xsa_routes_match_reference(config: dict[str, object]) -> None:
    from examples.xsa import ref_xsa

    q, k, v = _qkv(1, 4, 1024, 128, torch.bfloat16)
    _, out = code_and_output(_xsa_kernel(), (q, k, v), **config)
    torch.testing.assert_close(out, ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_xsa_head_dim_64_and_fp16_stage_aux() -> None:
    from examples.xsa import ref_xsa

    for dtype in (torch.bfloat16, torch.float16):
        q, k, v = _qkv(1, 4, 512, 64, dtype)
        code, out = code_and_output(_xsa_kernel(), (q, k, v), **_FA4)
        assert "cute.copy(_flash_tma_aux0" in code
        torch.testing.assert_close(out, ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_xsa_two_inputs_do_not_alias_the_staged_aux() -> None:
    """The aux tile is TMA-loaded per work item; the same compiled kernel run
    on a second input set must not see the first one's rows."""
    from examples.xsa import ref_xsa

    q, k, v = _qkv(1, 4, 2048, 128, torch.bfloat16)
    bound = _xsa_kernel().bind((q, k, v))
    fn = bound.compile_config(bound._normalized_config_copy(helion.Config(**_FA4_2CTA)))
    torch.testing.assert_close(fn(q, k, v), ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)
    torch.manual_seed(1)
    q2, k2, v2 = (torch.randn_like(q) for _ in range(3))
    torch.testing.assert_close(
        fn(q2, k2, v2), ref_xsa(q2, k2, v2), atol=2e-2, rtol=2e-2
    )


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "config, shape",
    (
        pytest.param(_FA4_2CTA, (4, 16, 2048, 128), id="fa4_2cta-persistent-epi_tma"),
        pytest.param(_FA4, (4, 16, 2048, 128), id="fa4-persistent-epi_tma"),
        pytest.param(
            {**_FA4_2CTA, **_CORRECTION},
            (4, 16, 2048, 128),
            id="fa4_2cta-correction-route",
        ),
        pytest.param(
            {**_WS, "cute_flash_persistent": True, "cute_flash_epi_stg": True},
            (4, 32, 1024, 64),
            id="ws-persistent-epi_stg",
        ),
    ),
)
def test_persistent_ctas_run_several_xsa_work_items(
    config: dict[str, object], shape: tuple[int, int, int, int]
) -> None:
    """Every persistent CTA (or 2-CTA cluster) loops over several work items.

    With 148 SMs, (1, 4, 2048, 128) is one fa4 work item per CTA, so the sO
    stages, the staged aux tile and the carried barrier phases are never
    reused inside a CTA.  These shapes give every CTA three to seven work
    items (both phase parities), and fresh inputs go through the same
    compiled kernel; the ``pfor`` pre-arrive race only showed at such shapes.
    """
    from examples.xsa import ref_xsa

    batch, heads, seq, head_dim = shape
    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    # An fa4 work item spans two 128-row Q tiles (a 2-CTA cluster shares one).
    assert batch * heads * seq // 256 >= 3 * sm_count
    q, k, v = _qkv(batch, heads, seq, head_dim, torch.bfloat16)
    bound = _xsa_kernel().bind((q, k, v))
    fn = bound.compile_config(bound._normalized_config_copy(helion.Config(**config)))
    for seed in range(3):
        torch.manual_seed(seed)
        q, k, v = (torch.randn_like(q) for _ in range(3))
        torch.testing.assert_close(fn(q, k, v), ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_packed_and_scalar_row_programs_agree() -> None:
    from examples.xsa import ref_xsa

    q, k, v = _qkv(1, 4, 512, 128, torch.bfloat16)
    _, packed = code_and_output(_xsa_kernel(), (q, k, v), **_FA4)
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_ROW_EPILOGUE_PACKED": "0"}):
        code, scalar = code_and_output(_xsa_kernel(), (q, k, v), **_FA4)
    assert "cute.arch.fma_packed_f32x2(" not in code
    assert "cute.arch.mul_packed_f32x2(" not in code
    assert "cute.math.fma(" in code
    torch.testing.assert_close(packed, ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)
    # Same element pairs, same accumulators and the same fused multiply-adds
    # (scalar fma for packed fma): the two instruction forms round identically.
    assert torch.equal(packed, scalar)
