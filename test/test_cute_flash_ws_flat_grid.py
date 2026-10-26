"""ws_overlap flat-grid prologue and staged-epilogue trims.

At tiny problem sizes (e.g. ``[1, 4, 256, 64]``: 8 CTAs, 2 KV tiles) the fused
flash kernel is pure fixed latency. Three general trims of the two-warpgroup
``ws_overlap`` body are covered here:

- the prologue creates every pipeline with ``defer_sync`` and, on the flat
  one-tile grid, warp 0 issues the Q and prologue K/V TMA loads before the
  TMEM-allocation barrier (the loads overlap the rest of the prologue);
- the O epilogue stages the tile through a swizzled ``sO`` and drains it with
  row-contiguous 16-byte vector stores (``cute_flash_epi_stg``, now legal for
  ``ws_overlap``) instead of one 128-byte row per lane;
- the flat grid frees TMEM under the output drain;
- the otherwise idle warp 1 owns the TMEM allocation (allocated, with the
  permit handed back, while warp 0 sets up the mbarriers and issues the first
  loads) and frees it after a named teardown barrier the consumer warpgroup
  joins right after its last TMEM read;
- ``cute_flash_ws_one_pass``: a 64-row tile over exactly two KV tiles reads
  both score tiles as one 256-wide row (one max, one exp2 sweep, one P
  handoff, both PV MMAs accumulate without a rescale) instead of two
  online-softmax rounds with their handshakes.
"""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import cute_flash

_WS = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "ws_overlap",
    "cute_flash_s_stage": 2,
    "cute_flash_kv_stage": 2,
}
_FLAT = {**_WS, "cute_flash_persistent": False}
_PERSISTENT = {**_WS, "cute_flash_persistent": True}
# 64-row query tiles (tcgen05 M=64): twice the CTAs, two rows per consumer
# thread shared across a quad.
_M64 = {**_FLAT, "cute_flash_q_tile_m": 64}
# One-pass softmax over the two KV tiles of a 256-long sequence.
_M64_ONE_PASS = {**_M64, "cute_flash_ws_one_pass": True}


def _attention_kernel() -> helion.Kernel:
    from examples.attention import attention

    return helion.kernel(
        attention.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


def _relu_kernel() -> helion.Kernel:
    from examples.attention import attention_relu_output

    return helion.kernel(
        attention_relu_output.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


def _xsa_kernel() -> helion.Kernel:
    from examples.xsa import xsa_kernel

    return helion.kernel(
        xsa_kernel.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _causal_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
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
    return out.view(q_in.size())


def _code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    bound = kernel.bind(args)
    return bound.to_code(bound._normalized_config_copy(helion.Config(**config)))


def _qkv(
    batch: int, heads: int, seq: int, head_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=DEVICE)
        for _ in range(3)
    )
    return q, k, v


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    if causal:
        mask = torch.ones(q.size(-2), k.size(-2), device=q.device, dtype=torch.bool)
        scores = scores.masked_fill(~torch.tril(mask), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    return out, lse


# --------------------------------------------------------------------------
# Config resolution and search surface
# --------------------------------------------------------------------------


def test_staged_epilogue_is_the_ws_default_and_legal_only_for_two_warpgroups() -> None:
    ws = cute_flash.resolve_flash_config(
        64, 2, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    )
    assert ws.topology == "ws_overlap" and ws.s_stage == 2
    assert ws.epi_stg is True
    assert ws.epi_stg_store == "whole" and ws.epi_stg_gmem == "stage"
    # The staged path's O chunk defaults to a whole 64-column load.
    assert ws.corr_tile_size == 64
    direct = cute_flash.resolve_flash_config(
        64,
        2,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_EPI_STG_KEY: False,
        },
    )
    assert direct.epi_stg is False
    single = cute_flash.resolve_flash_config(
        64,
        2,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_S_STAGE_KEY: 1,
            cute_flash.FLASH_EPI_STG_KEY: True,
        },
    )
    assert single.s_stage == 1 and single.epi_stg is False
    # fa4 keeps its own defaults.
    fa4 = cute_flash.resolve_flash_config(
        64, 2, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    )
    assert fa4.epi_stg is False and fa4.corr_tile_size == 8
    # ws chunk widths: multiples that divide head_dim from {16, 32, 64}; fa4-only
    # values fall back to the default.
    for requested, expected in ((16, 16), (32, 32), (64, 64), (8, 64)):
        cfg = cute_flash.resolve_flash_config(
            64,
            2,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_CORR_TILE_SIZE_KEY: requested,
            },
        )
        assert cfg.corr_tile_size == expected, requested
    hd128 = cute_flash.resolve_flash_config(
        128, 2, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    )
    assert hd128.epi_stg is True and hd128.corr_tile_size == 64


def test_ws_search_surface_measures_the_staged_epilogue() -> None:
    fragments = cute_flash.flash_autotune_fragments(
        64,
        2,
        num_bh=4,
        dtype=torch.bfloat16,
        standard_dense_output=True,
        pipeline_family_override="ws_overlap",
    )
    epi_stg = fragments[cute_flash.FLASH_EPI_STG_KEY]
    assert epi_stg.default() is True
    assert set(epi_stg.search_choices or ()) == {False, True}
    # The epilogue chunk width and drain shape stay fixed at their measured
    # defaults so the small ws-only spaces remain exactly enumerable.
    corr_tile = fragments[cute_flash.FLASH_CORR_TILE_SIZE_KEY]
    assert corr_tile.default() == 64
    assert tuple(corr_tile.search_choices or ()) == (64,)
    store = fragments[cute_flash.FLASH_EPI_STG_STORE_KEY]
    assert store.default() == "whole"
    assert tuple(store.search_choices or ()) == ("whole",)
    # The flat ws_overlap grid is seeded next to the persistent family seed.
    seeds = cute_flash.flash_attention_seed_configs(
        64, 2, num_bh=4, dtype=torch.bfloat16, standard_dense_output=True
    )
    flat_ws = [
        seed
        for seed in seeds
        if seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == "ws_overlap"
        and seed.config.get(cute_flash.FLASH_PERSISTENT_KEY) is False
    ]
    assert flat_ws


# --------------------------------------------------------------------------
# Codegen structure
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
def test_flat_grid_issues_loads_before_the_tmem_barrier_and_stages_o() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_attention_kernel(), (q, k, v), **_FLAT)
    assert code.count("defer_sync=True).make_participants()") == 6
    assert code.count("cute.arch.mbarrier_init_fence()") == 3
    # Warp 0 issues K tile 0 and Q as soon as the Q/K pipelines exist, then
    # creates the V pipeline and issues the V tiles, all before the TMEM
    # barrier; warp 1 then allocates TMEM (and hands the permit back) ahead of
    # the handshake pipelines, so the tcgen05.alloc never runs next to the
    # mbarrier inits.
    q_issue = code.index("cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile]")
    k_issue = code.index("cute.copy(_flash_tma_k, tKgK[None, 0]")
    k1_issue = code.index("cute.copy(_flash_tma_k, tKgK[None, 1]")
    v_pipeline = code.index("tx_count=flash_v_bytes")
    v_issue = code.index("cute.copy(_flash_tma_v, tVgV[None, 0]")
    allocate = code.index("flash_tmem.allocate(512)")
    wait = code.index("flash_tmem.wait_for_alloc()")
    assert "barrier_for_retrieve=flash_tmem_bar, allocator_warp_id=1)" in code
    assert (
        "flash_tmem.allocate(512)\n    if warp_idx == 1:\n"
        "        flash_tmem.relinquish_alloc_permit()\n"
        "    flash_mma_s_prod, flash_mma_s_cons" in code
    )
    assert code.index("cute.arch.sync_warp()") < k_issue < q_issue < k1_issue
    assert k1_issue < v_pipeline < v_issue < allocate < wait
    assert code.count("cute.arch.sync_warp()") == 2
    assert code.count("relinquish_alloc_permit()") == 1
    assert code.index("prefetch_descriptor(_flash_tma_q)") < code.index(
        "flash_shared_storage("
    )
    # The producer role only waits for Q (no second issue).
    assert code.count("cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile]") == 1
    # Staged epilogue with the dedicated sO tile and the coalesced drain; the
    # TMEM free sits in the producer role, ahead of the consumer role.
    assert "o_stage=1, q_rows=128)" in code
    assert "sO = storage.sO.get_tensor(_flash_osl.outer" in code
    assert "fa4_correction_epilogue_to_smem_scoped(" in code
    assert ", 64, 64, cutlass.BFloat16)" in code
    assert "fa4_store_o_smem_to_gmem_whole(" in code
    # The consumer joins the named teardown barrier right after its last TMEM
    # read; warp 1 waits on it and frees TMEM while the stores drain. No CTA
    # barrier remains: warp 0 and the idle warps simply exit.
    assert (
        "cute.arch.fence_view_async_tmem_load()\n        flash_o_full.release()\n"
        "        cute.arch.barrier(barrier_id=3, number_of_threads=160)" in code
    )
    assert (
        "if warp_idx == 1:\n"
        "        cute.arch.barrier(barrier_id=3, number_of_threads=160)\n"
        "        flash_tmem.free(flash_tmem_ptr)" in code
    )
    assert code.count("cute.arch.barrier()") == 0
    consumer_start = code.index("flash_row_max = cutlass.Float32(-cutlass.Float32.inf)")
    assert code.index("flash_tmem.free(flash_tmem_ptr)") < consumer_start
    # The LSE store needs only the row statistics: it precedes the O epilogue.
    assert code.index("_flash_mLSE[") < code.index(
        "flash_o_full = flash_mma_o_cons.wait_and_advance()\n        flash_inv_sum"
    )
    assert "cute.autovec_copy(flash_rego" not in code
    sliced = _code(
        _attention_kernel(), (q, k, v), **_FLAT, cute_flash_epi_stg_store="slice"
    )
    assert "fa4_store_o_smem_to_gmem(" in sliced


@onlyBackends(["cute"])
def test_flat_grid_direct_store_keeps_the_late_teardown() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_attention_kernel(), (q, k, v), **_FLAT, cute_flash_epi_stg=False)
    assert "o_stage=0, q_rows=128)" in code
    assert "fa4_store_o_smem_to_gmem(" not in code
    assert "cute.autovec_copy(flash_rego" in code
    # Early loads still apply; the teardown (a CTA barrier, then the TMEM
    # owner's free) stays after both roles.
    assert code.index("cute.copy(_flash_tma_k, tKgK[None, 0]") < code.index(
        "flash_tmem.allocate(512)"
    )
    consumer_start = code.index("flash_row_max = cutlass.Float32(-cutlass.Float32.inf)")
    assert consumer_start < code.index(
        "cute.arch.barrier()\n    if warp_idx == 1:\n        flash_tmem.free(flash_tmem_ptr)"
    )
    assert "barrier_id=3" not in code


@onlyBackends(["cute"])
def test_persistent_grid_keeps_loads_in_the_producer_loop() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_attention_kernel(), (q, k, v), **_PERSISTENT)
    loop = code.index("while flash_tile_id < _flash_total_tiles")
    assert code.index("flash_tmem.allocate(512)") < loop
    assert loop < code.index("cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile]")
    assert "cute.arch.sync_warp()" not in code
    # Staged store with a warpgroup-scoped sync (the per-tile CTA barrier of
    # the persistent loop separates consecutive tiles' sO use).
    assert "fa4_store_o_smem_to_gmem_whole(" in code
    assert "cute.arch.barrier(barrier_id=2, number_of_threads=128)" in code
    consumer_start = code.index("flash_row_max = cutlass.Float32(-cutlass.Float32.inf)")
    assert consumer_start < code.index(
        "cute.arch.barrier()\n    if warp_idx == 1:\n        flash_tmem.free(flash_tmem_ptr)"
    )


@onlyBackends(["cute"])
def test_causal_flat_grid_guards_the_early_loads_with_the_active_range() -> None:
    q, k, v = _qkv(1, 2, 512, 64, torch.bfloat16)
    code = _code(_causal_attention, (q, k, v), **_FLAT)
    early = code.index("flash_tmem.allocate(512)")
    prelude = code[:early]
    assert (
        "flash_active_count = flash_m_tile - cutlass.Int32(0) + cutlass.Int32(1)"
        in prelude
    )
    assert "if cutlass.Int32(0) < flash_active_count:" in prelude
    assert (
        "cute.copy(_flash_tma_k, tKgK[None, cutlass.Int32(0) + cutlass.Int32(0)]"
        in prelude
    )
    # The range is computed once, by every warp, ahead of the K and the V
    # issue blocks that both guard their loads with it (the producer role
    # recomputes it for its own loop).
    assert prelude.count("flash_active_count = ") == 1
    assert prelude.index("flash_active_count = ") < prelude.index(
        "cute.arch.sync_warp()"
    )
    assert prelude.count("if cutlass.Int32(1) < flash_active_count:") == 2
    assert prelude.count("cute.arch.sync_warp()") == 2


@onlyBackends(["cute"])
def test_row_epilogue_stages_aux_rows_through_smem_ahead_of_the_kv_loop() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_xsa_kernel(), (q, k, v), **_FLAT)
    # Warp 0 TMA-loads the aux tile into sO with the prologue loads; the
    # consumer waits for it after the KV loop, runs the aux-only pass under
    # the last PV and reads every aux chunk from its own sO rows.
    issue = code.index(
        "cute.copy(_flash_tma_aux0, tAgA_aux_tma[None, flash_m_tile, 0, flash_bh]"
    )
    allocate = code.index("flash_tmem.allocate(512)")
    loop = code.index("for flash_kv in cutlass.range(_flash_num_kv_tiles, unroll=1)")
    aux_wait = code.index("mbar_spin_wait(flash_aux_full_ptr, flash_aux_full_phase")
    aux_load = code.index("cute.autovec_copy(_ep_tOsO[None, 0, 0, 0], _ep_a0_0)")
    final_wait = code.index(
        "flash_o_full = flash_mma_o_cons.wait_and_advance()\n        flash_inv_sum"
    )
    assert issue < allocate < loop < aux_wait < aux_load < final_wait
    assert "prefetch_global_l2(" not in code and "_ep_tgaux0" not in code
    assert (
        "fa4_correction_epilogue_partitions(flash_pvt, tOtO, sO, flash_local_tidx, 64, 64"
        in code
    )
    assert "fa4_store_o_smem_to_gmem_whole(" in code
    # The direct-store route keeps the L2 prefetch of its gmem aux rows.
    direct = _code(_xsa_kernel(), (q, k, v), **_FLAT, cute_flash_epi_stg=False)
    assert "_flash_tma_aux0" not in direct
    assert "cute.autovec_copy(_ep_tdgaux0[None, 0, 0], _ep_a0_0)" in direct


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------

_RUNTIME_CONFIGS = (
    pytest.param(_FLAT, id="flat-staged-c64"),
    pytest.param({**_FLAT, "cute_flash_corr_tile_size": 16}, id="flat-staged-c16"),
    pytest.param({**_FLAT, "cute_flash_corr_tile_size": 32}, id="flat-staged-c32"),
    pytest.param({**_FLAT, "cute_flash_epi_stg": False}, id="flat-direct"),
    pytest.param(
        {**_FLAT, "cute_flash_epi_stg_store": "slice"}, id="flat-staged-slice"
    ),
    pytest.param(_PERSISTENT, id="persistent-staged"),
    pytest.param({**_PERSISTENT, "cute_flash_epi_stg": False}, id="persistent-direct"),
    pytest.param({**_FLAT, "cute_flash_kv_stage": 3}, id="flat-staged-kv3"),
    pytest.param(_M64, id="flat-m64"),
    pytest.param({**_M64, "cute_flash_corr_tile_size": 16}, id="flat-m64-c16"),
    pytest.param({**_M64, "cute_flash_packed_reduce": False}, id="flat-m64-unpacked"),
    pytest.param({**_M64, "cute_flash_rescale_threshold": 0.0}, id="flat-m64-rescale"),
    pytest.param({**_M64, "cute_flash_e2e_schedule": "xu"}, id="flat-m64-xu"),
)


@onlyBackends(["cute"])
@pytest.mark.parametrize("config", _RUNTIME_CONFIGS)
@pytest.mark.parametrize(
    ("shape", "dtype"),
    (
        ((1, 4, 256, 64), torch.bfloat16),
        ((1, 4, 256, 64), torch.float16),
        ((2, 3, 384, 64), torch.bfloat16),
        ((1, 2, 256, 128), torch.bfloat16),
    ),
)
def test_ws_attention_matches_reference(
    config: dict[str, object], shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    q, k, v = _qkv(*shape, dtype)
    expected_out, expected_lse = _reference(q, k, v)
    code, (out, lse) = code_and_output(_attention_kernel(), (q, k, v), **config)
    assert "flash_tmem.allocate(512)" in code
    # The M=64 rows must run the 64-row body, not a silent 128-row fallback.
    assert f"q_rows={config.get('cute_flash_q_tile_m', 128)})" in code
    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_ws_attention_two_inputs_do_not_alias_the_output() -> None:
    # Two different inputs on the same compiled kernel: guards against a store
    # path that leaves the output buffer stale.
    kernel = _attention_kernel()
    outs = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        q, k, v = (
            torch.randn(1, 4, 256, 64, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        )
        _, (out, _) = code_and_output(kernel, (q, k, v), **_FLAT)
        expected, _ = _reference(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        outs.append(out)
    assert not torch.allclose(outs[0], outs[1])


@onlyBackends(["cute"])
@pytest.mark.parametrize("config", (_FLAT, {**_FLAT, "cute_flash_epi_stg": False}))
def test_ws_causal_matches_reference(config: dict[str, object]) -> None:
    q, k, v = _qkv(1, 2, 512, 64, torch.bfloat16)
    expected_out, _ = _reference(q, k, v, causal=True)
    code, out = code_and_output(_causal_attention, (q, k, v), **config)
    assert "flash_active_count" in code
    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)


@onlyBackends(["cute"])
@pytest.mark.parametrize("config", (_FLAT, _PERSISTENT))
def test_ws_relu_epilogue_matches_reference(config: dict[str, object]) -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    expected, _ = _reference(q, k, v)
    code, out = code_and_output(_relu_kernel(), (q, k, v), **config)
    assert "relu_output=True" in code
    torch.testing.assert_close(out, torch.relu(expected), atol=1e-2, rtol=1e-2)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "config",
    (
        pytest.param(_FLAT, id="flat-c64"),
        pytest.param({**_FLAT, "cute_flash_corr_tile_size": 16}, id="flat-c16"),
        pytest.param(_PERSISTENT, id="persistent-c64"),
        pytest.param({**_FLAT, "cute_flash_epi_stg": False}, id="flat-direct"),
    ),
)
def test_ws_row_epilogue_matches_reference(config: dict[str, object]) -> None:
    from examples.xsa import ref_xsa

    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code, out = code_and_output(_xsa_kernel(), (q, k, v), **config)
    if config.get("cute_flash_epi_stg", True):
        assert "cute.copy(_flash_tma_aux0" in code
    else:
        assert "_ep_gaux0 = cute.flat_divide(_flash_mEpiAux0" in code
    torch.testing.assert_close(out, ref_xsa(q, k, v), atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------
# 64-row query tiles
# --------------------------------------------------------------------------


def test_q_tile_m64_is_legal_only_for_the_flat_staged_dense_body() -> None:
    def resolve(**overrides: object) -> int:
        return cute_flash.resolve_flash_config(
            64,
            2,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_PERSISTENT_KEY: False,
                cute_flash.FLASH_Q_TILE_M_KEY: 64,
                **overrides,
            },
        ).q_tile_m

    assert resolve() == 64
    assert resolve(cute_flash_persistent=True) == 128
    assert resolve(cute_flash_epi_stg=False) == 128
    assert resolve(cute_flash_s_stage=1) == 128
    assert (
        cute_flash.resolve_flash_config(
            64,
            2,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_PERSISTENT_KEY: False,
                cute_flash.FLASH_Q_TILE_M_KEY: 64,
            },
            is_causal=True,
        ).q_tile_m
        == 128
    )
    assert (
        cute_flash.resolve_flash_config(
            64,
            2,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                cute_flash.FLASH_Q_TILE_M_KEY: 64,
            },
        ).q_tile_m
        == 128
    )
    # Default stays 128. A family-pinned ws surface keeps the persistent
    # default (from which 64 cannot round-trip), so 64 is sampled from the
    # shared paired surface below.
    dense = cute_flash.flash_autotune_fragments(
        64, 2, num_bh=4, dtype=torch.bfloat16, pipeline_family_override="ws_overlap"
    )
    assert dense[cute_flash.FLASH_Q_TILE_M_KEY].default() == 128
    assert tuple(dense[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == (128,)
    assert set(dense[cute_flash.FLASH_Q_TILE_M_KEY].choices) == {64, 128}
    causal = cute_flash.flash_autotune_fragments(
        64,
        2,
        num_bh=4,
        dtype=torch.bfloat16,
        is_causal=True,
    )
    assert tuple(causal[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == (128,)
    # Odd KV counts are ws-only classes that the autotuner enumerates exactly:
    # the tile height is not sampled there.
    odd = cute_flash.flash_autotune_fragments(
        64, 3, num_bh=4, dtype=torch.bfloat16, pipeline_family_override="ws_overlap"
    )
    assert tuple(odd[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == (128,)
    # The shared paired surface (fa4 default topology) also samples 64 so the
    # ws_overlap family can reach it.
    shared = cute_flash.flash_autotune_fragments(
        64, 2, num_bh=4, dtype=torch.bfloat16, standard_dense_output=True
    )
    assert set(shared[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == {64, 128}


def test_q_tile_m64_is_canonicalized_away_for_fused_or_modified_rows() -> None:
    # Score modifiers and fused row epilogues have the 128-row body only; the
    # resolver and the search surface both learn that through plain_row_body,
    # so a tuned config cannot record 64 while running the 128-row body.
    config = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
        cute_flash.FLASH_PERSISTENT_KEY: False,
        cute_flash.FLASH_Q_TILE_M_KEY: 64,
    }
    assert cute_flash.resolve_flash_config(64, 2, config).q_tile_m == 64
    assert (
        cute_flash.resolve_flash_config(64, 2, config, plain_row_body=False).q_tile_m
        == 128
    )
    common = {"num_bh": 4, "dtype": torch.bfloat16, "standard_dense_output": True}
    plain = cute_flash.flash_autotune_fragments(64, 2, **common)
    assert set(plain[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == {64, 128}
    fused = cute_flash.flash_autotune_fragments(
        64, 2, plain_row_body=False, has_row_epilogue=True, **common
    )
    assert tuple(fused[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == (128,)


@onlyBackends(["cute"])
def test_xsa_surface_has_no_64_row_tile() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    xsa = _xsa_kernel().bind((q, k, v))
    spec = xsa.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_plain_row_body is False
    fragment = spec._cute_flash_autotune_fragments()[cute_flash.FLASH_Q_TILE_M_KEY]
    assert tuple(fragment.search_choices or ()) == (128,)
    normalized = xsa._normalized_config_copy(helion.Config(**_M64))
    assert normalized.config[cute_flash.FLASH_Q_TILE_M_KEY] == 128
    # Plain attention on the same shape keeps the knob and records 64.
    attention = _attention_kernel().bind((q, k, v))
    assert attention.config_spec._cute_flash_plain_row_body is True
    fragment = attention.config_spec._cute_flash_autotune_fragments()[
        cute_flash.FLASH_Q_TILE_M_KEY
    ]
    assert set(fragment.search_choices or ()) == {64, 128}
    normalized = attention._normalized_config_copy(helion.Config(**_M64))
    assert normalized.config[cute_flash.FLASH_Q_TILE_M_KEY] == 64


def test_ws_corr_tile_size_is_canonical_unless_the_epilogue_is_staged() -> None:
    # Only the staged ws epilogue reads the chunk width; direct-store and
    # single-stage configs normalize to one value instead of three aliases.
    def resolve(**overrides: object) -> int:
        return cute_flash.resolve_flash_config(
            128,
            2,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_PERSISTENT_KEY: False,
                **overrides,
            },
        ).corr_tile_size

    for width in (16, 32, 64):
        assert resolve(cute_flash_corr_tile_size=width) == width
        assert resolve(cute_flash_corr_tile_size=width, cute_flash_epi_stg=False) == 64
        assert resolve(cute_flash_corr_tile_size=width, cute_flash_s_stage=1) == 64


@onlyBackends(["cute"])
def test_q_tile_m64_codegen_uses_two_row_tmem_copies_and_quad_reductions() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_attention_kernel(), (q, k, v), **_M64)
    assert "q_rows=64)" in code
    assert "partition_shape_C((64, 128))" in code
    assert "cute.select((64, 128, 64), mode=[0, 2])" in code
    assert "Ld16x256bOp(cute_tcgen05_flash.Repetition(16))" in code
    assert "St16x128bOp(cute_tcgen05_flash.Repetition(16))" in code
    assert "tLDrS_a, tLDrS_b = _helion_flash_rt.ws_m64_row_views(tLDrS)" in code
    assert code.count("_helion_flash_rt.quad_max(") == 2
    assert code.count("_helion_flash_rt.quad_sum(") == 2
    assert "rescale_o_tmem_m64(" in code
    assert "ws_m64_epilogue_to_smem(" in code
    assert "(64, 64), (flash_m_tile, 0))" in code
    # Both rows of a thread store their LSE at their own row coordinates.
    assert (
        "_flash_mLSE[flash_m_tile * 64 + cutlass.Int32(tLDcS[0][0]), flash_bh]" in code
    )
    assert (
        "_flash_mLSE[flash_m_tile * 64 + cutlass.Int32(tLDcS[2][0]), flash_bh]" in code
    )
    assert "'q_tile_m': 64" in code
    # Fused row epilogues fall back to the 128-row body.
    xsa = _code(_xsa_kernel(), (q, k, v), **_M64)
    assert "q_rows=128)" in xsa
    assert "'q_tile_m': 128" in xsa


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("shape", "dtype"),
    (
        ((1, 4, 256, 64), torch.bfloat16),
        ((1, 4, 256, 64), torch.float16),
        ((2, 3, 384, 64), torch.bfloat16),
        ((1, 2, 512, 128), torch.bfloat16),
    ),
)
def test_q_tile_m64_matches_reference_and_the_128_row_tile(
    shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    kernel = _attention_kernel()
    for seed in (3, 4):
        torch.manual_seed(seed)
        q, k, v = (torch.randn(*shape, dtype=dtype, device=DEVICE) for _ in range(3))
        expected_out, expected_lse = _reference(q, k, v)
        code, (out, lse) = code_and_output(kernel, (q, k, v), **_M64)
        assert "q_rows=64)" in code
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)
        # A freed same-sized buffer holding a value no correct run produces:
        # the caching allocator hands it to the 128-row output below, so a
        # store path that leaves the output stale cannot pass. ``out`` and
        # ``lse`` stay alive, so their blocks cannot be recycled either.
        poison = torch.full_like(out, 7.0)
        del poison
        _, (out128, lse128) = code_and_output(
            kernel, (q.clone(), k.clone(), v.clone()), **_FLAT
        )
        torch.testing.assert_close(out, out128, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, lse128, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_q_tile_m64_relu_epilogue_matches_reference() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    expected, _ = _reference(q, k, v)
    code, out = code_and_output(_relu_kernel(), (q, k, v), **_M64)
    assert "ws_m64_epilogue_to_smem(" in code
    assert "relu_output=True" in code
    torch.testing.assert_close(out, torch.relu(expected), atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# One-pass softmax over two KV tiles
# --------------------------------------------------------------------------


def test_ws_one_pass_is_legal_only_for_two_kv_tiles_of_the_64_row_body() -> None:
    def resolve_config(
        num_kv: int = 2, **overrides: object
    ) -> cute_flash.FlashAttentionConfig:
        return cute_flash.resolve_flash_config(
            64,
            num_kv,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_PERSISTENT_KEY: False,
                cute_flash.FLASH_Q_TILE_M_KEY: 64,
                cute_flash.FLASH_WS_ONE_PASS_KEY: True,
                **overrides,
            },
        )

    def resolve(num_kv: int = 2, **overrides: object) -> bool:
        return resolve_config(num_kv, **overrides).ws_one_pass

    assert resolve() is True
    assert resolve(cute_flash_kv_stage=3) is True
    # Exactly two KV tiles and the 64-row body (any reason the tile falls
    # back to 128 rows takes the one-pass form with it). The two-warpgroup
    # body keeps a ring of at least two stages, so both K tiles are issued in
    # the prologue.
    assert resolve(num_kv=1) is False
    assert resolve(num_kv=3) is False
    single_stage = resolve_config(cute_flash_kv_stage=1)
    assert single_stage.kv_stage == 2 and single_stage.ws_one_pass is True
    assert resolve(cute_flash_q_tile_m=128) is False
    assert resolve(cute_flash_persistent=True) is False
    assert resolve(cute_flash_epi_stg=False) is False
    # Off unless asked for.
    assert (
        cute_flash.resolve_flash_config(
            64,
            2,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_PERSISTENT_KEY: False,
                cute_flash.FLASH_Q_TILE_M_KEY: 64,
            },
        ).ws_one_pass
        is False
    )
    # Sampled where the 64-row tile is (the shared paired surface), only for
    # sequences of exactly two KV tiles; pinned ws surfaces and other lengths
    # keep the single inert value.
    paired = cute_flash.flash_autotune_fragments(
        64, 2, num_bh=4, dtype=torch.bfloat16, standard_dense_output=True
    )
    assert set(paired[cute_flash.FLASH_Q_TILE_M_KEY].search_choices or ()) == {64, 128}
    assert set(paired[cute_flash.FLASH_WS_ONE_PASS_KEY].search_choices or ()) == {
        False,
        True,
    }
    for fragments in (
        cute_flash.flash_autotune_fragments(
            64, 4, num_bh=4, dtype=torch.bfloat16, standard_dense_output=True
        ),
        cute_flash.flash_autotune_fragments(
            64,
            2,
            num_bh=4,
            dtype=torch.bfloat16,
            pipeline_family_override="ws_overlap",
        ),
        cute_flash.flash_autotune_fragments(
            64, 2, num_bh=4, dtype=torch.bfloat16, is_causal=True
        ),
    ):
        assert tuple(
            fragments[cute_flash.FLASH_WS_ONE_PASS_KEY].search_choices or ()
        ) == (False,)


@onlyBackends(["cute"])
def test_ws_one_pass_canonicalizes_to_off_where_it_is_illegal() -> None:
    kernel = _attention_kernel()
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    bound = kernel.bind((q, k, v))
    on = helion.Config(**_M64_ONE_PASS)
    bound.config_spec.normalize(on)
    assert on.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is True
    off = helion.Config(**{**_M64_ONE_PASS, "cute_flash_q_tile_m": 128})
    bound.config_spec.normalize(off)
    assert off.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is False
    # Four KV tiles: the knob is inert and normalizes to False.
    q4, k4, v4 = _qkv(1, 4, 512, 64, torch.bfloat16)
    longer = helion.Config(**_M64_ONE_PASS)
    kernel.bind((q4, k4, v4)).config_spec.normalize(longer)
    assert longer.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is False
    # Fused row epilogues have no 64-row body.
    xsa = helion.Config(**_M64_ONE_PASS)
    _xsa_kernel().bind((q, k, v)).config_spec.normalize(xsa)
    assert xsa.config[cute_flash.FLASH_Q_TILE_M_KEY] == 128
    assert xsa.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is False


@onlyBackends(["cute"])
def test_ws_one_pass_codegen_reads_both_score_tiles_before_one_exp2_sweep() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code = _code(_attention_kernel(), (q, k, v), **_M64_ONE_PASS)
    assert "'q_tile_m': 64" in code and "'ws_one_pass': True" in code
    # Producer: QK(0) -> S0, QK(1) -> S1; both V tiles and the O accumulator
    # are waited for before the P handoffs; PV(0) then PV(1) accumulating with
    # one mma_o commit.
    assert code.count("flash_s_handle = flash_mma_s_prod.acquire_and_advance()") == 2
    assert code.count("flash_p_full = flash_p_ready_cons.wait_and_advance()") == 2
    assert code.count("flash_o_handle = flash_mma_o_prod.acquire_and_advance()") == 1
    assert code.count("flash_o_handle.commit()") == 1
    assert code.index("flash_v_full1 = flash_v_cons.wait_and_advance()") < code.index(
        "flash_p_full = flash_p_ready_cons.wait_and_advance()"
    )
    assert "Field.ACCUMULATE, cutlass.Boolean(True))" in code
    assert "flash_o_started" not in code
    # Consumer: both S tiles are in registers before the max; the exp2 sweep
    # hands off P(0) before the second tile's exp2; two p_ready arrivals.
    s1_load = code.index("cute.copy(flash_tiled_ld1, tLDtS1, tLDrS1)")
    assert code.index("cute.copy(flash_tiled_ld0, tLDtS0, tLDrS0)") < s1_load
    assert s1_load < code.index("flash_row_max_a = _helion_flash_rt.quad_max(")
    assert code.index("flash_row_max_b = ") < code.index("exp2_split_inplace(")
    assert code.count("exp2_split_inplace(") == 4
    assert code.index("cute.copy(flash_tiled_st0, tSTrS0, tSTtS0)") < code.index(
        "exp2_split_inplace(tLDrS1_a"
    )
    assert code.count("flash_p_handle = flash_p_ready_prod.acquire_and_advance()") == 2
    assert "rescale_o_tmem_m64(" not in code
    assert (
        code.index("flash_s_full0.release()")
        < code.index("exp2_split_inplace(tLDrS1_a")
        < code.index("flash_s_full1.release()")
    )
    assert "cutlass.range(_flash_num_kv_tiles" not in code
    # The staged 64-row epilogue and the two-row LSE store are unchanged.
    assert "ws_m64_epilogue_to_smem(" in code
    assert (
        "_flash_mLSE[flash_m_tile * 64 + cutlass.Int32(tLDcS[2][0]), flash_bh]" in code
    )
    # A three-deep ring still prefetches and consumes exactly two K/V tiles.
    three = _code(
        _attention_kernel(), (q, k, v), **{**_M64_ONE_PASS, "cute_flash_kv_stage": 3}
    )
    assert three.count("cute.copy(_flash_tma_k, tKgK[None, ") == 2
    assert three.count("flash_k_full = flash_k_cons.wait_and_advance()") == 2


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("shape", "dtype"),
    (
        ((1, 4, 256, 64), torch.bfloat16),
        ((1, 4, 256, 64), torch.float16),
        ((2, 3, 256, 128), torch.bfloat16),
    ),
)
def test_ws_one_pass_matches_reference_and_the_two_pass_tile(
    shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    kernel = _attention_kernel()
    outs = []
    for seed in (5, 6):
        torch.manual_seed(seed)
        q, k, v = (torch.randn(*shape, dtype=dtype, device=DEVICE) for _ in range(3))
        expected_out, expected_lse = _reference(q, k, v)
        code, (out, lse) = code_and_output(kernel, (q, k, v), **_M64_ONE_PASS)
        assert "flash_s_full1 = flash_mma_s_cons.wait_and_advance()" in code
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)
        poison = torch.full_like(out, 7.0)
        del poison
        _, (out2, lse2) = code_and_output(
            kernel, (q.clone(), k.clone(), v.clone()), **_M64
        )
        torch.testing.assert_close(out, out2, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, lse2, atol=2e-2, rtol=2e-2)
        outs.append(out)
    # Two different inputs through the same compiled kernel: no stale output.
    assert not torch.allclose(outs[0], outs[1])


@onlyBackends(["cute"])
def test_ws_one_pass_relu_epilogue_matches_reference() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    expected, _ = _reference(q, k, v)
    code, out = code_and_output(_relu_kernel(), (q, k, v), **_M64_ONE_PASS)
    assert "flash_s_full1 = flash_mma_s_cons.wait_and_advance()" in code
    assert "relu_output=True" in code
    torch.testing.assert_close(out, torch.relu(expected), atol=1e-2, rtol=1e-2)


def test_small_grids_seed_the_64_row_tile_with_the_one_pass_softmax() -> None:
    def seeds(num_kv: int, num_bh: int, sm_count: int) -> list[helion.Config]:
        return [
            seed
            for seed in cute_flash.flash_attention_seed_configs(
                64,
                num_kv,
                num_bh=num_bh,
                dtype=torch.bfloat16,
                standard_dense_output=True,
                device_sm_count=sm_count,
            )
            if seed.config.get(cute_flash.FLASH_Q_TILE_M_KEY) == 64
        ]

    # 4 heads x 2 tiles of 128 rows cannot fill 148 SMs: one flat ws_overlap
    # seed with the 64-row tile and, at two KV tiles, its one-pass softmax.
    (seed,) = seeds(2, 4, 148)
    assert seed.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
    assert seed.config[cute_flash.FLASH_PERSISTENT_KEY] is False
    assert seed.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is True
    # The seed resolves as written: the 64-row tile exists only with the
    # staged O epilogue (the fragment default is the direct store, which
    # would canonicalize the tile back to 128 rows and the softmax to two
    # passes), and it carries the two-stage ring the tile measured best with.
    assert seed.config[cute_flash.FLASH_EPI_STG_KEY] is True
    resolved = cute_flash.resolve_flash_config(64, 2, seed.config)
    assert resolved.q_tile_m == 64
    assert resolved.ws_one_pass is True
    assert resolved.epi_stg is True
    assert resolved.kv_stage == 2
    # Longer sequences keep the 64-row tile but the one-pass form is not legal.
    (longer,) = seeds(4, 4, 148)
    assert longer.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is False
    resolved = cute_flash.resolve_flash_config(64, 4, longer.config)
    assert resolved.q_tile_m == 64
    assert resolved.ws_one_pass is False
    # Grids that fill the device, and an unknown SM count, seed nothing extra.
    assert seeds(2, 148, 148) == []
    assert seeds(2, 4, 0) == []
    # Surfaces without the 64-row tile (fused row programs) never carry it.
    assert not [
        seed
        for seed in cute_flash.flash_attention_seed_configs(
            64,
            2,
            num_bh=4,
            dtype=torch.bfloat16,
            standard_dense_output=True,
            has_row_epilogue=True,
            plain_row_body=False,
            device_sm_count=148,
        )
        if seed.config.get(cute_flash.FLASH_Q_TILE_M_KEY) == 64
    ]


def test_flat_ws_overlap_seed_carries_the_staged_epilogue() -> None:
    # The flat ws_overlap source variant seeds the staged-store flat kernel
    # the measured winners use, not the direct store of the fa4 fragment
    # default, on surfaces whose default family is fa4 and on surfaces that
    # require ws_overlap alike.
    for num_kv, num_bh, requires_ws_overlap in ((64, 16, False), (2, 4, True)):
        flat = [
            seed
            for seed in cute_flash.flash_attention_seed_configs(
                64,
                num_kv,
                num_bh=num_bh,
                dtype=torch.bfloat16,
                standard_dense_output=True,
                requires_ws_overlap=requires_ws_overlap,
            )
            if seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == "ws_overlap"
            and seed.config.get(cute_flash.FLASH_PERSISTENT_KEY) is False
        ]
        assert flat
        for seed in flat:
            resolved = cute_flash.resolve_flash_config(64, num_kv, seed.config)
            assert not resolved.persistent
            assert resolved.epi_stg is True


@onlyBackends(["cute"])
def test_small_grid_seed_survives_normalization_and_seed_deduplication() -> None:
    # The bound (1, 4, 256, 64) problem has 8 CTAs of 128 rows on a device
    # with far more SMs, so the compiler seeds carry the one-pass 64-row tile
    # once, and it stays that tile through normalization, the seed
    # deduplication of the initial population and the population itself.
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    bound = _attention_kernel().bind((q, k, v))
    spec = bound.config_spec
    assert spec._cute_flash_device_sm_count > 8
    one_pass = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get(cute_flash.FLASH_WS_ONE_PASS_KEY) is True
    ]
    assert len(one_pass) == 1
    (seed,) = one_pass
    # The seed canonicalizes the way the initial population transfers it.
    config_gen = spec.create_config_generation()
    _flat, normalized = config_gen.canonicalize_flat(config_gen.flatten(seed))
    assert normalized.config[cute_flash.FLASH_Q_TILE_M_KEY] == 64
    assert normalized.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is True
    assert normalized.config[cute_flash.FLASH_EPI_STG_KEY] is True
    assert normalized.config[cute_flash.FLASH_KV_STAGE_KEY] == 2
    assert "ws_m64_epilogue_to_smem" in bound.to_code(normalized)
    survivors = [
        config
        for _flat, config in config_gen.seed_flat_config_pairs()
        if config.config[cute_flash.FLASH_WS_ONE_PASS_KEY] is True
    ]
    assert survivors == [normalized]
    # Every compiler seed, this one included, is a member of the initial
    # population of a full-effort search.
    population = config_gen.random_population(100)
    assert normalized in population
