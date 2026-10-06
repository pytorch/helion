"""The ``row_mma`` flash pipeline family: register-MMA row programs.

One CTA per 8- or 16-row query tile, its warps splitting the keys, warp-level
``mma.sync.m16n8k16`` for ``S^T = K Q^T`` and ``O^T = V^T P^T`` (no M padding
for eight query rows), exact fp32 softmax, ``cp.async`` staging with a
double-buffered chunk loop for long key ranges, and a deterministic fixed-order
cross-warp combine.  Measured in round 4 of the flash-small-shape work at
``(1, 4, 256, 64)``: 3.07-3.17 us device span against 3.62-3.68 us for the
Triton winner and 4.55-4.62 us for the 64-row tcgen05 one-pass tile.

A fused row epilogue (``flash_row_epilogue.py``) runs inside the combine
epilogue: each thread evaluates the program on its row's eight normalized
columns, head_dim reductions complete over the lanes sharing the row, aux rows
are staged through shared memory by ``cp.async`` at kernel entry.  XSA at
``(1, 4, 256, 64)``: 3.26-3.30 us device span against 3.78 us for the Triton
winner and 5.54 us for the 128-row ws_overlap tile with the fused epilogue.
"""

from __future__ import annotations

import dataclasses
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
from helion._compiler.cute import cute_flash_row_mma
from helion.autotuner.config_fragment import EnumFragment

FAMILY = cute_flash.FLASH_PIPELINE_FAMILY_KEY
WARPS = cute_flash.FLASH_ROW_WARPS_KEY
TILE_M = cute_flash.FLASH_ROW_TILE_M_KEY
_ROW = {"block_sizes": [1, 128, 128], FAMILY: "row_mma"}


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


def _ref_xsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """fp32 reference of ``examples.xsa`` (``F.normalize`` semantics)."""
    y = _attention_fp32(q, k, v)
    vn = torch.nn.functional.normalize(v.float(), dim=-1, eps=eps)
    return (y - (y * vn).sum(dim=-1, keepdim=True) * vn).to(q.dtype)


def _attention_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gated_by_fp32_rows(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, g_in: torch.Tensor
) -> torch.Tensor:
    """``out = o * g`` with a separate fp32 gate of the output's geometry."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    g_view = g_in.reshape([-1, m_dim, head_dim])
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
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        o = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = (o * g_view[tile_b, tile_m, :]).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _amax_normalized_rows(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, eps: float
) -> torch.Tensor:
    """``out = o / max(amax(|o|), eps)``: a max reduction over head_dim."""
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
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        o = acc / l_i[:, :, None]
        scale = torch.clamp(torch.amax(torch.abs(o), dim=-1, keepdim=True), min=eps)
        out[tile_b, tile_m, :] = (o / scale).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _biased_xsa_like(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """XSA's row epilogue over scores with an additive bias (a score modifier)."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    bias_view = bias.reshape([-1, m_dim, n_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + bias_view[tile_b, tile_m, tile_n]
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        o = acc / l_i[:, :, None]
        v_self = v_view[tile_b, tile_m, :].to(torch.float32)
        v_norm = torch.sqrt(torch.sum(v_self * v_self, dim=-1, keepdim=True))
        vn = v_self / torch.clamp(v_norm, min=eps)
        proj = torch.sum(o * vn, dim=-1, keepdim=True)
        out[tile_b, tile_m, :] = (o - proj * vn).to(out.dtype)
    return out.view(q_in.size())


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
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # fp32 math rather than SDPA: PyTorch's default SDPA backend reads an
    # 8-byte-aligned base from the 16-byte-aligned address below it, too.
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    out = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    return out, lse


def _active(fragment: object) -> tuple[object, ...]:
    assert isinstance(fragment, EnumFragment)
    return (
        fragment.choices if fragment.search_choices is None else fragment.search_choices
    )


def _legal_families(head_dim: int, num_kv: int, **options: object) -> tuple[str, ...]:
    settings: dict[str, object] = {
        "num_bh": 4,
        "dtype": torch.bfloat16,
        "is_causal": False,
        "has_kv_tile_pruning": False,
        "requires_ws_overlap": False,
        "small_biased_candidate": False,
        "standard_dense_output": True,
        "standard_causal_output": False,
        "output_requires_tma": False,
        "supports_tensor_4d_tma": True,
        "requested_family": None,
    }
    settings.update(options)
    return cute_flash._flash_legal_autotune_pipeline_families(
        head_dim,
        num_kv,
        **settings,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Config resolution, legality and the search surface
# --------------------------------------------------------------------------


def test_row_mma_resolves_its_two_knobs_over_a_fixed_base() -> None:
    cfg = cute_flash.resolve_flash_config(
        64, 2, {FAMILY: "row_mma"}, dtype=torch.bfloat16, num_bh=4
    )
    assert cfg.topology == cfg.pipeline_family == "row_mma"
    assert (cfg.row_warps, cfg.row_tile_m) == (4, 8)
    assert not cfg.persistent
    # The two row knobs are read; every tcgen05 knob is dead and pinned, so a
    # config that also sets them resolves to the same base.
    other = cute_flash.resolve_flash_config(
        64,
        2,
        {
            FAMILY: "row_mma",
            WARPS: 8,
            TILE_M: 16,
            cute_flash.FLASH_KV_STAGE_KEY: 4,
            cute_flash.FLASH_S_STAGE_KEY: 1,
            cute_flash.FLASH_Q_TILE_M_KEY: 64,
            cute_flash.FLASH_WS_ONE_PASS_KEY: True,
            cute_flash.FLASH_EPI_STG_KEY: True,
        },
        dtype=torch.bfloat16,
        num_bh=4,
    )
    assert (other.row_warps, other.row_tile_m) == (8, 16)
    assert dataclasses.replace(other, row_warps=4, row_tile_m=8) == cfg
    values = cute_flash.flash_effective_config_values(other)
    assert values[FAMILY] == "row_mma"
    assert (values[WARPS], values[TILE_M]) == (8, 16)
    # Values outside the knob choices fall back to the defaults.
    odd = cute_flash.resolve_flash_config(
        64, 2, {FAMILY: "row_mma", WARPS: 3, TILE_M: 32}, dtype=torch.bfloat16
    )
    assert (odd.row_warps, odd.row_tile_m) == (4, 8)
    # Other families keep the row knobs at their defaults.
    ws = cute_flash.resolve_flash_config(
        64, 2, {FAMILY: "ws_overlap", WARPS: 8, TILE_M: 16}, dtype=torch.bfloat16
    )
    assert ws.pipeline_family == "ws_overlap"
    assert (ws.row_warps, ws.row_tile_m) == (4, 8)


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "dtype", "options", "legal", "searched"),
    (
        # Four heads: up to four KV tiles stay inside the 256-row-program
        # search bound; longer sequences keep the family legal for explicit
        # configs but out of unattended searches.
        (64, 2, torch.bfloat16, {}, True, True),
        (64, 1, torch.float16, {}, True, True),
        (64, 4, torch.float16, {}, True, True),
        (64, 5, torch.float16, {}, True, False),
        (64, 97, torch.float16, {}, True, False),
        (128, 3, torch.bfloat16, {}, True, True),
        (128, 32, torch.float16, {}, True, False),
        (
            64,
            2,
            torch.bfloat16,
            {"is_causal": True, "standard_dense_output": False},
            False,
            False,
        ),
        # A score modifier (plain_row_body False without a row epilogue, or
        # has_score_modifiers) excludes the family; a fused row epilogue alone
        # runs in the combine epilogue and is searched like the plain body.
        (64, 2, torch.bfloat16, {"plain_row_body": False}, False, False),
        (
            64,
            2,
            torch.bfloat16,
            {"plain_row_body": False, "has_score_modifiers": True},
            False,
            False,
        ),
        (
            64,
            2,
            torch.bfloat16,
            {"has_row_epilogue": True, "plain_row_body": False},
            True,
            True,
        ),
        (
            128,
            2,
            torch.float16,
            {"has_row_epilogue": True, "plain_row_body": False},
            True,
            True,
        ),
        (
            64,
            64,
            torch.bfloat16,
            {"has_row_epilogue": True, "plain_row_body": False},
            True,
            False,
        ),
        (
            64,
            2,
            torch.bfloat16,
            {
                "has_row_epilogue": True,
                "plain_row_body": False,
                "has_score_modifiers": True,
            },
            False,
            False,
        ),
        (64, 2, torch.bfloat16, {"has_kv_tile_pruning": True}, False, False),
        (64, 2, torch.bfloat16, {"small_biased_candidate": True}, False, False),
        (64, 2, torch.bfloat16, {"requires_ws_overlap": True}, False, False),
    ),
)
def test_row_mma_legality_is_structural_and_searched_on_small_grids(
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype,
    options: dict[str, object],
    legal: bool,
    searched: bool,
) -> None:
    families = _legal_families(head_dim, num_kv, dtype=dtype, **options)
    assert ("row_mma" in families) == searched
    explicit = _legal_families(
        head_dim, num_kv, dtype=dtype, requested_family="row_mma", **options
    )
    assert (explicit == ("row_mma",)) == legal
    resolved = cute_flash.resolve_flash_config(
        head_dim,
        num_kv,
        {FAMILY: "row_mma"},
        dtype=dtype,
        num_bh=4,
        is_causal=bool(options.get("is_causal")),
        has_kv_tile_pruning=bool(options.get("has_kv_tile_pruning")),
        requires_ws_overlap=bool(options.get("requires_ws_overlap")),
        small_biased_candidate=bool(options.get("small_biased_candidate")),
        plain_row_body=bool(options.get("plain_row_body", True)),
        has_row_epilogue=bool(options.get("has_row_epilogue")),
        has_score_modifiers=bool(options.get("has_score_modifiers")),
    )
    assert (resolved.pipeline_family == "row_mma") == legal


def test_row_mma_declines_an_aux_staging_that_does_not_fit() -> None:
    # The fused row epilogue's aux rows are known at codegen only. Knobs whose
    # staging does not fit next to the K/V stages fall back to the default
    # knobs (hd128 x 16 rows x 8 warps holds four fp32 aux rows, not five);
    # when even the default cannot be planned the family declines this
    # binding and the tcgen05 default takes it: the kernel still compiles.
    fused = {"plain_row_body": False, "has_row_epilogue": True}
    wide = {FAMILY: "row_mma", WARPS: 8, TILE_M: 16}
    f32 = "cutlass.Float32"
    resolve = cute_flash.resolve_flash_config
    four = resolve(
        128,
        6,
        wide,
        dtype=torch.bfloat16,
        num_bh=1,
        row_mma_aux_dtypes=[f32] * 4,
        **fused,
    )
    assert four.pipeline_family == "row_mma"
    assert (four.row_warps, four.row_tile_m) == (8, 16)
    five = resolve(
        128,
        6,
        wide,
        dtype=torch.bfloat16,
        num_bh=1,
        row_mma_aux_dtypes=[f32] * 5,
        **fused,
    )
    assert five.pipeline_family == "row_mma"
    assert (five.row_warps, five.row_tile_m) == (4, 8)
    many = resolve(
        128,
        6,
        wide,
        dtype=torch.bfloat16,
        num_bh=1,
        row_mma_aux_dtypes=[f32] * 40,
        **fused,
    )
    default = resolve(128, 6, None, dtype=torch.bfloat16, num_bh=1, **fused)
    assert many.pipeline_family == default.pipeline_family != "row_mma"
    assert many == default


def test_row_mma_search_grid_bound() -> None:
    grid = cute_flash_row_mma.row_mma_search_grid
    # 256 row programs at the default 8-row tile: sixteen 128-row tiles.
    assert cute_flash_row_mma.ROW_MMA_SEARCH_MAX_ROW_PROGRAMS == 256
    assert grid(num_bh=1, num_kv=16)
    assert not grid(num_bh=1, num_kv=17)
    assert grid(num_bh=4, num_kv=2)  # (1, 4, 256, 64): 128 programs
    assert grid(num_bh=16, num_kv=1)
    assert not grid(num_bh=16, num_kv=8)  # (2, 8, 1024, 64): 2048 programs
    assert not grid(num_bh=None, num_kv=1)


def test_row_mma_surface_is_length_invariant_inside_the_search_bound() -> None:
    # One head: every length up to sixteen KV tiles is searched with the same
    # two knobs.
    for num_kv in (1, 2, 3, 4, 15, 16):
        fragments = cute_flash.flash_autotune_fragments(
            64, num_kv, num_bh=1, dtype=torch.float16
        )
        assert "row_mma" in _active(fragments[FAMILY])
        assert _active(fragments[WARPS]) == (4, 2, 8)
        assert _active(fragments[TILE_M]) == (8, 16)
    # Beyond the bound (and for an unknown batch) the family leaves the
    # unattended surface and its knobs are pinned, as on a causal surface.
    for num_kv, num_bh in ((17, 1), (2, 16), (96, 64), (2, None)):
        large = cute_flash.flash_autotune_fragments(
            64, num_kv, num_bh=num_bh, dtype=torch.float16
        )
        assert "row_mma" not in _active(large[FAMILY])
        assert _active(large[WARPS]) == (4,)
        assert _active(large[TILE_M]) == (8,)
    causal = cute_flash.flash_autotune_fragments(
        64, 2, num_bh=1, dtype=torch.float16, is_causal=True
    )
    assert "row_mma" not in _active(causal[FAMILY])
    assert _active(causal[WARPS]) == (4,)
    assert _active(causal[TILE_M]) == (8,)
    # Requesting the family pins every dead knob's search to its default, on
    # any grid.
    for num_bh in (1, 64):
        override = cute_flash.flash_autotune_fragments(
            64,
            96,
            num_bh=num_bh,
            dtype=torch.float16,
            pipeline_family_override="row_mma",
        )
        assert _active(override[FAMILY]) == ("row_mma",)
        assert _active(override[WARPS]) == (4, 2, 8)
        assert _active(override[TILE_M]) == (8, 16)
        for key, fragment in override.items():
            if key in (FAMILY, WARPS, TILE_M):
                continue
            assert isinstance(fragment, EnumFragment)
            assert _active(fragment) == (fragment.default(),), key


def test_row_mma_chunking_and_shape_rules() -> None:
    chunk_keys = cute_flash_row_mma.row_mma_chunk_keys
    assert chunk_keys(64, 64) == 64
    assert chunk_keys(64, 32) == 32
    assert chunk_keys(64, 96) == 48
    assert chunk_keys(64, 16) == 16
    assert chunk_keys(64, 1024) == 64
    assert chunk_keys(128, 64) == 32
    assert chunk_keys(128, 48) == 16
    supported = cute_flash_row_mma.row_mma_shape_supported
    assert supported(seq=128, head_dim=64, row_warps=8, row_tile_m=16)
    assert supported(seq=256, head_dim=64, row_warps=2, row_tile_m=8)
    assert supported(seq=384, head_dim=64, row_warps=8, row_tile_m=8)
    assert not supported(seq=64, head_dim=64, row_warps=8, row_tile_m=8)
    assert not supported(seq=200, head_dim=64, row_warps=4, row_tile_m=8)
    # The K/V staging plan: double buffered down to the smallest chunk, one
    # stage as the fallback, all inside the shared-memory budget.
    plan = cute_flash_row_mma.row_mma_plan
    assert plan(seq=256, head_dim=64, row_warps=4, row_tile_m=8) == (64, 1)
    assert plan(seq=512, head_dim=64, row_warps=4, row_tile_m=8) == (64, 2)
    # Eight warps of 16 rows cannot double buffer 64-key chunks: the chunk
    # shrinks before the staging drops to one stage.
    assert plan(seq=4096, head_dim=64, row_warps=8, row_tile_m=16) == (32, 2)
    assert plan(seq=256, head_dim=128, row_warps=4, row_tile_m=8) == (32, 2)
    assert plan(seq=768, head_dim=128, row_warps=8, row_tile_m=8) == (16, 2)
    assert plan(seq=768, head_dim=128, row_warps=8, row_tile_m=16) == (16, 1)
    # A fused row epilogue's aux rows take one slot per epilogue thread and
    # pass (eight columns each): 128 threads x 16 B for XSA's bf16 V rows at
    # hd64 x 8 rows, two passes of 64 threads for two warps at hd128.
    aux_bytes = cute_flash_row_mma.row_mma_aux_smem_bytes
    reps = cute_flash_row_mma.row_mma_epilogue_reps
    assert reps(head_dim=64, row_tile_m=8, row_warps=4) == 1
    assert reps(head_dim=128, row_tile_m=8, row_warps=2) == 2
    assert reps(head_dim=128, row_tile_m=16, row_warps=8) == 1
    bf16, f32 = "cutlass.BFloat16", "cutlass.Float32"
    assert aux_bytes(head_dim=64, row_tile_m=8, row_warps=4, aux_dtypes=[bf16]) == 2048
    assert (
        aux_bytes(head_dim=128, row_tile_m=8, row_warps=2, aux_dtypes=[f32, bf16])
        == 2 * 64 * 8 * 6
    )
    assert plan(seq=256, head_dim=64, row_warps=4, row_tile_m=8, aux_bytes=2048) == (
        64,
        1,
    )
    # The widest configuration still fits four fp32 aux rows beside its
    # single-stage staging; a fifth does not.
    four = aux_bytes(head_dim=128, row_tile_m=16, row_warps=8, aux_dtypes=[f32] * 4)
    assert plan(seq=768, head_dim=128, row_warps=8, row_tile_m=16, aux_bytes=four) == (
        16,
        1,
    )
    five = aux_bytes(head_dim=128, row_tile_m=16, row_warps=8, aux_dtypes=[f32] * 5)
    assert (
        plan(seq=768, head_dim=128, row_warps=8, row_tile_m=16, aux_bytes=five) is None
    )
    for head_dim in (64, 128):
        for warps in (2, 4, 8):
            for rows in (8, 16):
                for seq in (128, 256, 384, 1024, 4096):
                    found = plan(
                        seq=seq, head_dim=head_dim, row_warps=warps, row_tile_m=rows
                    )
                    assert found is not None, (head_dim, warps, rows, seq)
                    assert (
                        cute_flash_row_mma.row_mma_smem_bytes(
                            head_dim=head_dim,
                            row_tile_m=rows,
                            row_warps=warps,
                            chunk=found[0],
                            stages=found[1],
                        )
                        <= cute_flash_row_mma.ROW_MMA_SMEM_BUDGET
                    )


def test_row_mma_seed_covers_every_small_grid_dense_surface() -> None:
    for head_dim, num_kv, dtype in ((64, 2, torch.bfloat16), (128, 4, torch.float16)):
        seeds = cute_flash.flash_attention_seed_configs(
            head_dim,
            num_kv,
            num_bh=4,
            dtype=dtype,
            standard_dense_output=True,
            device_sm_count=148,
        )
        row = [seed for seed in seeds if seed.config.get(FAMILY) == "row_mma"]
        assert len(row) == 1
        assert (row[0].config[WARPS], row[0].config[TILE_M]) == (4, 8)
    causal = cute_flash.flash_attention_seed_configs(
        64,
        2,
        num_bh=4,
        dtype=torch.bfloat16,
        is_causal=True,
        standard_causal_output=True,
    )
    assert not [seed for seed in causal if seed.config.get(FAMILY) == "row_mma"]
    # Larger grids seed nothing for the family: its cost grows with the
    # sequence while the tcgen05 tiles already fill the device.
    for head_dim, num_kv, num_bh in ((64, 8, 16), (128, 16, 32), (64, 128, 8)):
        large = cute_flash.flash_attention_seed_configs(
            head_dim,
            num_kv,
            num_bh=num_bh,
            dtype=torch.bfloat16,
            standard_dense_output=True,
            device_sm_count=148,
        )
        assert not [seed for seed in large if seed.config.get(FAMILY) == "row_mma"]
        assert all(seed.config[WARPS] == 4 for seed in large)


def test_row_mma_launcher_refuses_an_under_aligned_schema() -> None:
    from helion import exc
    from helion.runtime.cute import launcher

    plan: dict[str, object] = {
        "kind": "helion_flash_row_mma",
        "batch": 4,
        "seq": 256,
        "head_dim": 64,
        "row_warps": 4,
        "row_tile_m": 8,
        "total_tiles": 128,
        "q_idx": 0,
        "k_idx": 1,
        "v_idx": 2,
        "o_idx": 3,
        "lse_idx": 4,
    }
    aligned: tuple[object, ...] = (
        "tensor",
        "torch.bfloat16",
        3,
        (4, 256, 64),
        (16384, 64, 1),
    )
    schema = [aligned] * 4 + [("tensor", "torch.float32", 2, (4, 256), (256, 1))]
    body: list[str] = []
    launcher._append_cute_wrapper_plan(body, [], plan, schema_key=tuple(schema))
    assert body == [
        "    grid_x = cutlass.Int32(128)",
        "    grid_y = cutlass.Int32(1)",
        "    grid_z = cutlass.Int32(1)",
    ]
    # A base the schema proves only 8-byte aligned is refused for every tensor.
    for index in range(len(schema)):
        under = list(schema)
        under[index] = (*under[index], 8)
        with pytest.raises(exc.BackendUnsupported, match="16-byte-aligned"):
            launcher._append_cute_wrapper_plan([], [], plan, schema_key=tuple(under))
    body = []
    launcher._append_cute_wrapper_plan(body, [], plan)
    assert len(body) == 3
    # A fused row epilogue's aux rows are packets from their bases as well.
    fused = {**plan, "epi_aux_count": 1, "epi_aux0_idx": 5}
    fused_schema = [
        *schema,
        ("tensor", "torch.float32", 3, (4, 256, 64), (16384, 64, 1)),
    ]
    body = []
    launcher._append_cute_wrapper_plan(body, [], fused, schema_key=tuple(fused_schema))
    assert len(body) == 3
    fused_schema[5] = (*fused_schema[5], 8)
    with pytest.raises(exc.BackendUnsupported, match="16-byte-aligned"):
        launcher._append_cute_wrapper_plan(
            [], [], fused, schema_key=tuple(fused_schema)
        )
    # The launcher stages an under-aligned aux operand like q/k/v.
    kernel = type("Kernel", (), {"_helion_cute_wrapper_plans": [fused]})()
    staged = launcher._cute_staged_plan_operands(kernel)
    assert dict(staged) == {0: False, 1: False, 2: False, 3: True, 4: True, 5: False}
    # An aux binding proven only 8-byte aligned at launch is cloned to a
    # 16-byte base (an input: nothing is copied back); the rest pass through.
    operands = [torch.randn(4, 256, 64, dtype=torch.bfloat16) for _ in range(4)]
    operands.append(torch.zeros(4, 256, dtype=torch.float32))
    numel = 4 * 256 * 64
    aux = torch.randn(numel + 4, dtype=torch.float32)[2 : 2 + numel].view(4, 256, 64)
    assert aux.data_ptr() % 16 == 8
    staging = launcher._cute_stage_under_aligned_operands(kernel, (*operands, aux))
    assert staging is not None
    launch_args, copy_backs = staging
    assert all(a is b for a, b in zip(launch_args[:5], operands, strict=True))
    assert isinstance(launch_args[5], torch.Tensor)
    assert launch_args[5] is not aux and launch_args[5].data_ptr() % 16 == 0
    torch.testing.assert_close(launch_args[5], aux)
    assert copy_backs == []


# --------------------------------------------------------------------------
# Generated code
# --------------------------------------------------------------------------


def _code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    bound = kernel.bind(args)
    return bound.to_code(bound._normalized_config_copy(helion.Config(**config)))


@onlyBackends(["cute"])
def test_row_mma_renders_the_single_chunk_row_program() -> None:
    code = _code(_attention_kernel(), _qkv(1, 4, 256, 64, torch.bfloat16), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    assert "'row_warps': 4, 'row_tile_m': 8, 'total_tiles': 128" in code
    for name in ("q", "k", "v", "o", "lse"):
        assert f"'{name}_idx': " in code
    assert "block=(128, 1, 1)" in code
    # Every byte offset is Int64 from the head index on (seq * row_bytes =
    # 32768 for 256 rows of 128 bytes; 1024 for the fp32 LSE rows).
    assert "rm_bh_off = cutlass.Int64(rm_bh) * 32768" in code
    assert "rm_bh_off + cutlass.Int64(rm_row0) * 128" in code
    assert "rm_bh_off + cutlass.Int64(rm_key0) * 128" in code
    assert (
        "rm_bh_off + cutlass.Int64(rm_row0 + rm_erow) * 128 + cutlass.Int64(rm_ecol * 16)"
        in code
    )
    assert "cutlass.Int64(rm_bh) * 1024 + cutlass.Int64(rm_row0 + rm_erow) * 4" in code
    assert "rm_bh * 256" not in code
    runtime = "_helion_flash_rowmma"
    # Four warps x 64 keys x head 64: 16 QK^T and 16 PV mma.sync per warp,
    # 2 + 16 + 16 cp.async packets per lane, one P^T transpose per 8x8 tile.
    assert code.count(f"{runtime}.mma_bf16(") == 32
    assert code.count(f"{runtime}.cp_async_16(") == 34
    assert code.count(f"{runtime}.ldmatrix_x4(") == 18
    assert code.count(f"{runtime}.ldmatrix_x4_trans(") == 16
    assert code.count(f"{runtime}.movmatrix_trans(") == 8
    # Every MMA sits behind the cp.async waits: K (one group still pending)
    # before QK^T, V before PV; a single chunk needs no chunk loop.
    assert code.index("cp_async_wait_group(1)") < code.index(f"{runtime}.mma_bf16(")
    assert code.index("cp_async_wait_group(0)") < code.index(
        f"{runtime}.ldmatrix_x4_trans("
    )
    assert "cutlass.range(" not in code
    assert "cute.arch.barrier()" in code
    assert f"{runtime}.stg_v4_b32(" in code
    assert "cute.math.log2(rm_l_all)" in code
    assert (
        "tcgen05"
        not in code.split("_helion_cute_wrapper_plans")[0].split(
            "def _helion_attention"
        )[1]
    )


@onlyBackends(["cute"])
def test_row_mma_renders_two_octets_and_eight_warps() -> None:
    code = _code(
        _attention_kernel(),
        _qkv(1, 4, 256, 64, torch.float16),
        **_ROW,
        **{WARPS: 8, TILE_M: 16},
    )
    assert "'row_warps': 8, 'row_tile_m': 16, 'total_tiles': 64" in code
    assert "block=(256, 1, 1)" in code
    # Eight warps x 32 keys: 2 M-tiles x 4 k-steps x 2 octets QK^T MMAs plus
    # 4 d-tiles x 2 k-steps x 2 octets PV MMAs.
    assert code.count("_helion_flash_rowmma.mma_f16(") == 32
    assert code.count("_helion_flash_rowmma.pack_f16x2(") == 8 + 4


@onlyBackends(["cute"])
def test_row_mma_renders_the_chunk_loop_for_long_key_ranges() -> None:
    code = _code(_attention_kernel(), _qkv(1, 1, 1024, 64, torch.bfloat16), **_ROW)
    # 256 keys per warp stream as four 64-key chunks through two stages.
    assert "for rm_chunk in cutlass.range(1, 4):" in code
    assert "rm_o = cute.make_rmem_tensor((16,), cutlass.Float32)" in code
    assert "if rm_chunk + 1 < 4:" in code
    assert "cute.arch.exp2((rm_mo_0_a - rm_m_0_a) * rm_scale)" in code


@onlyBackends(["cute"])
def test_row_mma_renders_the_single_stage_chunk_loop() -> None:
    # 96 keys per warp of the 128-wide head with eight warps of 16 rows: six
    # 16-key chunks through one stage, refilled after each chunk's reads.
    code = _code(
        _attention_kernel(),
        _qkv(1, 3, 768, 128, torch.bfloat16),
        **_ROW,
        **{WARPS: 8, TILE_M: 16},
    )
    assert "for rm_chunk in cutlass.range(1, 6):" in code
    assert "if rm_chunk + 1 < 6:" not in code
    assert "rm_kv_off = cutlass.Int64(rm_chunk) * 4096" in code
    assert "cute.arch.alloc_smem(cutlass.BFloat16, 32768, alignment=128)" in code


@onlyBackends(["cute"])
def test_row_mma_renders_the_relu_epilogue() -> None:
    code = _code(_relu_kernel(), _qkv(1, 4, 256, 64, torch.bfloat16), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    assert "cute.arch.fmax(rm_acc_0 * rm_inv, rm_zero)" in code


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("shape", "dtype", "warps", "tile_m"),
    (
        ((1, 4, 256, 64), torch.bfloat16, 4, 8),
        ((1, 4, 256, 64), torch.float16, 8, 16),
        ((2, 2, 128, 64), torch.bfloat16, 2, 8),
        ((1, 2, 512, 64), torch.bfloat16, 4, 8),
        ((1, 2, 256, 128), torch.bfloat16, 4, 8),
        ((1, 2, 384, 128), torch.float16, 4, 16),
        ((1, 3, 768, 128), torch.bfloat16, 8, 16),
    ),
)
def test_row_mma_matches_reference(
    shape: tuple[int, int, int, int], dtype: torch.dtype, warps: int, tile_m: int
) -> None:
    q, k, v = _qkv(*shape, dtype)
    expected_out, expected_lse = _reference(q, k, v)
    code, (out, lse) = code_and_output(
        _attention_kernel(), (q, k, v), **_ROW, **{WARPS: warps, TILE_M: tile_m}
    )
    assert "'kind': 'helion_flash_row_mma'" in code
    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_row_mma_relu_output_matches_reference() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    expected_out, _ = _reference(q, k, v)
    code, out = code_and_output(_relu_kernel(), (q, k, v), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    torch.testing.assert_close(out, torch.relu(expected_out), atol=1e-2, rtol=1e-2)


@onlyBackends(["cute"])
@pytest.mark.parametrize("under_aligned", ("q", "k", "v"))
def test_row_mma_falls_back_to_tcgen05_for_an_under_aligned_base(
    under_aligned: str,
) -> None:
    kernel = _attention_kernel()
    tensors = dict(zip("qkv", _qkv(1, 4, 256, 64, torch.bfloat16), strict=True))
    numel = 1 * 4 * 256 * 64
    torch.manual_seed(7)
    buffer = torch.randn(numel + 64, dtype=torch.bfloat16, device=DEVICE)
    # Four bf16 elements into the buffer leave an 8-byte-aligned contiguous
    # view: the row programs' 16-byte packets cannot start there, so that
    # binding resolves to the tcgen05 families (whose launch stages the view
    # through an aligned copy; see test_cute_flash_alignment.py). Eight
    # elements restore the 16-byte base and the row programs.
    for offset, kind in ((4, "helion_flash"), (8, "helion_flash_row_mma")):
        view = buffer[offset : offset + numel].view(1, 4, 256, 64)
        assert view.is_contiguous()
        assert view.data_ptr() % 16 == (offset * 2) % 16
        args = {**tensors, under_aligned: view}
        q, k, v = args["q"], args["k"], args["v"]
        expected_out, expected_lse = _reference(q, k, v)
        code, (out, lse) = code_and_output(kernel, (q, k, v), **_ROW)
        assert f"'kind': '{kind}'" in code
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_row_mma_byte_offsets_pass_two_gib() -> None:
    # 2**30 + 8192 elements per tensor: the last heads' bases lie past 2**31
    # bytes while the element count stays inside the flash surface's bound,
    # so every offset must be formed in Int64.
    batch, heads, seq, head_dim = 1, 131073, 128, 64
    numel = batch * heads * seq * head_dim
    needed = 4 * numel * 2 + batch * heads * seq * 4 + (1 << 30)
    free, _total = torch.cuda.mem_get_info()
    if free < needed:
        pytest.skip(f"needs {needed >> 30} GiB of free device memory")
    torch.manual_seed(8)
    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    code, (out, lse) = code_and_output(_attention_kernel(), (q, k, v), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    for rows in (slice(0, 4), slice(65534, 65540), slice(heads - 4, heads)):
        expected_out, expected_lse = _reference(q[:, rows], k[:, rows], v[:, rows])
        torch.testing.assert_close(out[:, rows], expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse[:, rows], expected_lse, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
def test_row_mma_is_deterministic_and_does_not_alias_the_output() -> None:
    kernel = _attention_kernel()
    outs = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        q, k, v = (
            torch.randn(1, 4, 256, 64, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        )
        _, (out, lse) = code_and_output(kernel, (q, k, v), **_ROW)
        _, (out2, lse2) = code_and_output(kernel, (q, k, v), **_ROW)
        # The fixed-order combine makes repeated launches bit-identical.
        assert torch.equal(out, out2)
        assert torch.equal(lse, lse2)
        expected_out, _ = _reference(q, k, v)
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        outs.append(out)
    assert not torch.equal(outs[0], outs[1])


@onlyBackends(["cute"])
def test_row_mma_seed_survives_normalization_and_seed_deduplication() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    bound = _attention_kernel().bind((q, k, v))
    spec = bound.config_spec
    row = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get(FAMILY) == "row_mma"
    ]
    assert len(row) == 1
    (seed,) = row
    config_gen = spec.create_config_generation()
    _flat, normalized = config_gen.canonicalize_flat(config_gen.flatten(seed))
    assert normalized.config[FAMILY] == "row_mma"
    assert (normalized.config[WARPS], normalized.config[TILE_M]) == (4, 8)
    assert "'kind': 'helion_flash_row_mma'" in bound.to_code(normalized)
    survivors = [
        config
        for _flat, config in config_gen.seed_flat_config_pairs()
        if config.config[FAMILY] == "row_mma"
    ]
    assert survivors == [normalized]
    assert normalized in config_gen.random_population(100)


# --------------------------------------------------------------------------
# Fused row epilogues in the combine epilogue
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
def test_row_mma_renders_the_fused_row_epilogue() -> None:
    code = _code(_xsa_kernel(), _qkv(1, 4, 256, 64, torch.bfloat16), **_ROW)
    runtime = "_helion_flash_rowmma"
    assert "'kind': 'helion_flash_row_mma'" in code
    assert "'epi_aux_count': 1" in code and "'epi_aux0_idx': 2" in code
    assert "'lse_idx'" not in code
    # Every epilogue thread stages its own 16-byte packet of V[row] in group 0
    # (one slot per thread, before the Q tile), and nothing else changes in
    # the body: 34 operand packets plus the aux packet.
    assert (
        "rm_s_aux0 = cute.arch.alloc_smem(cutlass.BFloat16, 1024, alignment=128)"
        in code
    )
    assert (
        "rm_aux0_g = v_view.iterator.toint() + cutlass.Int64(rm_bh) * 32768 "
        "+ cutlass.Int64(rm_row0 + rm_aerow) * 128 + cutlass.Int64(rm_aecol * 16)"
    ) in code
    aux_issue = f"{runtime}.cp_async_16(rm_aux0 + rm_ae * 16 + 0, rm_aux0_g + 0)"
    assert code.count(aux_issue) == 1
    assert code.index(aux_issue) < code.index("rm_q_tile + cutlass.Int64(")
    assert code.index(aux_issue) < code.index("cp_async_commit_group()")
    assert code.count(f"{runtime}.cp_async_16(") == 35
    # The program: the value norm, the projection and the residual as three
    # passes over this thread's eight columns (element pairs), each head_dim
    # reduction completed over the eight lanes sharing the row, the row values
    # (sqrt, eps clamp, reciprocal) recomputed by every lane.
    assert code.count("for rm_ep_j in cutlass.range_constexpr(0, 8, 2):") == 3
    assert f"rm_ep_v4 = {runtime}.lane_group_sum(rm_ep_v4, 8)" in code
    assert f"rm_ep_v10 = {runtime}.lane_group_sum(rm_ep_v10, 8)" in code
    assert "rm_ep_v5 = cute.math.sqrt(rm_ep_v4)" in code
    assert "cute.math.max(rm_ep_v5, cutlass.Float32(eps), propagate_nan=True)" in code
    assert "cute.arch.fma_packed_f32x2((-rm_ep_v10, -rm_ep_v10)" in code
    assert code.count("rm_ep_o0[0] = rm_acc_0 * rm_inv") == 2
    # The program consumes the fixed-order cross-warp combine (warp 0's partial
    # first, then warps 1-3) and nothing moves it: every combine step precedes
    # the first read of the normalized columns.
    combine = [code.index(f"rm_acc_0 = rm_acc_0 + rm_al_{w} * rm_v0") for w in range(4)]
    assert combine == sorted(combine)
    assert combine[-1] < code.index("rm_ep_o0[0] = rm_acc_0 * rm_inv")
    assert (
        "cute.autovec_copy(cute.make_tensor(cute.make_ptr(cutlass.BFloat16, "
        "rm_aux0 + rm_e * 16, cute.AddressSpace.smem, assumed_align=16), "
        "cute.make_layout((8,))), rm_ep_a0_0)"
    ) in code
    # The identity store is replaced by the program's output packet.
    assert "rm_out_0" not in code
    assert code.count(f"{runtime}.stg_v4_b32(") == 1
    assert f"{runtime}.pack_bf16x2(rm_ep_out0[6], rm_ep_out0[7])" in code
    assert f"{runtime}.stg_f32(" not in code


@onlyBackends(["cute"])
def test_row_mma_fused_row_epilogue_renders_sixteen_lane_groups_and_passes() -> None:
    # hd128: sixteen 16-byte packets per row, so the reductions complete over
    # sixteen lanes; two warps leave 64 threads for 128 (row, packet) pairs,
    # which the epilogue walks in two passes with one aux slot each.
    code = _code(
        _xsa_kernel(), _qkv(1, 2, 256, 128, torch.bfloat16), **_ROW, **{WARPS: 2}
    )
    runtime = "_helion_flash_rowmma"
    assert "'row_warps': 2, 'row_tile_m': 8" in code
    assert (
        "rm_s_aux0 = cute.arch.alloc_smem(cutlass.BFloat16, 1024, alignment=128)"
        in code
    )
    assert "rm_ae = 0 + rm_tidx" in code and "rm_ae = 64 + rm_tidx" in code
    assert (
        code.count(f"{runtime}.cp_async_16(rm_aux0 + rm_ae * 16 + 0, rm_aux0_g + 0)")
        == 2
    )
    assert code.count("lane_group_sum(rm_ep_v4, 16)") == 2
    assert code.count("lane_group_sum(rm_ep_v10, 16)") == 2
    assert code.count(f"{runtime}.stg_v4_b32(") == 2


@onlyBackends(["cute"])
def test_row_mma_fused_row_epilogue_renders_fp32_aux_rows_and_max_reductions() -> None:
    # An fp32 aux row needs two packets per thread (32 bytes of eight columns).
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    g = torch.randn(1, 4, 256, 64, dtype=torch.float32, device=DEVICE)
    code = _code(_gated_by_fp32_rows, (q, k, v, g), **_ROW)
    runtime = "_helion_flash_rowmma"
    assert "'kind': 'helion_flash_row_mma'" in code
    assert "'epi_aux_count': 1" in code and "'epi_aux0_idx': " in code
    assert (
        "rm_s_aux0 = cute.arch.alloc_smem(cutlass.Float32, 1024, alignment=128)" in code
    )
    assert (
        "cutlass.Int64(rm_bh) * 65536 + cutlass.Int64(rm_row0 + rm_aerow) * 256" in code
    )
    assert f"{runtime}.cp_async_16(rm_aux0 + rm_ae * 32 + 0, rm_aux0_g + 0)" in code
    assert f"{runtime}.cp_async_16(rm_aux0 + rm_ae * 32 + 16, rm_aux0_g + 16)" in code
    assert code.count(f"{runtime}.cp_async_16(") == 36
    assert "cute.make_ptr(cutlass.Float32, rm_aux0 + rm_e * 32," in code
    assert "lane_group_" not in code  # no reduction: a single pointwise pass
    assert code.count("for rm_ep_j in cutlass.range_constexpr(0, 8, 2):") == 1
    # A max reduction completes with the NaN-propagating lane-group max.
    code = _code(_amax_normalized_rows, (q, k, v, 1e-6), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    # No aux rows: the plan carries no aux keys and no slot is staged.
    assert "epi_aux" not in code and "rm_s_aux" not in code
    assert f"{runtime}.lane_group_max(rm_ep_v2, 8)" in code
    assert "cute.math.max(rm_ep_v2_0_p0, rm_ep_v1_p0, propagate_nan=True)" in code


def _ws_overlap_output(kernel: helion.Kernel, args: tuple[object, ...]) -> torch.Tensor:
    _code_, out = code_and_output(
        kernel,
        args,
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family="ws_overlap",
        cute_flash_persistent=False,
        cute_flash_epi_stg=True,
    )
    return out


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("shape", "dtype", "warps", "tile_m"),
    (
        ((1, 4, 256, 64), torch.bfloat16, 4, 8),
        ((1, 4, 256, 64), torch.float16, 8, 16),
        ((1, 2, 256, 128), torch.bfloat16, 2, 8),
        ((1, 1, 1024, 64), torch.bfloat16, 4, 8),
    ),
)
def test_row_mma_fused_row_epilogue_matches_the_tcgen05_body_and_reference(
    shape: tuple[int, int, int, int], dtype: torch.dtype, warps: int, tile_m: int
) -> None:
    q, k, v = _qkv(*shape, dtype)
    kernel = _xsa_kernel()
    code, out = code_and_output(
        kernel, (q, k, v, 1e-6), **_ROW, **{WARPS: warps, TILE_M: tile_m}
    )
    assert "'kind': 'helion_flash_row_mma'" in code
    assert "lane_group_sum(" in code
    expected = _ref_xsa(q, k, v)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=1e-2)
    # The same program on the 128-row ws_overlap body: both evaluate exact fp32
    # softmax statistics and the fp32 program, so the outputs differ by at most
    # the output rounding.
    tcgen05 = _ws_overlap_output(kernel, (q, k, v, 1e-6))
    torch.testing.assert_close(out.float(), tcgen05.float(), atol=1.6e-2, rtol=1e-2)
    _code_, again = code_and_output(
        kernel, (q, k, v, 1e-6), **_ROW, **{WARPS: warps, TILE_M: tile_m}
    )
    assert torch.equal(out, again)


@onlyBackends(["cute"])
def test_row_mma_fused_row_epilogue_keeps_a_zero_value_row_finite() -> None:
    # ``F.normalize`` semantics: a zero V row divides by eps, not by zero.
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    v[..., 0, :] = 0.0
    v[..., 9, :] = 0.0
    code, out = code_and_output(_xsa_kernel(), (q, k, v, 1e-6), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(), _ref_xsa(q, k, v).float(), atol=2e-2, rtol=1e-2
    )


@onlyBackends(["cute"])
def test_row_mma_fused_row_epilogue_fp32_aux_rows_match_reference() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    torch.manual_seed(3)
    buffer = torch.randn(1 * 4 * 256 * 64 + 64, dtype=torch.float32, device=DEVICE)
    for offset, kind in ((8, "helion_flash_row_mma"), (2, "helion_flash")):
        # Two fp32 elements into the buffer leave an 8-byte-aligned gate view:
        # that binding resolves to the tcgen05 families (whose launch stages
        # the view), eight elements keep the row programs.
        g = buffer[offset : offset + 1 * 4 * 256 * 64].view(1, 4, 256, 64)
        assert g.data_ptr() % 16 == (offset * 4) % 16
        code, out = code_and_output(_gated_by_fp32_rows, (q, k, v, g), **_ROW)
        assert f"'kind': '{kind}'" in code
        expected = (_attention_fp32(q, k, v) * g).to(q.dtype)
        torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=1e-2)


@onlyBackends(["cute"])
def test_row_mma_fused_row_epilogue_max_reduction_matches_reference() -> None:
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    code, out = code_and_output(_amax_normalized_rows, (q, k, v, 1e-6), **_ROW)
    assert "'kind': 'helion_flash_row_mma'" in code
    y = _attention_fp32(q, k, v)
    expected = (y / torch.clamp(y.abs().amax(dim=-1, keepdim=True), min=1e-6)).to(
        q.dtype
    )
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=1e-2)
    _code_, again = code_and_output(_amax_normalized_rows, (q, k, v, 1e-6), **_ROW)
    assert torch.equal(out, again)


@onlyBackends(["cute"])
def test_row_mma_joins_the_fused_row_surface_inside_the_search_bound() -> None:
    # XSA at (1, 4, 256, 64): 128 row programs, so the family is searched with
    # its two knobs and seeded; the seed survives normalization and dedup.
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    bound = _xsa_kernel().bind((q, k, v, 1e-6))
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_has_row_epilogue
    assert not spec._cute_flash_plain_row_body
    assert not spec._cute_flash_has_score_modifiers
    fragments = spec._cute_flash_autotune_fragments()
    assert "row_mma" in _active(fragments[FAMILY])
    assert _active(fragments[WARPS]) == (4, 2, 8)
    assert _active(fragments[TILE_M]) == (8, 16)
    row = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get(FAMILY) == "row_mma"
    ]
    assert len(row) == 1
    (seed,) = row
    config_gen = spec.create_config_generation()
    config_gen.validate_flash_structural_coverage()
    _flat, normalized = config_gen.canonicalize_flat(config_gen.flatten(seed))
    assert normalized.config[FAMILY] == "row_mma"
    assert (normalized.config[WARPS], normalized.config[TILE_M]) == (4, 8)
    assert "'kind': 'helion_flash_row_mma'" in bound.to_code(normalized)
    survivors = [
        config
        for _flat, config in config_gen.seed_flat_config_pairs()
        if config.config[FAMILY] == "row_mma"
    ]
    assert survivors == [normalized]
    assert normalized in config_gen.random_population(100)
    # The large XSA grids (the campaign's shapes 1 and 2: 2048 and 8192 row
    # programs) keep today's surface: no row programs, pinned knobs, no seed.
    for shape in ((2, 8, 1024, 64), (2, 16, 2048, 128)):
        q, k, v = _qkv(*shape, torch.bfloat16)
        large = _xsa_kernel().bind((q, k, v, 1e-6)).config_spec
        assert large._cute_flash_has_row_epilogue
        fragments = large._cute_flash_autotune_fragments()
        assert "row_mma" not in _active(fragments[FAMILY])
        assert _active(fragments[WARPS]) == (4,)
        assert _active(fragments[TILE_M]) == (8,)
        assert not [
            seed
            for seed in large.compiler_seed_configs
            if seed.config.get(FAMILY) == "row_mma"
        ]


@onlyBackends(["cute"])
def test_row_mma_is_not_offered_to_a_fused_row_epilogue_over_biased_scores() -> None:
    # A score modifier next to the row epilogue: the flash surface records it
    # and the row programs stay out (plain_row_body alone cannot tell a
    # modifier from a fused epilogue).
    q, k, v = _qkv(1, 4, 256, 64, torch.bfloat16)
    bias = torch.randn(1, 4, 256, 256, dtype=torch.bfloat16, device=DEVICE)
    bound = _biased_xsa_like.bind((q, k, v, bias, 1e-6))
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_has_row_epilogue
    assert spec._cute_flash_has_score_modifiers
    fragments = spec._cute_flash_autotune_fragments()
    assert "row_mma" not in _active(fragments[FAMILY])
    assert _active(fragments[WARPS]) == (4,)
    with pytest.raises(helion.exc.InvalidConfig, match="row_mma"):
        bound._normalized_config_copy(
            helion.Config(
                block_sizes=[1, 128, 128], cute_flash_pipeline_family="row_mma"
            )
        )
    normalized = bound._normalized_config_copy(
        helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="ws_overlap",
            cute_flash_persistent=False,
        )
    )
    assert "'kind': 'helion_flash'" in bound.to_code(normalized)
