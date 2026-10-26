"""The per-chunk staged-P release (``cute_flash_p_chunk_arrive``).

With the split release the softmax warpgroup publishes P in two arrivals (the
first three 32-column chunks on ``pfor``, the last on ``pfor2``) and the MMA
warp's PTX PV stream waits once, before the last K-chunk quarter. The per-chunk
release publishes every chunk as it is stored (``pfor``, ``pforc0``,
``pforc1``, ``pfor2``) and the PV stream waits before each K-chunk quarter, so
three quarters of PV run under the exp pass. Same stores, same MMA order: the
outputs are bitwise identical to the split release.
"""

from __future__ import annotations

import ast
import math
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.flash_schedule import FlashScheduleError
from helion._compiler.cute.flash_schedule import FlashScheduleSpec
from helion._compiler.cute.flash_schedule import FlashSyncScope
from helion._compiler.cute.flash_schedule import build_fa4_schedule
from helion._compiler.cute.flash_schedule import verify_flash_schedule
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_P_CHUNK_ARRIVE_KEY

# The (2, 16, 2048, 128) bf16 campaign winner (fa4 one-CTA, persistent, chunked
# softmax, 16/6 schedule, 8x2 packet), reduced to the flash keys that select
# its body.
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
# The earlier two-CTA winner of the same shape.
_WINNER_2CTA: dict[str, object] = {
    **_WINNER,
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_e2e_schedule": "8/2",
    "cute_flash_e2e_offset": 4,
    "cute_flash_e2e_offset0": 5,
    "cute_flash_exp2_packet": "1x1",
    "cute_flash_first_load_order": 3,
    "cute_flash_epi_tma": False,
    "cute_flash_epi_stg": True,
    "cute_flash_rescale_threshold": 8.0,
    "cute_flash_softmax_regs": 184,
    "cute_flash_corr_regs": 80,
    "cute_flash_other_regs": 64,
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


def _fragments(
    head_dim: int = 128, *, dtype: torch.dtype = torch.bfloat16, is_causal: bool
) -> dict[str, object]:
    return cute_flash.flash_autotune_fragments(
        head_dim,
        16,
        num_bh=32,
        dtype=dtype,
        is_causal=is_causal,
        standard_dense_output=False,
        standard_causal_output=False,
        plain_row_body=True,
    )


def _resolve(
    config: dict[str, object], *, head_dim: int = 128, is_causal: bool = False
) -> cute_flash.FlashAttentionConfig:
    with patch.dict(os.environ, {}, clear=True):
        return cast(
            "cute_flash.FlashAttentionConfig",
            cute_flash.resolve_flash_config(
                head_dim,
                16,
                config,
                dtype=torch.bfloat16 if head_dim == 128 else torch.float16,
                num_bh=32,
                is_causal=is_causal,
                standard_dense_output=False,
                standard_causal_output=False,
                plain_row_body=True,
            ),
        )


def _emit(config: dict[str, object]) -> str:
    cfg = _resolve(config)
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=128,
        num_kv=16,
        sequence_extent=2048,
        num_bh=32,
        total_tiles=128,
        cfg=cfg,
        has_lse=True,
        io_dtype="cutlass.BFloat16",
        score_plan=dense_score_plan(128),
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


# --------------------------------------------------------------------------
# surface and resolver
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head_dim", "dtype"),
    ((128, torch.bfloat16), (64, torch.float16)),
    ids=("hd128_bf16", "hd64_fp16"),
)
def test_dense_fa4_surface_searches_the_per_chunk_release(
    head_dim: int, dtype: torch.dtype
) -> None:
    fragment = cast("object", _fragments(head_dim, dtype=dtype, is_causal=False)[KEY])
    assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]
    assert tuple(fragment.search_choices) == (False, True)  # pyrefly: ignore[missing-attribute]


def test_causal_surface_keeps_the_split_release() -> None:
    """The causal chunked passes keep the 3/4 + 1/4 split: the knob is legal
    (configs round-trip) but not searched, and the resolver clears it."""
    fragment = cast("object", _fragments(is_causal=True)[KEY])
    assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]
    assert tuple(fragment.search_choices) == (False,)  # pyrefly: ignore[missing-attribute]
    assert not _resolve({**_WINNER, KEY: True}, is_causal=True).p_chunk_arrive


@pytest.mark.parametrize("base", (_WINNER, _WINNER_2CTA), ids=("one_cta", "two_cta"))
def test_resolver_keeps_the_per_chunk_release_on_the_dense_winners(
    base: dict[str, object],
) -> None:
    assert not _resolve(base).p_chunk_arrive
    cfg = _resolve({**base, KEY: True})
    assert cfg.p_chunk_arrive
    assert cfg.split_p_arrive and cfg.mma_ptx and cfg.softmax_disc


@pytest.mark.parametrize(
    "override",
    (
        {"cute_flash_split_p_arrive": False},
        {"cute_flash_p_store_rep": 32},
        {"cute_flash_s_load_rep": 16},
        {"cute_flash_softmax_disc": False},
        {"cute_flash_pipeline_family": "ws_overlap"},
    ),
    ids=("no_split", "rep32", "sload16", "whole_row_body", "ws_overlap"),
)
def test_resolver_clears_the_per_chunk_release_off_its_body(
    override: dict[str, object],
) -> None:
    """The release is written for the dense chunked pass with Rep16 P stores
    (four 32-column chunks) and the PTX PV stream that carries the waits."""
    assert not _resolve({**_WINNER, KEY: True, **override}).p_chunk_arrive


def test_resolver_clears_the_per_chunk_release_without_the_ptx_stream() -> None:
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}, clear=True):
        cfg = cast(
            "cute_flash.FlashAttentionConfig",
            cute_flash.resolve_flash_config(
                128,
                16,
                {**_WINNER, KEY: True},
                dtype=torch.bfloat16,
                num_bh=32,
                is_causal=False,
                standard_dense_output=False,
                standard_causal_output=False,
                plain_row_body=True,
            ),
        )
    assert not cfg.mma_ptx
    assert not cfg.p_chunk_arrive


# --------------------------------------------------------------------------
# generated code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("base", (_WINNER, _WINNER_2CTA), ids=("one_cta", "two_cta"))
def test_codegen_emits_the_chunk_barriers_and_the_four_wait_streams(
    base: dict[str, object],
) -> None:
    code = _emit({**base, KEY: True})
    count = "256" if base is _WINNER_2CTA else "128"
    assert "flash_pforc_ptr = storage.pforc_mbar.data_ptr()" in code
    assert f"cute.arch.mbarrier_init(flash_pforc_ptr + 2 * flash_st, {count})" in code
    assert (
        f"cute.arch.mbarrier_init(flash_pforc_ptr + 2 * flash_st + 1, {count})" in code
    )
    # Two PV streams per Q slot (steady loop and epilogue), each waiting on its
    # slot's pforc pair before the middle quarters and on pfor2 before the last.
    pv_calls = [
        line for line in code.splitlines() if "gemm_ptx_precomputed_pv_ts(" in line
    ]
    assert len(pv_calls) == 4
    for line in pv_calls:
        if "flash_o0_addr" in line:
            assert "mbar_ptr=flash_pfor2_ptr + 0" in line
            assert "mbar_mid_ptrs=(flash_pforc_ptr + 0, flash_pforc_ptr + 1)" in line
        else:
            assert "mbar_ptr=flash_pfor2_ptr + 1" in line
            assert "mbar_mid_ptrs=(flash_pforc_ptr + 2, flash_pforc_ptr + 3)" in line
    # The chunked exp pass of each slot publishes through its pforc pair.
    pass_calls = [
        line for line in code.splitlines() if "fa4_disc_exp_convert_store_pipe(" in line
    ]
    assert len(pass_calls) == 2
    assert (
        sum("pforc_ptr_stage=flash_pforc_ptr + 0)" in line for line in pass_calls) == 1
    )
    assert (
        sum("pforc_ptr_stage=flash_pforc_ptr + 2)" in line for line in pass_calls) == 1
    )


def test_codegen_without_the_knob_keeps_the_split_release() -> None:
    code = _emit(_WINNER)
    assert "pforc" not in code
    assert "mbar_mid_ptrs" not in code
    assert code.count("mbar_ptr=flash_pfor2_ptr + 0") == 2


def test_serial_pass_also_publishes_per_chunk() -> None:
    code = _emit({**_WINNER, KEY: True, "cute_flash_disc_pipe": 1})
    pass_calls = [
        line for line in code.splitlines() if "fa4_disc_exp_convert_store(" in line
    ]
    assert len(pass_calls) == 2
    assert all("pforc_ptr_stage=flash_pforc_ptr + " in line for line in pass_calls)


# --------------------------------------------------------------------------
# schedule model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cooperative", (False, True), ids=("one_cta", "two_cta"))
def test_schedule_models_one_barrier_per_chunk(cooperative: bool) -> None:
    spec = FlashScheduleSpec(
        head_dim=128,
        kv_depth=3,
        cta_count=2 if cooperative else 1,
        multicast_kv=cooperative,
        cooperative_mma=cooperative,
        persistent=True,
        kv_iterations=16,
        split_p_arrive=True,
        p_chunk_arrive=True,
    )
    schedule = build_fa4_schedule(spec)
    verify_flash_schedule(schedule)
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    expected = 256 if cooperative else 128
    scope = FlashSyncScope.CLUSTER_LEADER if cooperative else FlashSyncScope.CTA
    for slot in range(2):
        for index in range(2):
            barrier = barriers[f"pforc{index}_q{slot}"]
            assert barrier.expected_arrivals == expected
            assert barrier.scope is scope
            sources = {
                edge.source
                for edge in schedule.edges
                if edge.barrier == f"pforc{index}_q{slot}"
            }
            assert sources and all("softmax" in source for source in sources)
    cycles = {cycle.barrier: cycle for cycle in schedule.phase_cycles}
    assert cycles["pforc0_q0"].phases == cycles["pfor2_q0"].phases


def test_schedule_rejects_per_chunk_release_without_the_split() -> None:
    with pytest.raises(FlashScheduleError, match="per-chunk"):
        build_fa4_schedule(
            FlashScheduleSpec(
                head_dim=128, kv_depth=3, split_p_arrive=False, p_chunk_arrive=True
            )
        )


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


def _run(
    config: dict[str, object], *args: torch.Tensor, launches: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = helion.kernel(
        _attention_with_lse.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config(**config),
    )
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
    ("base", "persistent", "shape", "dtype"),
    (
        (_WINNER, True, (2, 16, 2048, 128), torch.bfloat16),
        (_WINNER_2CTA, True, (2, 16, 2048, 128), torch.bfloat16),
        (_WINNER, False, (1, 2, 1024, 128), torch.bfloat16),
        (
            {**_WINNER, "cute_flash_disc_pipe": 1},
            True,
            (1, 2, 4096, 128),
            torch.bfloat16,
        ),
        (_WINNER, True, (1, 4, 2048, 64), torch.float16),
    ),
    ids=(
        "one_cta_persistent",
        "two_cta_persistent",
        "one_cta_flat",
        "serial_pass",
        "hd64_fp16",
    ),
)
def test_per_chunk_release_matches_the_split_release_bitwise(
    base: dict[str, object],
    persistent: bool,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> None:
    """Same P stores, same MMA order: the output and LSE are identical; the
    repeated launches cover the barrier parity carried across KV steps and
    work items."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(*shape, dtype=dtype, device=DEVICE) for _ in range(3))
    config = {**base, "cute_flash_persistent": persistent}
    out_ref, lse_ref = _reference(q, k, v)
    out_split, lse_split = _run({**config, KEY: False}, q, k, v)
    out_chunk, lse_chunk = _run({**config, KEY: True}, q, k, v, launches=8)
    assert torch.equal(out_split, out_chunk)
    assert torch.equal(lse_split, lse_chunk)
    assert (out_chunk.float() - out_ref.float()).abs().max().item() < 1e-2
    assert (lse_chunk - lse_ref).abs().max().item() < 1e-3
