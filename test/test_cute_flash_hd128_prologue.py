"""hd128 dense FA4 levers: the exp2 packet surface with an LSE output (the
plain batched packets; the degree-2 polynomial packet stays with the standard
dense output), the QK0-before-Q1 MMA prologue and the staged first-load order.

The kernel here mirrors ``examples.attention.attention`` (row-plain body with a
base-2 LSE store), i.e. the surface the attention example autotunes on.
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
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

_DEG2_PACKET = "deg2_16x6"
_PLAIN_PACKETS = ("1x1", "4x1", "4x2", "8x1", "8x2")

# The (2, 16, 2048, 128) bf16 campaign winner (fa4_2cta, persistent, chunked
# softmax, 8/2 schedule), reduced to the flash keys that select its body.
_WINNER: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_kv_tile_n": 128,
    "cute_flash_s_stage": 2,
    "cute_flash_kv_stage": 3,
    "cute_flash_persistent": True,
    "cute_flash_e2e_schedule": "8/2",
    "cute_flash_e2e_offset": 4,
    "cute_flash_e2e_offset0": 5,
    "cute_flash_exp2_packet": "1x1",
    "cute_flash_mma_interleave": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_softmax_disc": True,
    "cute_flash_disc_pipe": 2,
    "cute_flash_split_p_arrive": True,
    "cute_flash_p_store_rep": 16,
    "cute_flash_s_load_rep": 32,
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


def _fragments(*, is_causal: bool, plain_row_body: bool = True) -> dict[str, object]:
    return cute_flash.flash_autotune_fragments(
        128,
        16,
        num_bh=32,
        dtype=torch.bfloat16,
        is_causal=is_causal,
        standard_dense_output=False,
        standard_causal_output=False,
        plain_row_body=plain_row_body,
    )


def _resolve(config: dict[str, object], *, is_causal: bool = False) -> object:
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.resolve_flash_config(
            128,
            16,
            config,
            dtype=torch.bfloat16,
            num_bh=32,
            is_causal=is_causal,
            standard_dense_output=False,
            standard_causal_output=False,
            plain_row_body=True,
        )


def _emit(config: dict[str, object], *, has_lse: bool = True) -> str:
    cfg = _resolve(config)
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=128,
        num_kv=16,
        sequence_extent=2048,
        num_bh=32,
        total_tiles=128,
        cfg=cast("cute_flash.FlashAttentionConfig", cfg),
        has_lse=has_lse,
        io_dtype="cutlass.BFloat16",
        score_plan=dense_score_plan(128),
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


# --------------------------------------------------------------------------
# exp2 packet surface at hd128
# --------------------------------------------------------------------------


def test_dense_hd128_lse_surface_offers_the_plain_packets_only() -> None:
    """The dense hd128 surface with an LSE output searches the plain batched
    packets (like hd64) and not the degree-2 compound packet: the polynomial
    pass is held to the bf16 output's rounding floor, and an fp32 LSE output
    exposes the emulated row sum (LSE error 1e-4 -> 2.5e-3 on peaky rows)."""
    packet = cast(
        "object", _fragments(is_causal=False)[cute_flash.FLASH_EXP2_PACKET_KEY]
    )
    choices = tuple(packet.choices)  # pyrefly: ignore[missing-attribute]
    assert set(choices) == set(_PLAIN_PACKETS)
    assert packet.search_choices is None or set(choices) == set(  # pyrefly: ignore[missing-attribute]
        packet.search_choices  # pyrefly: ignore[missing-attribute]
    )


def test_dense_hd128_standard_output_surface_keeps_the_degree2_packet() -> None:
    """The standard dense output (no LSE store) keeps its measured packet."""
    packet = cast(
        "object",
        cute_flash.flash_autotune_fragments(
            128,
            16,
            num_bh=32,
            dtype=torch.bfloat16,
            is_causal=False,
            standard_dense_output=True,
            standard_causal_output=False,
            plain_row_body=True,
        )[cute_flash.FLASH_EXP2_PACKET_KEY],
    )
    assert _DEG2_PACKET in tuple(packet.choices)  # pyrefly: ignore[missing-attribute]


def test_causal_hd128_surface_keeps_the_neutral_packet_only() -> None:
    packet = cast(
        "object", _fragments(is_causal=True)[cute_flash.FLASH_EXP2_PACKET_KEY]
    )
    choices = tuple(packet.choices)  # pyrefly: ignore[missing-attribute]
    assert "1x1" in choices
    for plain in _PLAIN_PACKETS[1:]:
        assert plain not in choices
    assert _DEG2_PACKET not in choices


def test_modified_row_body_keeps_the_neutral_hd128_packet() -> None:
    """Score modifiers (bias, alibi) do not get the degree-2 packet: it is
    measured on the plain row body only."""
    packet = cast(
        "object",
        _fragments(is_causal=False, plain_row_body=False)[
            cute_flash.FLASH_EXP2_PACKET_KEY
        ],
    )
    assert _DEG2_PACKET not in tuple(packet.choices)  # pyrefly: ignore[missing-attribute]


@pytest.mark.parametrize("packet", _PLAIN_PACKETS)
def test_dense_hd128_resolver_keeps_the_requested_plain_packet(packet: str) -> None:
    cfg = _resolve({**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: packet})
    assert cfg.exp2_packet == packet  # pyrefly: ignore[missing-attribute]
    assert cfg.e2e_schedule == "8/2"  # pyrefly: ignore[missing-attribute]


def test_dense_hd128_resolver_canonicalizes_the_degree2_packet_away_on_the_lse_body() -> (
    None
):
    cfg = _resolve({**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: _DEG2_PACKET})
    assert cfg.exp2_packet == "1x1"  # pyrefly: ignore[missing-attribute]
    assert cfg.e2e_schedule == "8/2"  # pyrefly: ignore[missing-attribute]
    with patch.dict(os.environ, {}, clear=True):
        standard = cute_flash.resolve_flash_config(
            128,
            16,
            {**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: _DEG2_PACKET},
            dtype=torch.bfloat16,
            num_bh=32,
            is_causal=False,
            standard_dense_output=True,
            standard_causal_output=False,
            plain_row_body=True,
        )
    # The standard dense output keeps the packet and its measured cadence.
    assert standard.exp2_packet == _DEG2_PACKET
    assert standard.e2e_schedule == "16/6"


@pytest.mark.parametrize("packet", (*_PLAIN_PACKETS[1:], _DEG2_PACKET))
def test_causal_hd128_resolver_canonicalizes_plain_packets_away(packet: str) -> None:
    cfg = _resolve(
        {
            **_WINNER,
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_persistent": False,
            cute_flash.FLASH_EXP2_PACKET_KEY: packet,
        },
        is_causal=True,
    )
    assert cfg.exp2_packet == "1x1"  # pyrefly: ignore[missing-attribute]


def test_degree2_request_emits_the_exact_pass_for_the_lse_body() -> None:
    code = _emit({**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: _DEG2_PACKET})
    assert "degree2=True" not in code
    assert "_flash_mLSE" in code


def test_plain_packet_emits_batched_pairs_for_the_lse_body() -> None:
    code = _emit({**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: "8x2"})
    assert "pair_batch=8" in code
    assert "emu_batch=2" in code
    assert "pair_batch" not in _emit(_WINNER)


# --------------------------------------------------------------------------
# MMA prologue: QK0 before the Q1 wait; staged first loads
# --------------------------------------------------------------------------


def _mma_prologue_lines(code: str) -> list[str]:
    lines = code.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if "flash_q0_full = flash_q_cons.wait_and_advance()" in line
    )
    end = next(
        i for i in range(start, len(lines)) if "flash_k0_full.release()" in lines[i]
    )
    return lines[start : end + 1]


@pytest.mark.parametrize(
    "family", ("fa4_2cta", "fa4", "fa4_deep_1cta"), ids=("two_cta", "one_cta", "deep")
)
def test_mma_prologue_issues_qk0_before_waiting_for_q1(family: str) -> None:
    """The first score tile only needs Q0 and K0; Q1 is waited for right before
    QK1 so the second Q tile's load stays off the first tile's critical path."""
    code = _emit({**_WINNER, "cute_flash_pipeline_family": family})
    prologue = _mma_prologue_lines(code)
    q1_wait = next(
        i
        for i, line in enumerate(prologue)
        if "flash_q1_full = flash_q_cons.wait_and_advance()" in line
    )
    k0_wait = next(i for i, line in enumerate(prologue) if "flash_k0_full =" in line)
    first_commit = next(
        i for i, line in enumerate(prologue) if "commit(flash_s_full_ptr + 0" in line
    )
    second_commit = next(
        i for i, line in enumerate(prologue) if "commit(flash_s_full_ptr + 1" in line
    )
    assert k0_wait < first_commit < q1_wait < second_commit


def test_staged_first_load_order_is_a_searchable_fa4_choice() -> None:
    fragment = cast(
        "object", _fragments(is_causal=False)[cute_flash.FLASH_FIRST_LOAD_ORDER_KEY]
    )
    assert cute_flash.FLASH_FIRST_LOAD_ORDER_STAGED in tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    cfg = _resolve({**_WINNER, cute_flash.FLASH_FIRST_LOAD_ORDER_KEY: 5})
    assert cfg.first_load_order == 5  # pyrefly: ignore[missing-attribute]
    # Outside the FA4 topology the order is canonical.
    ws = _resolve(
        {
            **_WINNER,
            "cute_flash_pipeline_family": "ws_overlap",
            cute_flash.FLASH_FIRST_LOAD_ORDER_KEY: 5,
        }
    )
    assert ws.first_load_order == 0  # pyrefly: ignore[missing-attribute]


@pytest.mark.parametrize(
    ("family", "remote_arrive"),
    (("fa4_2cta", True), ("fa4", False)),
    ids=("two_cta", "one_cta"),
)
def test_staged_first_load_order_emits_the_prologue_handshake(
    family: str, remote_arrive: bool
) -> None:
    code = _emit(
        {
            **_WINNER,
            "cute_flash_pipeline_family": family,
            cute_flash.FLASH_FIRST_LOAD_ORDER_KEY: 5,
        }
    )
    assert "flash_prologue_ptr = storage.prologue_mbar.data_ptr()" in code
    assert "cute.arch.mbarrier_init(flash_prologue_ptr, 1)" in code
    # Load warp: K0 and Q0 are issued, the first work item waits, then Q1 and V0.
    lines = code.splitlines()
    k0 = next(
        i
        for i, line in enumerate(lines)
        if "cute.copy(_flash_tma_k, tKgK[None, 0]" in line
    )
    wait = next(
        i
        for i, line in enumerate(lines)
        if "mbar_spin_wait(flash_prologue_ptr, cutlass.Int32(0)" in line
    )
    q1 = next(
        i for i, line in enumerate(lines) if "tQgQ[None, flash_q_mma_tile1]" in line
    )
    v0 = next(
        i
        for i, line in enumerate(lines)
        if "cute.copy(_flash_tma_v, tVgV[None, 0]" in line
    )
    q0 = next(
        i for i, line in enumerate(lines) if "tQgQ[None, flash_q_mma_tile0]" in line
    )
    assert k0 < q0 < wait < q1 < v0
    assert "flash_load_first = cutlass.Boolean(True)" in code
    # MMA warp: one elected lane arrives after its first Q0/K0 wait.
    assert "cute.arch.mbarrier_arrive(flash_prologue_ptr)" in code
    assert "flash_mma_first = cutlass.Boolean(True)" in code
    remote = (
        "mbarrier_arrive(flash_prologue_ptr, cutlass.Int32(1), flash_mma_tile_coord_v)"
    )
    assert (remote in code) is remote_arrive


def test_unstaged_orders_do_not_emit_the_prologue_handshake() -> None:
    for order in range(5):
        code = _emit({**_WINNER, cute_flash.FLASH_FIRST_LOAD_ORDER_KEY: order})
        assert "prologue_mbar" not in code
        assert "flash_load_first" not in code
        assert "flash_mma_first" not in code


# --------------------------------------------------------------------------
# GPU numerics
# --------------------------------------------------------------------------


def _run(
    config: dict[str, object], *args: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = helion.kernel(
        _attention_with_lse.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config(**config),
    )
    return kernel(*args)


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
    ("family", "persistent", "shape"),
    (
        ("fa4_2cta", True, (2, 16, 2048, 128)),
        ("fa4", True, (1, 4, 2048, 128)),
        ("fa4", False, (1, 2, 1024, 128)),
    ),
    ids=("two_cta_persistent", "one_cta_persistent", "one_cta_flat"),
)
def test_staged_first_load_order_matches_the_unstaged_order_bitwise(
    family: str, persistent: bool, shape: tuple[int, int, int, int]
) -> None:
    """The staged order only changes when the first work item's Q1/V0 TMA
    loads are issued: same arithmetic, so the output and LSE are identical."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
    )
    base = {
        **_WINNER,
        "cute_flash_pipeline_family": family,
        "cute_flash_persistent": persistent,
    }
    out_ref, lse_ref = _reference(q, k, v)
    out4, lse4 = _run({**base, cute_flash.FLASH_FIRST_LOAD_ORDER_KEY: 4}, q, k, v)
    out5, lse5 = _run({**base, cute_flash.FLASH_FIRST_LOAD_ORDER_KEY: 5}, q, k, v)
    assert torch.equal(out4, out5)
    assert torch.equal(lse4, lse5)
    assert (out5.float() - out_ref.float()).abs().max().item() < 1e-2
    assert (lse5 - lse_ref).abs().max().item() < 1e-3


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_degree2_packet_is_rejected_on_the_lse_body() -> None:
    """A pinned degree-2 config is invalid for the LSE kernel rather than
    silently running the polynomial pass."""
    q, k, v = (
        torch.randn(1, 4, 2048, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    with pytest.raises(helion.exc.InvalidConfig, match=_DEG2_PACKET):
        _run({**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: _DEG2_PACKET}, q, k, v)


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_plain_packets_on_the_lse_body_are_bitwise_identical() -> None:
    """The batched packets reorder instruction emission only."""
    torch.manual_seed(2)
    q, k, v = (
        torch.randn(1, 8, 2048, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    out_ref, lse_ref = _run(_WINNER, q, k, v)
    for packet in _PLAIN_PACKETS[1:]:
        out, lse = _run({**_WINNER, cute_flash.FLASH_EXP2_PACKET_KEY: packet}, q, k, v)
        assert torch.equal(out, out_ref), packet
        assert torch.equal(lse, lse_ref), packet


def test_causal_plan_is_untouched_by_the_dense_packet_surface() -> None:
    """Sanity: the causal hd128 fragment surface still exists and resolves."""
    fragments = _fragments(is_causal=True)
    assert cute_flash.FLASH_EXP2_PACKET_KEY in fragments
    assert causal_score_plan(128) is not None


def test_compound_degree2_overrides_require_the_standard_dense_output() -> None:
    """The structural-qualification path asks the compound-packet overrides
    whether the packet is effective; they agree with the resolver and the
    fragment surface: the standard dense output only, never the LSE body."""
    common = {"dtype": torch.bfloat16, "is_causal": False}
    requirements = cute_flash._flash_compound_exp2_packet_overrides(
        128,
        16,
        {cute_flash.FLASH_EXP2_PACKET_KEY: _DEG2_PACKET},
        standard_dense_output=True,
        **common,
    )
    assert requirements[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4_2cta"
    assert requirements[cute_flash.FLASH_E2E_SCHEDULE_KEY] == "16/6"
    assert (
        cute_flash._flash_compound_exp2_packet_overrides(
            128,
            16,
            {cute_flash.FLASH_EXP2_PACKET_KEY: _DEG2_PACKET},
            standard_dense_output=False,
            **common,
        )
        == {}
    )


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_lse_body_config_generation_accepts_every_structural_leaf() -> None:
    """The flash structural qualification builds one ``create_config_generation``
    per structural leaf of the population; every leaf of the attention example
    (plain row body, LSE output) must be accepted, and the degree-2 leaf is not
    part of that population."""
    q, k, v = (
        torch.empty(2, 16, 2048, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    bound = _attention_with_lse.bind((q, k, v))
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    generation = spec.create_config_generation()
    leaves = set()
    for base in generation.flash_deterministic_population_configs():
        leaf = cute_flash.flash_structural_leaf_from_config(base.config)
        assert leaf is not None
        leaves.add(leaf)
        overrides: dict[str, object] = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: leaf.pipeline_family,
            cute_flash.FLASH_SOFTMAX_DISC_KEY: leaf.softmax_disc,
        }
        if leaf.compound_exp2_packet is not None:
            overrides[cute_flash.FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
        leaf_generation = spec.create_config_generation(overrides=overrides)
        flat = leaf_generation.flatten(base)
        _, canonical = leaf_generation.canonicalize_flat(flat)
        assert (
            canonical.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
            == leaf.pipeline_family
        )
    assert not any(leaf.compound_exp2_packet == _DEG2_PACKET for leaf in leaves)
    packets = {
        base.config.get(cute_flash.FLASH_EXP2_PACKET_KEY)
        for base in generation.flash_deterministic_population_configs()
    }
    assert set(_PLAIN_PACKETS) <= packets
    for plain in _PLAIN_PACKETS[1:]:
        spec.create_config_generation(
            overrides={cute_flash.FLASH_EXP2_PACKET_KEY: plain}
        )
