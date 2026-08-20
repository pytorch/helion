from __future__ import annotations

import ast
import dataclasses
import importlib
import math
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import SOFTCAP_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.causal_range import CausalRangeProof
from helion._compiler.cute.flash_policy import get_flash_target_policy
from helion._compiler.cute.flash_tuning import FlashSoftmaxLowering
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

# ``_flash_runtime`` imports cutlass at module scope, so it must be loaded after
# the skip gate above rather than with the normal top-of-file imports.
_flash_runtime = importlib.import_module("helion._compiler.cute._flash_runtime")

_DEG2_PACKET = "deg2_16x6"
_HYBRID_PACKET = "hybrid_deg1_16x8"
_DEG1_PACKET = "deg1_16x8"
_DEG1_SHORT_PACKET = "deg1_8x2_corr10"


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention_output(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * math.log2(math.e)
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
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _dense_attention_output(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * math.log2(math.e)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
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
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view(q_in.size())


def _manual_config(
    packet: str = _DEG2_PACKET, **overrides: object
) -> dict[str, object]:
    config: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_EXP2_PACKET_KEY: packet,
        cute_flash.FLASH_DISC_PIPE_KEY: 3,
        cute_flash.FLASH_P_STORE_REP_KEY: 16,
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    }
    config.update(overrides)
    return config


def _hybrid_runtime_config() -> dict[str, object]:
    return {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4",
        "cute_flash_s_stage": 2,
        "cute_flash_kv_stage": 2,
        "cute_flash_persistent": False,
        "cute_flash_e2e_schedule": "16/8",
        "cute_flash_masked_e2e_schedule": "16/8",
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 10,
        "cute_flash_disc_pipe": 3,
        "cute_flash_exp2_packet": _HYBRID_PACKET,
        "cute_flash_wait_hint": 10_000_000,
        "cute_flash_p_store_rep": 16,
        "cute_flash_s_load_rep": 32,
        "cute_flash_epi_tma": False,
        "cute_flash_epi_stg": False,
        "cute_flash_role_map": "helion",
        "cute_flash_rescale_chunk_cols": 16,
        "cute_flash_causal_loop_split": True,
        "cute_flash_causal_lpt_swizzle": 0,
        "cute_flash_causal_kv_order": "descending",
        "cute_flash_softmax_regs": 184,
    }


def _emit_causal_resident_native_source(
    *,
    capability: tuple[int, int] = (10, 3),
    has_lse: bool = False,
    score_plan: AttentionScorePlan | None = None,
    num_kv: int = 512,
    config_overrides: dict[str, object] | None = None,
) -> str:
    if score_plan is None:
        score_plan = causal_score_plan(64)
    with patch.dict(os.environ, {}, clear=True):
        seed = cute_flash.flash_attention_seed_config(
            64,
            num_kv,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 3),
        )
        assert seed is not None
        manual_overrides = dict(seed.config)
        if config_overrides is not None:
            manual_overrides.update(config_overrides)
        config = cute_flash.resolve_flash_config(
            64,
            num_kv,
            manual_overrides,
            dtype=torch.float16,
            is_causal=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=1,
        total_tiles=num_kv // 2,
        cfg=config,
        has_lse=has_lse,
        io_dtype="cutlass.Float16",
        score_plan=score_plan,
        target_device_capability=capability,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _emit_dense_resident_value_graph_source(
    *,
    capability: tuple[int, int] = (10, 3),
    has_lse: bool = False,
    score_plan: AttentionScorePlan | None = None,
    num_kv: int = 256,
    config_overrides: dict[str, object] | None = None,
) -> str:
    if score_plan is None:
        score_plan = dense_score_plan(64)
    with patch.dict(os.environ, {}, clear=True):
        seed = cute_flash.flash_attention_seed_config(
            64,
            num_kv,
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
            target_device_capability=(10, 3),
        )
        assert seed is not None
        manual_overrides = dict(seed.config)
        if config_overrides is not None:
            manual_overrides.update(config_overrides)
        config = cute_flash.resolve_flash_config(
            64,
            num_kv,
            manual_overrides,
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=64,
        total_tiles=num_kv * 16,
        cfg=config,
        has_lse=has_lse,
        io_dtype="cutlass.Float16",
        score_plan=score_plan,
        target_device_capability=capability,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_causal_split_equal_iteration_proof_is_explicit() -> None:
    failed = cute_flash._flash_fa4_causal_split_equal_iteration_proof(
        split_range_proof=CausalRangeProof(False, "range failed"),
        query_slots_per_cta=2,
    )
    wrong_slots = cute_flash._flash_fa4_causal_split_equal_iteration_proof(
        split_range_proof=CausalRangeProof(True, "range proven"),
        query_slots_per_cta=1,
    )
    proven = cute_flash._flash_fa4_causal_split_equal_iteration_proof(
        split_range_proof=CausalRangeProof(True, "range proven"),
        query_slots_per_cta=2,
    )

    assert failed == CausalRangeProof(False, "range failed")
    assert wrong_slots == CausalRangeProof(
        False, "FA4 resident path requires two query slots"
    )
    assert proven.proven


def test_degree2_polynomial_relative_error_bound() -> None:
    x = torch.linspace(0.0, 1.0, 100_001, dtype=torch.float32)
    c0, c1, c2 = _flash_runtime._POLY_EX2_DEG2
    approximate = (torch.tensor(c2) * x + torch.tensor(c1)) * x + torch.tensor(c0)
    relative_error = (approximate / torch.exp2(x) - 1.0).abs()

    assert relative_error.max().item() < 0.00173


def test_legacy_degree1_packet_uses_degree2_evaluator() -> None:
    pair_result = object()
    with patch.object(
        _flash_runtime, "ex2_emulation_deg2_2", return_value=pair_result
    ) as pair_evaluator:
        assert _flash_runtime.ex2_emulation_deg1_2("x", "y") is pair_result
    pair_evaluator.assert_called_once_with("x", "y")

    batch_result = object()
    pairs = [("x0", "y0"), ("x1", "y1")]
    with patch.object(
        _flash_runtime, "ex2_emulation_deg2_batch", return_value=batch_result
    ) as batch_evaluator:
        assert _flash_runtime.ex2_emulation_deg1_batch(pairs) is batch_result
    batch_evaluator.assert_called_once_with(pairs)


def test_degree2_packet_codegen_route_is_exact_and_manual_only() -> None:
    assert cute_flash._flash_disc_exp2_codegen_params(_DEG2_PACKET, 8, 2) == (
        16,
        6,
        8,
        3,
        True,
        False,
    )
    assert cute_flash._flash_disc_exp2_codegen_params(_HYBRID_PACKET, 8, 2) == (
        16,
        8,
        8,
        4,
        True,
        True,
    )
    assert cute_flash._flash_disc_exp2_codegen_params(_DEG1_PACKET, 8, 2) == (
        16,
        8,
        8,
        4,
        False,
        True,
    )
    current_packets = {
        "1x1": (1, 1),
        "4x1": (4, 1),
        "4x2": (4, 2),
        "8x1": (8, 1),
        "8x2": (8, 2),
    }
    for packet, (pair_batch, emu_batch) in current_packets.items():
        assert cute_flash._flash_disc_exp2_codegen_params(packet, 13, 5) == (
            13,
            5,
            pair_batch,
            emu_batch,
            False,
            False,
        )


def test_degree2_packet_emits_exact_causal_pass2_arguments() -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(),
            dtype=torch.float16,
            is_causal=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=512,
        sequence_extent=65_536,
        num_bh=1,
        total_tiles=256,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16",
        score_plan=causal_score_plan(64),
    )
    module = ast.Module(body=body, type_ignores=[])
    pass2_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "fa4_disc_exp_convert_store_pipe",
            "fa4_disc_exp_convert_store_pipe_causal",
        }
    ]
    assert len(pass2_calls) == 4
    for call in pass2_calls:
        assert ast.literal_eval(call.args[8]) == 16
        assert ast.literal_eval(call.args[9]) == 6
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
        assert keywords == {"pair_batch": 8, "emu_batch": 3, "degree2": True}

    unmasked_loops = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "flash_kv_unmask_iter"
    ]
    assert len(unmasked_loops) == 2
    for loop in unmasked_loops:
        assert not any(
            "flash_kv_unmask_iter" in ast.unparse(node.test)
            for node in ast.walk(loop)
            if isinstance(node, ast.If)
        )
        assert any(
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Name)
            and statement.value.id == "flash_alpha"
            for statement in loop.body
        )
        assert any(
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "flash_acc_log >= -8.0"
            for node in ast.walk(loop)
        )


def test_sm103_packed_f16x2_rewrite_requires_exact_promoted_seed() -> None:
    intermediate_source = _emit_dense_resident_value_graph_source(
        num_kv=384,
        config_overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
            cute_flash.FLASH_PERSISTENT_KEY: False,
            cute_flash.FLASH_EXP2_PACKET_KEY: _DEG1_SHORT_PACKET,
            cute_flash.FLASH_Q_TILE_COUNT_KEY: 2,
        },
    )
    manual_source = _emit_dense_resident_value_graph_source(
        num_kv=2048,
        config_overrides={cute_flash.FLASH_E2E_OFFSET0_KEY: 9},
    )
    promoted_source = _emit_dense_resident_value_graph_source(num_kv=2048)

    assert "f16x2_xu=True" not in intermediate_source
    assert "f16x2_xu=True" not in manual_source
    assert "f16x2_xu=True" in promoted_source


def test_sm103_scaled_all_xu_codegen_covers_256k() -> None:
    all_xu_source = _emit_dense_resident_value_graph_source(num_kv=2048)
    overflow_source = _emit_dense_resident_value_graph_source(
        num_kv=2048,
        config_overrides={cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 9.0},
    )
    all_xu_module = ast.parse(all_xu_source)
    overflow_module = ast.parse(overflow_source)

    def _pass2_calls(module: ast.Module) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fa4_sp_exp_convert_store_whole_rowsum"
        ]

    all_xu_calls = _pass2_calls(all_xu_module)
    overflow_calls = _pass2_calls(overflow_module)
    assert len(all_xu_calls) == len(overflow_calls) == 2
    for call in all_xu_calls:
        assert [ast.literal_eval(call.args[index]) for index in (6, 7, 8)] == [
            16,
            0,
            0,
        ]
        assert any(
            keyword.arg == "f16x2_xu" and ast.literal_eval(keyword.value)
            for keyword in call.keywords
        )
    assert (
        "cutlass.Float32(7.0) - flash_row_max_safe * _flash_scale_log2" in all_xu_source
    )
    for call in overflow_calls:
        assert [ast.literal_eval(call.args[index]) for index in (6, 7)] == [16, 8]
        assert not any(keyword.arg == "f16x2_xu" for keyword in call.keywords)
    assert "cutlass.Float32(7.0)" not in overflow_source


def test_dense_target_lowering_match_is_environment_independent() -> None:
    dense_seed = cute_flash.flash_attention_seed_config(
        64,
        256,
        standard_dense_output=True,
        target_device_capability=(10, 3),
    )
    assert dense_seed is not None
    dense_cfg = cute_flash.resolve_flash_config(
        64,
        256,
        dense_seed.config,
        standard_dense_output=True,
    )

    with patch.dict(os.environ, {"HELION_CUTE_FLASH_WAIT_HINT": "0"}):
        assert cute_flash._flash_dense_resident_seed_matches(dense_cfg, 256, (10, 3))


@pytest.mark.parametrize("num_kv", (256, 512, 1024))
def test_dense_resident_value_graph_codegen_and_barrier_protocol(num_kv: int) -> None:
    source = _emit_dense_resident_value_graph_source(num_kv=num_kv)
    module = ast.parse(source)
    value_graph_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resident_softmax_value_graph"
    ]

    assert len(value_graph_calls) == 2
    for call in value_graph_calls:
        assert ast.unparse(call.args[10]).endswith("_corr_empty_ptr + 0")
        assert ast.unparse(call.args[11]) == "flash_s_corr_prod_phase"
        assert ast.unparse(call.args[12]) == "flash_row_sum * flash_alpha"
        assert {
            keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
        } == {
            "pfor_peer_cta_rank": "cutlass.Int32(0)",
            "pfor_self_cta_rank": "None",
        }

    assert "fa4_sp_exp_convert_store_whole_rowsum" not in source
    assert "f16x2_xu=True" not in source
    assert "cutlass.Float32(7.0) - flash_row_max_safe * _flash_scale_log2" in source
    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(1)") == 2

    stage0_start = source.index("flash_s_corr_prod_phase = cutlass.Int32(1)")
    stage1_start = source.index(
        "flash_s_corr_prod_phase = cutlass.Int32(1)", stage0_start + 1
    )
    softmax0 = source[stage0_start:stage1_start]
    empty_wait = (
        "_helion_flash_rt.mbar_spin_wait(flash_s0_corr_empty_ptr + 0, "
        "flash_s_corr_prod_phase, 10000000)"
    )
    assert softmax0.count(empty_wait) == 2
    assert softmax0.count("flash_s_corr_prod_phase ^= 1") == 2
    entry_wait = softmax0.index(empty_wait)
    alpha_store = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_alpha"
    )
    value_graph = softmax0.index("resident_softmax_value_graph", alpha_store)
    phase_advance = softmax0.index("flash_s_corr_prod_phase ^= 1", value_graph)
    rowsum_store = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_row_sum",
        phase_advance,
    )
    tail_wait = softmax0.index(empty_wait, rowsum_store)
    assert (
        entry_wait
        < alpha_store
        < value_graph
        < phase_advance
        < rowsum_store
        < tail_wait
    )
    assert "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum" not in softmax0

    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    steady_loop = correction.index("for flash_kv")
    alpha0 = correction.index("flash_a0 =", steady_loop)
    pfor0 = correction.index("flash_pfor_ptr + 0", alpha0)
    release1 = correction.index(
        "cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)", pfor0
    )
    alpha1 = correction.index("flash_a1 =", release1)
    pfor1 = correction.index("flash_pfor_ptr + 1", alpha1)
    release0 = correction.index(
        "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)", pfor1
    )
    assert alpha0 < pfor0 < release1 < alpha1 < pfor1 < release0


def test_dense_resident_value_graph_gate_preserves_fallbacks() -> None:
    base_plan = dense_score_plan(64)
    modified_plan = dataclasses.replace(
        base_plan,
        modifiers=(AttentionScoreModifier(SOFTCAP_KIND, value_log2=2.0),),
    )
    fallback_sources = (
        _emit_dense_resident_value_graph_source(capability=(10, 0)),
        _emit_dense_resident_value_graph_source(capability=(10, 4)),
        _emit_dense_resident_value_graph_source(num_kv=2048),
        _emit_dense_resident_value_graph_source(has_lse=True),
        _emit_dense_resident_value_graph_source(score_plan=modified_plan),
        _emit_dense_resident_value_graph_source(
            config_overrides={cute_flash.FLASH_E2E_OFFSET_KEY: 4}
        ),
    )
    for fallback_source in fallback_sources:
        assert "resident_softmax_value_graph" not in fallback_source
        assert "fa4_sp_exp_convert_store_whole_rowsum" in fallback_source


def test_sm103_dense64_register_policy_is_target_specific() -> None:
    sm103_seed = cute_flash.flash_attention_seed_config(
        64,
        512,
        dtype=torch.float16,
        is_causal=False,
        standard_dense_output=True,
        target_device_capability=(10, 3),
    )
    sm100_seed = cute_flash.flash_attention_seed_config(
        64,
        512,
        dtype=torch.float16,
        is_causal=False,
        standard_dense_output=True,
        target_device_capability=(10, 0),
    )

    assert sm103_seed is not None
    assert sm100_seed is not None
    assert sm103_seed.config[cute_flash.FLASH_CORR_REGS_KEY] == 72
    assert sm103_seed.config[cute_flash.FLASH_OTHER_REGS_KEY] == 40
    assert sm100_seed.config[cute_flash.FLASH_CORR_REGS_KEY] == 80
    assert sm100_seed.config[cute_flash.FLASH_OTHER_REGS_KEY] == 32


def test_dense_resident_softmax_lowering_dispatch_is_exhaustive() -> None:
    policy = get_flash_target_policy((10, 3)).tuning
    shape_policy = policy.dense_policy(256)
    assert shape_policy is not None

    standard_policy = dataclasses.replace(
        policy,
        dense_policies=(
            dataclasses.replace(
                shape_policy,
                softmax_lowering=FlashSoftmaxLowering.STANDARD,
            ),
            *policy.dense_policies[1:],
        ),
    )
    with patch.object(
        cute_flash,
        "get_flash_target_policy",
        return_value=dataclasses.replace(
            get_flash_target_policy((10, 3)), tuning=standard_policy
        ),
    ):
        standard_source = _emit_dense_resident_value_graph_source()
    assert "resident_softmax_value_graph" not in standard_source
    assert "fa4_sp_exp_convert_store_whole_rowsum" in standard_source


def test_dense_probability_shift_fails_closed_before_fp16_overflow() -> None:
    target_policy = get_flash_target_policy((10, 3))
    dense_policy = target_policy.tuning.dense_policy(256)
    assert dense_policy is not None
    unsafe_tuning = dataclasses.replace(
        target_policy.tuning,
        dense_policies=tuple(
            dataclasses.replace(policy, probability_log2_shift=16)
            if policy.num_kv == 256
            else policy
            for policy in target_policy.tuning.dense_policies
        ),
    )
    with patch.object(
        cute_flash,
        "get_flash_target_policy",
        return_value=dataclasses.replace(target_policy, tuning=unsafe_tuning),
    ):
        source = _emit_dense_resident_value_graph_source()

    assert "resident_softmax_value_graph" not in source
    assert "f16x2_xu=True" not in source
    assert "cutlass.Float32(16.0)" not in source


def test_causal_resident_native_codegen_and_single_stat_protocol() -> None:
    source = _emit_causal_resident_native_source()
    module = ast.parse(source)

    value_graph_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resident_softmax_value_graph"
    ]
    assert not value_graph_calls

    assert "fa4_disc_exp_convert_store" not in source
    assert "fa4_sp_exp_convert_store_whole_rowsum" not in source
    assert "f16x2_xu=True" not in source
    assert "flash_s_corr_prod_index" not in source
    assert "flash_s_corr_cons_index" not in source
    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(1)") == 2
    assert source.count("ResidentSoftmaxState.create") == 2
    assert source.count("flash_softmax.reset()") == 2
    assert source.count("update_row_max_masked(tLDrS.load(), True)") == 2
    assert source.count("update_row_max_masked(tLDrS.load(), False)") == 2
    assert source.count("update_row_max_precomputed(flash_hw_row_max, False)") == 2
    assert source.count("flash_softmax.scale_subtract_rowmax") == 6
    assert source.count("flash_softmax.apply_exp2_convert") == 6
    assert source.count("flash_tSrP_f32 = cute.make_rmem_tensor") == 6
    assert source.count("dtype=cutlass.Float16), tLDrS.layout") == 6
    assert source.count("flash_softmax.acquire_stats") == 6
    assert source.count("flash_softmax.update_row_sum") == 6
    assert source.count("tLDrS.load(), flash_alpha, True") == 2
    assert source.count("tLDrS.load(), flash_alpha)") == 4
    assert source.count("for flash_kv_mask_iter") == 2
    assert source.count("for flash_kv_unmask_iter") == 2
    assert source.count("- flash_m_tile0 - cutlass.Int32(1)") == 1
    assert source.count("- flash_m_tile1 - cutlass.Int32(1)") == 1

    stage0_start = source.index("flash_s_corr_prod_phase = cutlass.Int32(1)")
    stage1_start = source.index(
        "flash_s_corr_prod_phase = cutlass.Int32(1)", stage0_start + 1
    )
    softmax0 = source[stage0_start:stage1_start]
    stage0_block = source[
        source.index("if warp_idx < 4:") : source.index(
            "if (warp_idx >= 4) & (warp_idx < 8):"
        )
    ]
    assert "flash_tiled_ld0" in stage0_block
    assert "flash_tiled_ld1" not in stage0_block
    masked_start = softmax0.index("for flash_kv_mask_iter")
    unmasked_start = softmax0.index("for flash_kv_unmask_iter")
    masked = softmax0[masked_start:unmasked_start]
    unmasked = softmax0[unmasked_start:]
    assert "causal_mask_t2r" in masked
    assert "flash_tiled_ldred0" not in masked
    assert "flash_tiled_ldred0" in unmasked
    assert "causal_mask_t2r" not in unmasked

    alpha_store = masked.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_alpha"
    )
    alpha_ready = masked.index("named_barrier_arrive_unaligned", alpha_store)
    scale_subtract = masked.index("flash_softmax.scale_subtract_rowmax", alpha_ready)
    fresh_p = masked.index("flash_tSrP_f32 = cute.make_rmem_tensor", scale_subtract)
    early_pack = masked.index("flash_softmax.apply_exp2_convert", fresh_p)
    stats_acquire = masked.index("flash_softmax.acquire_stats", early_pack)
    row_sum = masked.index("flash_softmax.update_row_sum", stats_acquire)
    phase_advance = masked.index("flash_s_corr_prod_phase ^= 1", row_sum)
    assert (
        alpha_store
        < alpha_ready
        < scale_subtract
        < fresh_p
        < early_pack
        < stats_acquire
        < row_sum
        < phase_advance
    )
    assert "flash_row_sum * flash_alpha + flash_p_sum" not in masked
    assert (
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_softmax.row_sum[0]"
        in softmax0
    )

    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    assert "flash_a0 = flash_scale_t[0 * 128 + flash_local_tidx]" in correction
    assert "flash_a1 = flash_scale_t[1 * 128 + flash_local_tidx]" in correction
    assert "cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)" in correction
    assert "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)" in correction


def test_causal_resident_stage_local_stat_protocol() -> None:
    source = _emit_causal_resident_native_source()
    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    ready0 = "_helion_flash_rt.named_barrier_wait_unaligned(3 + warp_idx % 4, 64)"
    ready1 = "_helion_flash_rt.named_barrier_wait_unaligned(7 + warp_idx % 4, 64)"
    empty0 = "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)"
    empty1 = "cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)"

    dummy_ready0 = correction.index(ready0)
    dummy_empty0 = correction.index(empty0, dummy_ready0)
    dummy_ready1 = correction.index(ready1, dummy_empty0)
    dummy_empty1 = correction.index(empty1, dummy_ready1)
    steady_loop = correction.index("for flash_kv", dummy_empty1)
    alpha0 = correction.index("flash_a0 =", steady_loop)
    pfor0 = correction.index("flash_pfor_ptr + 0", alpha0)
    steady_empty0 = correction.index(empty0, pfor0)
    alpha1 = correction.index("flash_a1 =", steady_empty0)
    pfor1 = correction.index("flash_pfor_ptr + 1", alpha1)
    steady_empty1 = correction.index(empty1, pfor1)
    final_ready0 = correction.index(ready0, steady_empty1)

    assert (
        dummy_ready0
        < dummy_empty0
        < dummy_ready1
        < dummy_empty1
        < steady_loop
        < alpha0
        < pfor0
        < steady_empty0
        < alpha1
        < pfor1
        < steady_empty1
        < final_ready0
    )

    cross_source = _emit_causal_resident_native_source(num_kv=1024)
    cross = cross_source[cross_source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    cross_loop = cross.index("for flash_kv")
    cross_alpha0 = cross.index("flash_a0 =", cross_loop)
    cross_pfor0 = cross.index("flash_pfor_ptr + 0", cross_alpha0)
    cross_empty1 = cross.index(empty1, cross_pfor0)
    cross_alpha1 = cross.index("flash_a1 =", cross_empty1)
    cross_pfor1 = cross.index("flash_pfor_ptr + 1", cross_alpha1)
    cross_empty0 = cross.index(empty0, cross_pfor1)
    held_empty1 = cross.index(empty1, cross_empty0)
    cross_final_ready0 = cross.index(ready0, held_empty1)
    assert (
        cross_alpha0
        < cross_pfor0
        < cross_empty1
        < cross_alpha1
        < cross_pfor1
        < cross_empty0
        < held_empty1
        < cross_final_ready0
    )


def test_causal_resident_softmax_lowering_dispatch_is_exhaustive() -> None:
    policy = get_flash_target_policy((10, 3)).tuning
    shape_policy = policy.causal_policy(512)
    assert shape_policy is not None

    standard_policy = dataclasses.replace(
        policy,
        causal_policies=(
            dataclasses.replace(
                shape_policy,
                softmax_lowering=FlashSoftmaxLowering.STANDARD,
            ),
            *policy.causal_policies[1:],
        ),
    )
    with patch.object(
        cute_flash,
        "get_flash_target_policy",
        return_value=dataclasses.replace(
            get_flash_target_policy((10, 3)), tuning=standard_policy
        ),
    ):
        standard_source = _emit_causal_resident_native_source()
    assert "resident_softmax_value_graph" not in standard_source
    assert "fa4_sp_exp_convert_store_whole_rowsum" in standard_source

    value_graph_policy = dataclasses.replace(
        policy,
        causal_policies=(
            dataclasses.replace(
                shape_policy,
                softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            ),
            *policy.causal_policies[1:],
        ),
    )
    with patch.object(
        cute_flash,
        "get_flash_target_policy",
        return_value=dataclasses.replace(
            get_flash_target_policy((10, 3)), tuning=value_graph_policy
        ),
    ):
        value_graph_source = _emit_causal_resident_native_source()
    assert "ResidentSoftmaxState.create" not in value_graph_source
    assert "resident_softmax_value_graph" in value_graph_source


@pytest.mark.parametrize(("num_kv", "kv_stage"), ((1024, 3), (2048, 3), (4096, 6)))
def test_causal_resident_native_additional_stage_codegen(
    num_kv: int, kv_stage: int
) -> None:
    source = _emit_causal_resident_native_source(num_kv=num_kv)

    assert f"num_stages={kv_stage}" in source
    assert "resident_softmax_value_graph" in source
    assert "fa4_sp_exp_convert_store_whole_rowsum" not in source
    assert "fa4_disc_exp_convert_store" not in source
    assert source.count("for flash_kv_mask_iter") == 2
    assert source.count("for flash_kv_unmask_iter") == 2
    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(1)") == 2


def test_causal_resident_native_gate_preserves_fallbacks() -> None:
    base_plan = causal_score_plan(64)
    modified_plan = dataclasses.replace(
        base_plan,
        modifiers=(
            AttentionScoreModifier(SOFTCAP_KIND, value_log2=2.0),
            *base_plan.modifiers,
        ),
    )
    fallback_sources = (
        _emit_causal_resident_native_source(capability=(10, 0)),
        _emit_causal_resident_native_source(capability=(10, 4)),
        _emit_causal_resident_native_source(has_lse=True),
        _emit_causal_resident_native_source(score_plan=modified_plan),
        _emit_causal_resident_native_source(num_kv=256),
        _emit_causal_resident_native_source(
            config_overrides={cute_flash.FLASH_EXP2_PACKET_KEY: "4x1"}
        ),
        _emit_causal_resident_native_source(
            config_overrides={cute_flash.FLASH_SOFTMAX_REGS_KEY: 176}
        ),
    )
    for fallback_source in fallback_sources:
        assert "fa4_disc_exp_convert_store" in fallback_source
        assert "resident_softmax_value_graph" not in fallback_source


@onlyBackends(["cute"])
def test_causal_resident_native_matches_sdpa_without_deadlock() -> None:
    capability = torch.cuda.get_device_capability()
    target_policy = get_flash_target_policy(capability)
    hardware = target_policy.hardware
    tuning = target_policy.tuning
    causal_policy = tuning.causal_policy(512)
    if (
        not hardware.supports_tmem_row_reduce
        or causal_policy is None
        or causal_policy.softmax_lowering is not FlashSoftmaxLowering.STATEFUL
    ):
        pytest.skip("causal resident native softmax is unsupported on this target")

    torch.manual_seed(109)
    shape = (1, 1, 65_536, 64)
    q = torch.randn(shape, dtype=torch.float16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    seed = cute_flash.flash_attention_seed_config(
        64,
        512,
        is_causal=True,
        standard_causal_output=True,
        target_device_capability=capability,
    )
    assert seed is not None
    config = helion.Config(**seed.config)
    bound = _causal_attention_output.bind((q, k, v))
    code = bound.to_triton_code(config)
    compiled = bound.compile_config(config)

    assert "ResidentSoftmaxState.create" in code
    assert "resident_softmax_value_graph" not in code
    assert "flash_softmax.update_row_sum(tLDrS.load(), flash_alpha)" in code
    assert "fa4_sp_exp_convert_store_whole_rowsum" not in code
    assert "fa4_disc_exp_convert_store" not in code
    assert "flash_s_corr_prod_index" not in code
    assert "f16x2_xu=True" not in code

    out = compiled(q, k, v)
    for _ in range(31):
        repeated = compiled(q, k, v)
    torch.cuda.synchronize()
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize(
    ("num_kv", "repeat_count"),
    (
        (1024, 31),
        (4096, 3),
    ),
)
@onlyBackends(["cute"])
def test_causal_resident_cross_stage_matches_sdpa_without_deadlock(
    num_kv: int, repeat_count: int
) -> None:
    capability = torch.cuda.get_device_capability()
    policy = get_flash_target_policy(capability).tuning
    shape_policy = policy.causal_policy(num_kv)
    if shape_policy is None:
        pytest.skip("causal resident native softmax is unsupported on this target")
    assert shape_policy.softmax_lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH

    torch.manual_seed(109 + num_kv)
    sequence_length = num_kv * 128
    shape = (1, 1, sequence_length, 64)
    q = torch.randn(shape, dtype=torch.float16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    seed = cute_flash.flash_attention_seed_config(
        64,
        num_kv,
        dtype=torch.float16,
        is_causal=True,
        standard_causal_output=True,
        target_device_capability=capability,
    )
    assert seed is not None
    config = helion.Config(**seed.config)
    bound = _causal_attention_output.bind((q, k, v))
    code = bound.to_triton_code(config)
    compiled = bound.compile_config(config)

    assert "resident_softmax_value_graph" in code
    assert "fa4_sp_exp_convert_store_whole_rowsum" not in code
    assert "fa4_disc_exp_convert_store" not in code

    out = compiled(q, k, v)
    repeated = out
    for _ in range(repeat_count):
        repeated = compiled(q, k, v)
    torch.cuda.synchronize()
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.02)


def test_causal_target_lowering_match_is_environment_independent() -> None:
    causal_seed = cute_flash.flash_attention_seed_config(
        64,
        512,
        is_causal=True,
        standard_causal_output=True,
        target_device_capability=(10, 3),
    )
    assert causal_seed is not None
    causal_cfg = cute_flash.resolve_flash_config(
        64,
        512,
        causal_seed.config,
        is_causal=True,
    )

    with patch.dict(os.environ, {"HELION_CUTE_FLASH_WAIT_HINT": "0"}):
        assert cute_flash._flash_causal_resident_native_seed_matches(
            causal_cfg, 512, (10, 3)
        )

    effective = cute_flash._flash_resident_softmax_config(causal_cfg)
    assert causal_cfg.exp2_packet == _DEG2_PACKET
    assert causal_cfg.stat_transport == "ring2"
    assert effective.exp2_packet == "1x1"
    assert effective.stat_transport == "single"
    assert effective.exp2_impl == "xu"


def test_hybrid_packet_uses_degree1_only_for_unmasked_pass2() -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(_HYBRID_PACKET),
            dtype=torch.float16,
            is_causal=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=512,
        sequence_extent=65_536,
        num_bh=1,
        total_tiles=256,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16",
        score_plan=causal_score_plan(64),
        target_device_capability=(10, 3),
    )
    module = ast.Module(body=body, type_ignores=[])
    pass2_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "fa4_disc_exp_convert_store_pipe",
            "fa4_disc_exp_convert_store_pipe_causal",
        }
    ]
    assert len(pass2_calls) == 4
    for call in pass2_calls:
        assert ast.literal_eval(call.args[8]) == 16
        assert ast.literal_eval(call.args[9]) == 8
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
        polynomial = (
            {"degree2": True}
            if call.func.attr.endswith("_causal")
            else {"degree1": True}
        )
        assert keywords == {"pair_batch": 8, "emu_batch": 4, **polynomial}


@pytest.mark.parametrize(
    (
        "packet",
        "num_kv",
        "sequence_extent",
        "total_tiles",
        "freq",
        "res",
        "emu",
        "polynomial_keyword",
    ),
    (
        (_DEG1_SHORT_PACKET, 256, 32_768, 4_096, 8, 2, 2, "degree1"),
        (_DEG1_SHORT_PACKET, 512, 65_536, 8_192, 8, 2, 2, "degree1"),
        (_DEG1_SHORT_PACKET, 1024, 131_072, 16_384, 8, 2, 2, "degree1"),
        (_DEG1_PACKET, 2048, 262_144, 32_768, 16, 8, 4, "degree1"),
        (_DEG2_PACKET, 2048, 262_144, 32_768, 16, 6, 3, "degree2"),
    ),
)
def test_polynomial_packet_uses_whole_row_dense_pass2(
    packet: str,
    num_kv: int,
    sequence_extent: int,
    total_tiles: int,
    freq: int,
    res: int,
    emu: int,
    polynomial_keyword: str,
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            num_kv,
            _manual_config(
                packet,
                **{cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
            ),
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
    assert config.exp2_packet == packet
    assert config.e2e_schedule == f"{freq}/{res}"
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=num_kv,
        sequence_extent=sequence_extent,
        num_bh=64,
        total_tiles=total_tiles,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16",
        score_plan=dense_score_plan(64),
        target_device_capability=(10, 0),
    )
    module = ast.Module(body=body, type_ignores=[])
    pass2_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "fa4_sp_exp_convert_store",
            "fa4_sp_exp_convert_store_whole_rowsum",
        }
    ]
    assert len(pass2_calls) == 2
    for call in pass2_calls:
        assert ast.literal_eval(call.args[6]) == freq
        assert ast.literal_eval(call.args[7]) == res
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
        assert keywords == {
            "early_split_publish": True,
            "pair_batch": 8,
            "emu_batch": emu,
            polynomial_keyword: True,
        }


@pytest.mark.parametrize(
    ("packet", "expected_alpha_stages"),
    ((_DEG1_SHORT_PACKET, ("flash_a1", "flash_a0")), ("8x2", ("flash_a0", "flash_a1"))),
)
def test_short_degree1_packet_reverses_steady_correction_order(
    packet: str, expected_alpha_stages: tuple[str, str]
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            256,
            _manual_config(
                packet,
                **{cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
            ),
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=256,
        sequence_extent=32_768,
        num_bh=64,
        total_tiles=4_096,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16",
        score_plan=dense_score_plan(64),
        target_device_capability=(10, 3),
    )
    correction_loop = next(
        node
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "flash_kv"
        and any(
            isinstance(child, ast.Name) and child.id == "flash_a0"
            for child in ast.walk(node)
        )
    )
    alpha_stages = [
        target.id
        for statement in correction_loop.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name) and target.id in ("flash_a0", "flash_a1")
    ]
    assert alpha_stages == list(expected_alpha_stages)


def _emit_dense_single_stat_source(
    *,
    pipeline_family: str = "fa4_2cta",
    packet: str = "8x2",
    softmax_disc: bool = False,
    rescale_threshold: float = 8.0,
    persistent: bool = True,
    final_only_stat_pipeline: bool = False,
) -> str:
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            256,
            _manual_config(
                packet,
                **{
                    cute_flash.FLASH_PIPELINE_FAMILY_KEY: pipeline_family,
                    cute_flash.FLASH_STAT_TRANSPORT_KEY: (
                        "single_final" if final_only_stat_pipeline else "single"
                    ),
                    cute_flash.FLASH_PERSISTENT_KEY: persistent,
                    cute_flash.FLASH_SOFTMAX_DISC_KEY: softmax_disc,
                    cute_flash.FLASH_RESCALE_THRESHOLD_KEY: rescale_threshold,
                },
            ),
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
        body = cute_flash.emit_flash_fa4_device_body(
            cast("DeviceFunction", None),
            head_dim=64,
            num_kv=256,
            sequence_extent=32_768,
            num_bh=64,
            total_tiles=4_096 if config.use_2cta_instrs else 8_192,
            cfg=config,
            has_lse=False,
            io_dtype="cutlass.Float16",
            score_plan=dense_score_plan(64),
            target_device_capability=(10, 3),
        )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_dense_single_stat_emits_fa4_late_acquire_protocol() -> None:
    source = _emit_dense_single_stat_source()

    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(1)") == 2
    assert "flash_s_corr_prod_phase = cutlass.Int32(0)" not in source

    stage0_start = source.index("flash_s_corr_prod_phase = cutlass.Int32(1)")
    stage1_start = source.index(
        "flash_s_corr_prod_phase = cutlass.Int32(1)", stage0_start + 1
    )
    softmax0 = source[stage0_start:stage1_start]
    empty_wait = (
        "_helion_flash_rt.mbar_spin_wait(flash_s0_corr_empty_ptr + 0, "
        "flash_s_corr_prod_phase, 10000000)"
    )
    empty_wait_lines = [line for line in softmax0.splitlines() if empty_wait in line]
    # Entry and post-P acquires live inside each persistent work item. The final
    # acquire drains the final row-sum handoff after the persistent loop.
    assert [len(line) - len(line.lstrip()) for line in empty_wait_lines] == [8, 12, 4]
    assert softmax0.count("flash_s_corr_prod_phase ^= 1") == 2
    assert (
        "else:\n"
        "                _helion_flash_rt.named_barrier_arrive_unaligned("
        "3 + warp_idx % 4, 64)"
    ) in softmax0
    entry_wait = softmax0.index(empty_wait)
    alpha_store = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_alpha"
    )
    p_publication = softmax0.index("flash_pfor_ptr + 0", alpha_store)
    post_p_wait = softmax0.index(empty_wait, p_publication)
    row_sum_update = softmax0.index(
        "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum", post_p_wait
    )
    rowsum_store = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_row_sum", row_sum_update
    )
    tail_wait = softmax0.index(empty_wait, rowsum_store)
    assert (
        entry_wait
        < alpha_store
        < p_publication
        < post_p_wait
        < row_sum_update
        < rowsum_store
        < tail_wait
    )

    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    work_loop = correction.index("for flash_tile_iter")
    ready0 = "_helion_flash_rt.named_barrier_wait_unaligned(3 + warp_idx % 4, 64)"
    ready1 = "_helion_flash_rt.named_barrier_wait_unaligned(7 + warp_idx % 4, 64)"
    empty0 = "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)"
    empty1 = "cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)"

    dummy_ready0 = correction.index(ready0, work_loop)
    dummy_release0 = correction.index(empty0, dummy_ready0)
    dummy_ready1 = correction.index(ready1, dummy_release0)
    steady_loop = correction.index("for flash_kv", dummy_ready1)
    alpha0 = correction.index("flash_a0 =", steady_loop)
    pfor0 = correction.index(
        "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0", alpha0
    )
    cross_release1 = correction.index(empty1, pfor0)
    alpha1 = correction.index("flash_a1 =", cross_release1)
    pfor1 = correction.index(
        "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 1", alpha1
    )
    cross_release0 = correction.index(empty0, pfor1)
    held_release1 = correction.index(empty1, cross_release0)
    final_ready0 = correction.index(ready0, held_release1)
    final_rowsum0 = correction.index("flash_inv_sum0 =", final_ready0)
    final_release0 = correction.index(empty0, final_rowsum0)
    final_ready1 = correction.index(ready1, final_release0)
    final_rowsum1 = correction.index("flash_inv_sum1 =", final_ready1)
    final_release1 = correction.index(empty1, final_rowsum1)
    assert (
        dummy_ready0
        < dummy_release0
        < dummy_ready1
        < steady_loop
        < alpha0
        < pfor0
        < cross_release1
        < alpha1
        < pfor1
        < cross_release0
        < held_release1
        < final_ready0
        < final_rowsum0
        < final_release0
        < final_ready1
        < final_rowsum1
        < final_release1
    )


def test_dense_nonpersistent_final_only_stat_pipeline_protocol() -> None:
    source = _emit_dense_single_stat_source(
        persistent=False,
        final_only_stat_pipeline=True,
    )

    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(0)") == 2
    assert "flash_s_corr_prod_phase = cutlass.Int32(1)" not in source
    assert "flash_s_corr_prod_phase ^= 1" not in source

    stage0_start = source.index("flash_s_corr_prod_phase = cutlass.Int32(0)")
    stage1_start = source.index(
        "flash_s_corr_prod_phase = cutlass.Int32(0)", stage0_start + 1
    )
    softmax0 = source[stage0_start:stage1_start]
    empty_wait = (
        "_helion_flash_rt.mbar_spin_wait(flash_s0_corr_empty_ptr + 0, "
        "flash_s_corr_prod_phase, 10000000)"
    )
    ready_arrive = (
        "_helion_flash_rt.named_barrier_arrive_unaligned(3 + warp_idx % 4, 64)"
    )
    assert softmax0.count(empty_wait) == 1
    assert softmax0.count(ready_arrive) == 2
    assert "else:\n            " + ready_arrive not in softmax0

    alpha_store = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_alpha"
    )
    p_publication = softmax0.index("flash_pfor_ptr + 0", alpha_store)
    row_sum_update = softmax0.index(
        "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum",
        p_publication,
    )
    terminal_wait = softmax0.index(empty_wait, row_sum_update)
    rowsum_store = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_row_sum",
        terminal_wait,
    )
    rowsum_ready = softmax0.index(ready_arrive, rowsum_store)
    dealloc = softmax0.index(
        "_helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)",
        rowsum_ready,
    )
    assert (
        alpha_store
        < p_publication
        < row_sum_update
        < terminal_wait
        < rowsum_store
        < rowsum_ready
        < dealloc
    )

    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    steady_loop = correction.index("for flash_kv")
    empty0 = "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)"
    empty1 = "cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)"
    assert correction.count(empty0) == 1
    assert correction.count(empty1) == 1

    alpha0 = correction.index("flash_a0 =", steady_loop)
    pfor0 = correction.index(
        "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0", alpha0
    )
    alpha1 = correction.index("flash_a1 =", pfor0)
    pfor1 = correction.index(
        "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 1", alpha1
    )
    terminal_release0 = correction.index(empty0, pfor1)
    terminal_release1 = correction.index(empty1, terminal_release0)
    final_ready0 = correction.index(
        "_helion_flash_rt.named_barrier_wait_unaligned(3 + warp_idx % 4, 64)",
        terminal_release1,
    )
    final_rowsum0 = correction.index("flash_inv_sum0 =", final_ready0)
    assert (
        steady_loop
        < alpha0
        < pfor0
        < alpha1
        < pfor1
        < terminal_release0
        < terminal_release1
        < final_ready0
        < final_rowsum0
    )
    assert empty0 not in correction[steady_loop:terminal_release0]
    assert empty1 not in correction[steady_loop:terminal_release0]
    assert "cute.arch.barrier(" not in correction
    assert (
        correction.count("_helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)")
        == 1
    )


@onlyBackends(["cute"])
def test_dense_degree2_final_only_is_deterministic_on_peaky_inputs() -> None:
    torch.manual_seed(104)
    q, k, v = (
        torch.randn(1, 1, 33_280, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )
    q.mul_(2.0)
    k.mul_(2.0)
    config = next(
        seed.config
        for seed in cute_flash.flash_attention_seed_configs(
            64,
            260,
            dtype=torch.float16,
            standard_dense_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _DEG2_PACKET
        and seed.config.get(cute_flash.FLASH_STAT_TRANSPORT_KEY) == "single_final"
    )

    bound = _dense_attention_output.bind((q, k, v))
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    out = bound(q, k, v)
    repeated = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    diff = (out.float() - expected.float()).abs()
    strict_failures = diff > (0.002 + 0.01 * expected.float().abs())
    normalized_rmse = torch.sqrt((diff * diff).mean(dtype=torch.float64)) / torch.sqrt(
        (expected.float() * expected.float()).mean(dtype=torch.float64)
    )

    assert "degree2=True" in code
    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    assert diff.max().item() < 0.01
    assert normalized_rmse.item() < 0.002
    assert strict_failures.count_nonzero().item() / out.numel() < 1e-5


@onlyBackends(["cute"])
def test_dense_bfloat16_final_only_stat_handoff_is_accurate() -> None:
    torch.manual_seed(106)
    q, k, v = (
        torch.randn(1, 1, 1024, 64, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4",
        "cute_flash_persistent": False,
        "cute_flash_softmax_disc": False,
        "cute_flash_stat_transport": "single_final",
        "cute_flash_rescale_threshold": 8.0,
    }
    resolved = cute_flash.resolve_flash_config(
        64,
        8,
        config,
        dtype=torch.bfloat16,
        standard_dense_output=True,
    )

    bound = _dense_attention_output.bind((q, k, v))
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    out = bound(q, k, v)
    repeated = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert resolved.stat_transport == "single_final"
    assert "flash_s_corr_prod_phase" in code
    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.02)


@onlyBackends(["cute"])
def test_persistent_legacy_degree1_final_only_is_accurate_across_work() -> None:
    torch.manual_seed(105)
    q = torch.randn(1, 1, 262_144, 64, dtype=torch.float16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q.mul_(2.0)
    k.mul_(2.0)
    config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4_2cta",
        "cute_flash_exp2_packet": _DEG1_PACKET,
        "cute_flash_e2e_schedule": "16/8",
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_kv_stage": 2,
        "cute_flash_persistent": True,
        "cute_flash_stat_transport": "single_final",
        "cute_flash_wait_hint": 0,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_rescale_chunk_cols": 8,
    }

    bound = _dense_attention_output.bind((q, k, v))
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    out = bound(q, k, v)
    repeated = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    diff = (out.float() - expected.float()).abs()
    strict_failures = diff > (0.002 + 0.01 * expected.float().abs())
    normalized_rmse = torch.sqrt((diff * diff).mean(dtype=torch.float64)) / torch.sqrt(
        (expected.float() * expected.float()).mean(dtype=torch.float64)
    )

    assert f"_helion_flash_runtime_abi = {cute_flash._FLASH_RUNTIME_ABI}" in code
    assert "degree1=True" in code
    assert "while flash_tile_id <" in code
    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    assert diff.max().item() < 0.01
    assert normalized_rmse.item() < 0.002
    assert strict_failures.count_nonzero().item() / out.numel() < 1e-5


def test_persistent_final_only_stat_transport_carries_terminal_phase() -> None:
    source = _emit_dense_single_stat_source(final_only_stat_pipeline=True)
    assert source != _emit_dense_single_stat_source()

    stage0_start = source.index("flash_s_corr_prod_phase = cutlass.Int32(0)")
    stage1_start = source.index(
        "flash_s_corr_prod_phase = cutlass.Int32(0)", stage0_start + 1
    )
    softmax0 = source[stage0_start:stage1_start]
    terminal_wait = (
        "_helion_flash_rt.mbar_spin_wait(flash_s0_corr_empty_ptr + 0, "
        "flash_s_corr_prod_phase, 10000000)"
    )
    assert softmax0.count(terminal_wait) == 1
    assert softmax0.count("flash_s_corr_prod_phase ^= 1") == 1
    wait_index = softmax0.index(terminal_wait)
    phase_index = softmax0.index("flash_s_corr_prod_phase ^= 1", wait_index)
    rowsum_index = softmax0.index(
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_row_sum",
        phase_index,
    )
    assert wait_index < phase_index < rowsum_index

    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    for stage in (0, 1):
        empty = f"cute.arch.mbarrier_arrive(flash_s{stage}_corr_empty_ptr + 0)"
        assert correction.count(empty) == 1


@pytest.mark.parametrize(
    ("packet", "softmax_disc", "rescale_threshold"),
    ((_DEG1_SHORT_PACKET, False, 8.0), ("8x2", True, 8.0), ("8x2", False, 0.0)),
)
def test_dense_single_stat_unsupported_schedules_use_conservative_protocol(
    packet: str, softmax_disc: bool, rescale_threshold: float
) -> None:
    source = _emit_dense_single_stat_source(
        pipeline_family="fa4" if softmax_disc else "fa4_2cta",
        packet=packet,
        softmax_disc=softmax_disc,
        rescale_threshold=rescale_threshold,
    )

    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(0)") == 2
    assert "flash_s_corr_prod_phase = cutlass.Int32(1)" not in source
    correction = source[source.index("(warp_idx >= 8) & (warp_idx < 12):") :]
    first_empty = correction.index(
        "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)"
    )
    first_ready = correction.index(
        "_helion_flash_rt.named_barrier_wait_unaligned(3 + warp_idx % 4, 64)"
    )
    assert first_empty < first_ready


def test_dense_ring2_emits_two_slot_stat_protocol() -> None:
    def emit(stat_transport: str) -> str:
        with patch.dict(os.environ, {}, clear=True):
            config = cute_flash.resolve_flash_config(
                64,
                4,
                _manual_config(
                    "8x2",
                    **{
                        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
                        cute_flash.FLASH_STAT_TRANSPORT_KEY: stat_transport,
                        cute_flash.FLASH_PERSISTENT_KEY: True,
                    },
                ),
                dtype=torch.float16,
                is_causal=False,
                standard_dense_output=True,
            )
        body = cute_flash.emit_flash_fa4_device_body(
            cast("DeviceFunction", None),
            head_dim=64,
            num_kv=4,
            sequence_extent=512,
            num_bh=64,
            total_tiles=128,
            cfg=config,
            has_lse=False,
            io_dtype="cutlass.Float16",
            score_plan=dense_score_plan(64),
            target_device_capability=(10, 3),
        )
        return ast.unparse(ast.Module(body=body, type_ignores=[]))

    ring2 = emit("ring2")
    single = emit("single")

    assert "flash_s_corr_prod_index = cutlass.Int32(0)" in ring2
    assert "flash_s_corr_cons_index = cutlass.Int32(0)" in ring2
    assert "flash_s_corr_prod_index ^= 1" in ring2
    assert "flash_s_corr_cons_index ^= 1" in ring2
    assert "if flash_s_corr_prod_index == 0:" in ring2
    assert "flash_scale_t[flash_s_corr_prod_index, 0, flash_local_tidx]" in ring2
    assert "flash_scale_t[flash_s_corr_cons_index, 0, flash_local_tidx]" in ring2
    assert "flash_s_corr_prod_index" not in single
    assert "flash_s_corr_cons_index" not in single
    assert ring2 != single


@onlyBackends(["cute"])
def test_hybrid_packet_runtime_matches_sdpa_at_causal_boundaries() -> None:
    torch.manual_seed(101)
    q, k, v = (
        torch.randn(1, 2, 1024, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )
    q[:, :, 0, :] = 4.0
    k[:, :, 0, :] = 4.0
    q[:, :, 128, :] = -4.0
    k[:, :, 128, :] = -4.0

    code, out = code_and_output(
        _causal_attention_output,
        (q, k, v),
        **_hybrid_runtime_config(),
    )
    assert "degree1=True" in code
    assert "degree2=True" in code
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.02)
    boundary_rows = torch.tensor([0, 127, 128, 255, 1023], device=DEVICE)
    torch.testing.assert_close(
        out[:, :, boundary_rows],
        expected[:, :, boundary_rows],
        atol=0.05,
        rtol=0.02,
    )


@onlyBackends(["cute"])
def test_hybrid_packet_runtime_matches_sdpa_at_long_causal_threshold() -> None:
    torch.manual_seed(102)
    q, k, v = (
        torch.randn(1, 1, 65_536, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )
    q[:, :, 0, :] = 4.0
    k[:, :, 0, :] = 4.0
    q[:, :, 32_768, :] = -4.0
    k[:, :, 32_768, :] = -4.0

    _code, out = code_and_output(
        _causal_attention_output,
        (q, k, v),
        **_hybrid_runtime_config(),
    )
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.02)


@onlyBackends(["cute"])
def test_transferred_degree2_packet_matches_sdpa_beyond_previous_seed_cap() -> None:
    torch.manual_seed(103)
    q, k, v = (
        torch.randn(1, 1, 1_048_576, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )

    config = next(
        seed.config
        for seed in cute_flash.flash_attention_seed_configs(
            64,
            8192,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _DEG2_PACKET
    )
    code, out = code_and_output(_causal_attention_output, (q, k, v), **config)
    assert "degree1=True" not in code
    assert "degree2=True" in code
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.02)


def test_degree2_packet_normalizes_by_effective_schedule() -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(),
            dtype=torch.float16,
            is_causal=True,
        )
    assert resolved.exp2_packet == _DEG2_PACKET
    assert resolved.e2e_schedule == "16/6"
    assert resolved.masked_e2e_schedule == "16/6"


@pytest.mark.parametrize(
    ("packet", "num_kv", "freq", "res"),
    (
        (_DEG1_SHORT_PACKET, 256, 8, 2),
        (_DEG1_SHORT_PACKET, 512, 8, 2),
        (_DEG1_SHORT_PACKET, 1024, 8, 2),
        (_DEG1_PACKET, 2048, 16, 8),
    ),
)
def test_degree1_packet_requires_standard_dense_output(
    packet: str, num_kv: int, freq: int, res: int
) -> None:
    overrides = _manual_config(
        packet,
        **{cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
    )
    with patch.dict(os.environ, {}, clear=True):
        nonstandard = cute_flash.resolve_flash_config(
            64,
            num_kv,
            overrides,
            dtype=torch.float16,
            is_causal=False,
        )
        standard = cute_flash.resolve_flash_config(
            64,
            num_kv,
            overrides,
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
        causal = cute_flash.resolve_flash_config(
            64,
            num_kv,
            overrides,
            dtype=torch.float16,
            is_causal=True,
        )

    assert nonstandard.exp2_packet == "8x2"
    assert causal.exp2_packet != packet
    assert standard.exp2_packet == packet
    assert standard.masked_e2e_schedule == "inherit"
    assert standard.masked_e2e_freq == standard.e2e_freq == freq
    assert standard.masked_e2e_res == standard.e2e_res == res


@pytest.mark.parametrize(
    ("packet", "num_kv", "expected"),
    (
        (_DEG1_PACKET, 256, _DEG1_SHORT_PACKET),
        (_DEG1_PACKET, 512, _DEG1_SHORT_PACKET),
        (_DEG1_PACKET, 1024, _DEG1_SHORT_PACKET),
        (_DEG1_SHORT_PACKET, 2048, _DEG1_PACKET),
    ),
)
def test_degree1_packet_normalizes_to_shape_specific_family(
    packet: str, num_kv: int, expected: str
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            num_kv,
            _manual_config(
                packet,
                **{cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
            ),
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
    assert resolved.exp2_packet == expected


def test_hybrid_packet_normalizes_by_effective_schedule() -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(_HYBRID_PACKET),
            dtype=torch.float16,
            is_causal=True,
        )
    assert resolved.exp2_packet == _HYBRID_PACKET
    assert resolved.e2e_schedule == "16/8"
    assert resolved.masked_e2e_schedule == "16/8"


def test_degree2_packet_canonicalizes_dead_cadence_fields() -> None:
    resolved = []
    with patch.dict(os.environ, {}, clear=True):
        for schedule, masked_schedule in (("8/2", "inherit"), ("16/4", "xu")):
            resolved.append(
                cute_flash.resolve_flash_config(
                    64,
                    512,
                    _manual_config(
                        **{
                            cute_flash.FLASH_E2E_SCHEDULE_KEY: schedule,
                            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: masked_schedule,
                            cute_flash.FLASH_E2E_OFFSET_KEY: 14,
                            cute_flash.FLASH_E2E_OFFSET0_KEY: 14,
                        }
                    ),
                    dtype=torch.float16,
                    is_causal=True,
                )
            )

    assert resolved[0] == resolved[1]
    assert resolved[0].e2e_offset == 14
    assert resolved[0].e2e_offset0 == 14


@pytest.mark.parametrize(
    ("head_dim", "dtype", "is_causal", "overrides"),
    (
        pytest.param(64, torch.float16, False, {}, id="dense"),
        pytest.param(128, torch.float16, True, {}, id="hd128"),
        pytest.param(64, torch.bfloat16, True, {}, id="bf16"),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
            id="ws-overlap",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_deep_1cta"},
            id="deep-ring",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_causal"},
            id="two-cta",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_SOFTMAX_DISC_KEY: False},
            id="whole-row",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_DISC_PIPE_KEY: 1},
            id="serial-disc",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_P_STORE_REP_KEY: 32},
            id="p-store-rep32",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_S_LOAD_REP_KEY: 16},
            id="s-load-rep16",
        ),
        pytest.param(
            64,
            torch.float16,
            True,
            {cute_flash.FLASH_E2E_SCHEDULE_KEY: "xu"},
            id="xu-only",
        ),
    ),
)
def test_degree2_packet_incompatible_schedule_normalizes(
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    overrides: dict[str, object],
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            head_dim,
            512,
            _manual_config(**overrides),
            dtype=dtype,
            is_causal=is_causal,
        )
    assert resolved.exp2_packet == "1x1"


@pytest.mark.parametrize("packet", tuple(cute_flash._FLASH_EXP2_PACKET_PARAMS))
def test_current_packets_keep_existing_normalization(packet: str) -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(**{cute_flash.FLASH_EXP2_PACKET_KEY: packet}),
            dtype=torch.float16,
            is_causal=True,
        )
    assert resolved.exp2_packet == packet


@pytest.mark.parametrize(
    ("packet", "schedule"),
    (
        (_DEG2_PACKET, "16/6"),
        (_HYBRID_PACKET, "16/8"),
    ),
)
def test_manual_packet_config_spec_is_fixed_not_searched(
    packet: str, schedule: str
) -> None:
    with patch.dict(
        os.environ,
        {"HELION_CUTE_FLASH_EXP2_PACKET": packet},
        clear=True,
    ):
        spec = ConfigSpec(
            backend=CuteBackend(),
            target_device_capability=(10, 0),
            device=torch.device("cpu"),
            num_sm=148,
        )
        for block_id, size_hint in enumerate((1, 128, 128)):
            spec.block_sizes.append(
                BlockSizeSpec(block_id=block_id, size_hint=size_hint)
            )
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=512,
            dtype=torch.float16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            is_causal=True,
        )

        fragment = spec._flat_fields()[cute_flash.FLASH_EXP2_PACKET_KEY]
        assert isinstance(fragment, EnumFragment)
        assert fragment.choices[0] == packet
        assert set(fragment.choices) == {
            *cute_flash._FLASH_EXP2_PACKET_PARAMS,
            *cute_flash._FLASH_MANUAL_EXP2_PACKET_PARAMS,
        }
        assert fragment.search_choices == (packet,)

        e2e_fragment = spec._flat_fields()[cute_flash.FLASH_E2E_SCHEDULE_KEY]
        assert isinstance(e2e_fragment, EnumFragment)
        assert e2e_fragment.choices == (schedule,)
        assert e2e_fragment.search_choices == (schedule,)
        masked_fragment = spec._flat_fields()[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY]
        assert isinstance(masked_fragment, EnumFragment)
        assert masked_fragment.choices == (schedule,)
        assert masked_fragment.search_choices == (schedule,)

        config = spec.default_config()
        spec.normalize(config)
        assert config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == packet


def test_dense_degree1_environment_builds_canonical_fragments() -> None:
    env = {
        "HELION_CUTE_FLASH_PIPELINE_FAMILY": "fa4_2cta",
        "HELION_CUTE_FLASH_EXP2_PACKET": _DEG1_PACKET,
    }
    with patch.dict(os.environ, env, clear=True):
        fragments = cute_flash.flash_autotune_fragments(
            64,
            2048,
            standard_dense_output=True,
        )
        nonstandard = cute_flash.flash_autotune_fragments(64, 2048)

    assert fragments[cute_flash.FLASH_EXP2_PACKET_KEY].search_choices == (_DEG1_PACKET,)
    assert fragments[cute_flash.FLASH_E2E_SCHEDULE_KEY].choices == ("16/8",)
    assert fragments[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY].choices == ("inherit",)
    assert nonstandard[cute_flash.FLASH_EXP2_PACKET_KEY].choices[0] == "8x2"
