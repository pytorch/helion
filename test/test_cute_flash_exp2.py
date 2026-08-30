from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
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
from helion._compiler.cute.flash_policy import registered_flash_target_policies
from helion._compiler.cute.flash_tuning import FlashCausalTuningPolicy
from helion._compiler.cute.flash_tuning import FlashSoftmaxLowering
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
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
_CAUSAL_HD128_RESIDENT_PACKET = (
    "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"
)


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


@helion.kernel(backend="cute", static_shapes=True)
def _pointwise_attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    is_causal: hl.constexpr,
    output_epilogue: hl.constexpr,
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
            if is_causal:
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
        result = acc / l_i[:, :, None]
        if output_epilogue == "relu":
            result = torch.relu(result)
        elif output_epilogue == "relu_after_bf16":
            result = torch.relu(result.to(out.dtype)).to(torch.float32)
        elif output_epilogue == "abs":
            result = torch.abs(result)
        if output_epilogue == "relu_to_fp16":
            out[tile_b, tile_m, :] = torch.relu(result).to(torch.float16)
        else:
            out[tile_b, tile_m, :] = result.to(out.dtype)
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


def _causal_hd128_resident_seed(num_kv: int) -> helion.Config:
    return next(
        seed
        for seed in cute_flash.flash_attention_seed_configs(
            128,
            num_kv,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY)
        == _CAUSAL_HD128_RESIDENT_PACKET
    )


def _emit_causal_hd128_resident(
    *, num_kv: int = 1024, sequence_extent: int = 131_072
) -> ast.Module:
    seed = _causal_hd128_resident_seed(num_kv)
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            128,
            num_kv,
            seed.config,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=128,
        num_kv=num_kv,
        sequence_extent=sequence_extent,
        num_bh=1,
        total_tiles=sequence_extent // 256,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.BFloat16",
        score_plan=causal_score_plan(128),
    )
    return ast.Module(body=body, type_ignores=[])


def _emit_output_epilogue_route_source(route: str, output_epilogue: str | None) -> str:
    config_values: dict[str, object]
    env: dict[str, str] = {}
    num_kv = 8
    sequence_extent = 1024
    total_tiles = 4
    if route == "ws_legacy":
        config_values = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_S_STAGE_KEY: 1,
        }
    elif route == "ws_two_warpgroup":
        config_values = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_S_STAGE_KEY: 2,
        }
    else:
        use_2cta = "2cta" in route
        if use_2cta:
            num_kv = 256
            sequence_extent = 32_768
            total_tiles = 4_096
        if "unscoped_handoff" in route:
            env["HELION_CUTE_FLASH_SCOPED_CORR_EPI"] = "0"
        elif "scoped_handoff" in route:
            env["HELION_CUTE_FLASH_SPLIT_CORR_EPILOGUE_HANDOFF"] = "0"
        config_values = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: ("fa4_2cta" if use_2cta else "fa4"),
            cute_flash.FLASH_EPI_TMA_KEY: "tma" in route,
            cute_flash.FLASH_EPI_STG_KEY: "stg" in route,
        }
    with patch.dict(os.environ, env, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            num_kv,
            config_values,
            dtype=torch.bfloat16,
            standard_dense_output=True,
        )
        common: dict[str, object] = {
            "head_dim": 64,
            "num_kv": num_kv,
            "cfg": config,
            "has_lse": False,
            "io_dtype": "cutlass.BFloat16",
            "score_plan": dense_score_plan(64),
        }
        if output_epilogue is not None:
            common["output_epilogue"] = output_epilogue
        if route == "ws_legacy":
            body = cute_flash.emit_flash_device_body(
                cast("DeviceFunction", None),
                **common,
            )
        elif route == "ws_two_warpgroup":
            body = cute_flash.emit_flash_ws_device_body(
                cast("DeviceFunction", None),
                **common,
            )
        else:
            body = cute_flash.emit_flash_fa4_device_body(
                cast("DeviceFunction", None),
                sequence_extent=sequence_extent,
                num_bh=64 if use_2cta else 1,
                total_tiles=total_tiles,
                **common,
            )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize(
    ("output_epilogue", "expected_flash"),
    (
        ("identity", True),
        ("relu", True),
        ("relu_to_fp16", False),
        ("relu_after_bf16", False),
        ("abs", False),
    ),
)
def test_bfloat16_output_epilogue_flash_matcher(
    is_causal: bool,
    output_epilogue: str,
    expected_flash: bool,
) -> None:
    args = tuple(
        torch.empty(
            8,
            32,
            262_144,
            64,
            dtype=torch.bfloat16,
            device="meta",  # @ignore-device-lint
        )
        for _ in range(3)
    )
    bound = _pointwise_attention_output.bind(
        (*args, hl.constexpr(is_causal), hl.constexpr(output_epilogue))
    )
    spec = bound.config_spec

    assert spec.cute_flash_search_enabled is expected_flash
    if not expected_flash:
        return
    assert spec._cute_flash_standard_causal_output is is_causal
    assert spec._cute_flash_standard_dense_output is not is_causal
    assert spec._cute_flash_output_requires_tma
    seeds = [*spec.compiler_seed_configs, *spec.autotune_seed_configs()]
    assert seeds
    for seed in seeds:
        resolved = spec._resolve_cute_flash_config(seed.config)
        assert resolved.epi_tma
        assert not resolved.epi_stg


def test_relu_output_epilogue_is_bfloat16_only() -> None:
    args = tuple(
        torch.empty(
            1,
            1,
            1024,
            64,
            dtype=torch.float16,
            device="meta",  # @ignore-device-lint
        )
        for _ in range(3)
    )
    bound = _pointwise_attention_output.bind(
        (*args, hl.constexpr(False), hl.constexpr("relu"))
    )
    assert not bound.config_spec.cute_flash_search_enabled


@pytest.mark.parametrize("auxiliary", ("lse", "tensor_bias"))
def test_relu_output_epilogue_rejects_auxiliary_attention(auxiliary: str) -> None:
    from test.test_cute_backend import cute_biased_attention
    from test.test_cute_backend import cute_dense_attention_with_lse

    sequence_length = 1024
    target = torch.empty(
        1,
        1,
        sequence_length,
        64,
        dtype=torch.bfloat16,
        device="meta",  # @ignore-device-lint
    )
    if auxiliary == "lse":
        fixture = cute_dense_attention_with_lse
        args = (target, target, target)
    else:
        fixture = cute_biased_attention
        bias = torch.empty(
            1,
            1,
            sequence_length,
            sequence_length,
            dtype=torch.bfloat16,
            device="meta",  # @ignore-device-lint
        )
        args = (target, target, target, bias)

    # Each clone has an independent bind cache. Patching the recognized output
    # epilogue therefore tests the policy on the exact same valid auxiliary graph.
    identity_fixture = helion.kernel(backend="cute", static_shapes=True)(fixture.fn)
    relu_fixture = helion.kernel(backend="cute", static_shapes=True)(fixture.fn)
    assert identity_fixture.bind(args).config_spec.cute_flash_search_enabled
    with patch.object(
        cute_flash,
        "_flash_store_value_output_epilogue",
        return_value="relu",
    ):
        assert not relu_fixture.bind(args).config_spec.cute_flash_search_enabled


@pytest.mark.parametrize(
    ("route", "staged_helper"),
    (
        ("ws_legacy", None),
        ("ws_two_warpgroup", None),
        ("fa4_direct", None),
        ("fa4_tma", "fa4_correction_epilogue_to_smem_scoped"),
        ("fa4_stg", "fa4_correction_epilogue_to_smem_scoped"),
        (
            "fa4_tma_scoped_handoff",
            "fa4_correction_epilogue_handoff_to_smem_scoped",
        ),
        ("fa4_tma_unscoped_handoff", "fa4_correction_epilogue_handoff_to_smem"),
        ("fa4_tma_2cta", "fa4_correction_epilogue_to_smem_scoped_2cta"),
        (
            "fa4_tma_2cta_scoped_handoff",
            "fa4_correction_epilogue_handoff_to_smem_scoped_2cta",
        ),
    ),
)
def test_relu_output_epilogue_is_emitted_for_every_store_route(
    route: str, staged_helper: str | None
) -> None:
    default = _emit_output_epilogue_route_source(route, None)
    identity = _emit_output_epilogue_route_source(route, "identity")
    relu = _emit_output_epilogue_route_source(route, "relu")

    assert identity == default
    assert "relu_fragment_inplace" not in identity
    assert "relu_output=True" not in identity
    if staged_helper is not None:
        assert f"_helion_flash_rt.{staged_helper}(" in identity
        assert f"_helion_flash_rt.{staged_helper}(" in relu
        assert "relu_output=True" in relu
    else:
        assert "_helion_flash_rt.relu_fragment_inplace(flash_reg" in relu


def test_flash_relu_fragment_matches_nan_and_signed_zero_semantics() -> None:
    source = inspect.getsource(_flash_runtime.relu_fragment_inplace)
    assert "value != value" in source
    assert "cute.where(value > 0.0, value, 0.0)" in source
    assert "fmax" not in source


@pytest.mark.parametrize(
    ("is_causal", "sequence_length", "pipeline_family", "epi_tma"),
    (
        (True, 8192, "fa4", False),
        (True, 8192, "fa4", True),
        (False, 32_768, "fa4_2cta", True),
    ),
)
@onlyBackends(["cute"])
def test_bfloat16_relu_output_epilogue_runtime_routes(
    is_causal: bool,
    sequence_length: int,
    pipeline_family: str,
    epi_tma: bool,
) -> None:
    torch.manual_seed(109)
    q = torch.randn(1, 1, sequence_length, 64, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.full_like(q, -1.0)
    v[..., 0] = float("nan")
    config = {
        "block_sizes": [1, 128, 128],
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: pipeline_family,
        cute_flash.FLASH_PERSISTENT_KEY: False,
        cute_flash.FLASH_EPI_TMA_KEY: epi_tma,
        cute_flash.FLASH_EPI_STG_KEY: False,
    }

    code, out = code_and_output(
        _pointwise_attention_output,
        (q, k, v, hl.constexpr(is_causal), hl.constexpr("relu")),
        **config,
    )
    expected = torch.relu(
        torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    )

    assert "relu_fragment_inplace" in code or "relu_output=True" in code
    assert torch.isnan(out[..., 0]).all()
    assert not torch.signbit(out[..., 1:]).any()
    torch.testing.assert_close(
        out,
        expected,
        atol=0.05,
        rtol=0.02,
        equal_nan=True,
    )


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
    seed_capability: tuple[int, int] | None = None,
    has_lse: bool = False,
    score_plan: AttentionScorePlan | None = None,
    num_kv: int = 512,
    config_overrides: dict[str, object] | None = None,
) -> str:
    if score_plan is None:
        score_plan = causal_score_plan(64)
    if seed_capability is None:
        seed_capability = capability
    with patch.dict(os.environ, {}, clear=True):
        seed = cute_flash.flash_attention_seed_config(
            64,
            num_kv,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=seed_capability,
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
            standard_causal_output=True,
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
    seed_capability: tuple[int, int] | None = None,
    has_lse: bool = False,
    score_plan: AttentionScorePlan | None = None,
    num_kv: int = 256,
    config_overrides: dict[str, object] | None = None,
) -> str:
    if score_plan is None:
        score_plan = dense_score_plan(64)
    if seed_capability is None:
        seed_capability = capability
    with patch.dict(os.environ, {}, clear=True):
        seed = cute_flash.flash_attention_seed_config(
            64,
            num_kv,
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
            target_device_capability=seed_capability,
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


def _assert_source_markers_in_order(source: str, markers: tuple[str, ...]) -> None:
    offsets = [source.index(marker) for marker in markers]
    assert offsets == sorted(offsets)


def test_causal_split_softmax_segments_require_proof() -> None:
    with pytest.raises(AssertionError, match="requires a proven"):
        cute_flash._flash_causal_split_softmax_segments(
            "flash_bound",
            "1",
            split_range_proof=CausalRangeProof(False, "not proven"),
        )
    masked, unmasked = cute_flash._flash_causal_split_softmax_segments(
        "flash_bound",
        "1",
        split_range_proof=CausalRangeProof(True, "proven"),
    )
    assert masked.loop_bound == "flash_bound - flash_m_tile1"
    assert masked.kv_expr == ("flash_bound - cutlass.Int32(1) - flash_kv_mask_iter")
    assert masked.not_first_condition == "flash_kv_mask_iter != 0"
    assert not masked.continues_previous_segment
    assert unmasked.loop_bound == "flash_m_tile1"
    assert unmasked.kv_expr == (
        "flash_m_tile1 - cutlass.Int32(1) - flash_kv_unmask_iter"
    )
    assert unmasked.not_first_condition == ("flash_kv_unmask_iter >= cutlass.Int32(0)")
    assert unmasked.continues_previous_segment


def test_online_softmax_codegen_preserves_target_alpha_schedules() -> None:
    b200_source = _emit_causal_resident_native_source(
        capability=(10, 0),
        seed_capability=(10, 3),
        num_kv=512,
    )
    b200_masked_start = b200_source.index("for flash_kv_mask_iter")
    b200_unmasked_start = b200_source.index(
        "for flash_kv_unmask_iter", b200_masked_start
    )
    b200_masked = b200_source[b200_masked_start:b200_unmasked_start]
    b200_markers = (
        "mbar_spin_wait(flash_s_full_ptr + 0",
        "flash_s_full_phase ^= 1",
        "flash_scale_t[flash_s_corr_prod_index, 0, flash_local_tidx] = flash_alpha",
        "flash_minus_max_scale =",
        "fa4_disc_exp_convert_store_pipe_causal",
        "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum",
    )
    _assert_source_markers_in_order(b200_masked, b200_markers)

    gb300_source = _emit_causal_resident_native_source(
        capability=(10, 3),
        num_kv=1024,
    )
    gb300_masked_start = gb300_source.index("for flash_kv_mask_iter")
    gb300_unmasked_start = gb300_source.index(
        "for flash_kv_unmask_iter", gb300_masked_start
    )
    gb300_masked = gb300_source[gb300_masked_start:gb300_unmasked_start]
    gb300_markers = (
        "mbar_spin_wait(flash_s_full_ptr + 0",
        "flash_s_full_phase ^= 1",
        "flash_minus_max_scale =",
        "flash_scale_t[0 * 128 + flash_local_tidx] = flash_alpha",
        "resident_softmax_value_graph",
        "flash_s_corr_prod_phase ^= 1",
    )
    _assert_source_markers_in_order(gb300_masked, gb300_markers)


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


def test_registered_causal_shape_selects_requested_resident_lowering() -> None:
    target_policy = get_flash_target_policy((10, 3))
    extra_policy = FlashCausalTuningPolicy(
        num_kv=768,
        kv_stage=4,
        e2e_offset=2,
        e2e_offset0=3,
        softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
        softmax_regs=184,
    )
    extended_tuning = dataclasses.replace(
        target_policy.tuning,
        causal_policies=(*target_policy.tuning.causal_policies, extra_policy),
    )
    base_fragments = cute_flash.flash_autotune_fragments(
        64,
        768,
        dtype=torch.float16,
        is_causal=True,
        standard_causal_output=True,
    )
    with patch(
        "helion._compiler.cute.cute_flash.get_flash_target_policy",
        return_value=dataclasses.replace(target_policy, tuning=extended_tuning),
    ):
        target_fragments = cute_flash.flash_autotune_fragments(
            64,
            768,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 3),
        )
        source = _emit_causal_resident_native_source(num_kv=768)

    base_kv_stage = cast("EnumFragment", base_fragments[cute_flash.FLASH_KV_STAGE_KEY])
    target_kv_stage = cast(
        "EnumFragment", target_fragments[cute_flash.FLASH_KV_STAGE_KEY]
    )
    assert extra_policy.kv_stage in base_kv_stage.choices
    assert target_kv_stage.fingerprint() == base_kv_stage.fingerprint()
    assert target_kv_stage.search_choices == base_kv_stage.search_choices
    assert "resident_softmax_value_graph" in source


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
            standard_causal_output=True,
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
        assert isinstance(loop.iter, ast.Call)
        assert ast.unparse(loop.iter.func) == "cutlass.range"
        assert len(loop.iter.keywords) == 1
        assert loop.iter.keywords[0].arg == "unroll"
        assert ast.literal_eval(loop.iter.keywords[0].value) == 1
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


def test_sm103_scaled_all_xu_codegen_requires_exact_rescale_threshold() -> None:
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


def test_dense_target_seed_match_ignores_conflicting_environment() -> None:
    with patch.dict(os.environ, {}, clear=True):
        dense_seed = cute_flash.flash_attention_seed_config(
            64,
            256,
            standard_dense_output=True,
            target_device_capability=(10, 3),
        )
    assert dense_seed is not None
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_WAIT_HINT": "0"}, clear=True):
        dense_cfg = cute_flash.resolve_flash_config(
            64,
            256,
            dense_seed.config,
            standard_dense_output=True,
        )
        policy = get_flash_target_policy((10, 3)).tuning.dense_policy(256)
        assert dense_cfg.wait_hint == 10000000
        assert cute_flash._flash_dense_target_seed_matches(dense_cfg, policy)


def test_dense_resident_value_graph_codegen_and_barrier_protocol() -> None:
    source = _emit_dense_resident_value_graph_source()
    module = ast.parse(source)
    value_graph_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resident_softmax_value_graph"
    ]

    assert len(value_graph_calls) == 2
    parameter_names = tuple(
        inspect.signature(_flash_runtime.resident_softmax_value_graph).parameters
    )
    for call in value_graph_calls:
        arguments = dict(zip(parameter_names, call.args, strict=False))
        assert ast.unparse(arguments["stats_empty_ptr_stage"]).endswith(
            "_corr_empty_ptr + 0"
        )
        assert ast.unparse(arguments["stats_empty_phase"]) == "flash_s_corr_prod_phase"
        assert ast.unparse(arguments["row_sum_init"]) == "flash_row_sum * flash_alpha"
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


@pytest.mark.parametrize(
    ("capability", "num_kv"),
    tuple(
        (capability, shape_policy.num_kv)
        for capability, target_policy in registered_flash_target_policies()
        for tuning in target_policy.tunings
        if tuning.workload.head_dim == 64 and tuning.workload.dtype.value == "float16"
        for shape_policy in tuning.dense_policies
        if shape_policy.softmax_lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
    ),
)
def test_dense_resident_value_graph_selects_registered_shapes(
    capability: tuple[int, int], num_kv: int
) -> None:
    source = _emit_dense_resident_value_graph_source(
        capability=capability,
        num_kv=num_kv,
    )

    assert "resident_softmax_value_graph" in source


def test_dense_resident_value_graph_gate_preserves_fallbacks() -> None:
    base_plan = dense_score_plan(64)
    modified_plan = dataclasses.replace(
        base_plan,
        modifiers=(AttentionScoreModifier(SOFTCAP_KIND, value_log2=2.0),),
    )
    fallback_sources = (
        _emit_dense_resident_value_graph_source(
            capability=(10, 0), seed_capability=(10, 3)
        ),
        _emit_dense_resident_value_graph_source(
            capability=(999, 999), seed_capability=(10, 3)
        ),
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
                probability_log2_shift=0,
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


def test_dense_probability_shift_is_rejected_before_fp16_overflow() -> None:
    target_policy = get_flash_target_policy((10, 3))
    dense_policy = target_policy.tuning.dense_policy(256)
    assert dense_policy is not None
    with pytest.raises(ValueError, match="exceed fp16 range"):
        dataclasses.replace(dense_policy, probability_log2_shift=16)


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
    assert "flash_softmax.reset()" not in source
    assert source.count("update_row_max_masked(tLDrS.load(), True)") == 2
    assert source.count("update_row_max_masked(tLDrS.load(), False)") == 2
    assert source.count("update_row_max_precomputed(flash_hw_row_max, False)") == 2
    assert source.count("flash_softmax.scale_subtract_rowmax") == 6
    assert source.count("flash_softmax.apply_exp2_convert") == 6
    assert source.count("flash_tSrP_f32 = cute.make_rmem_tensor") == 6
    assert source.count("dtype=cutlass.Float16), tLDrS.layout") == 6
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
    stats_wait = (
        "_helion_flash_rt.mbar_spin_wait(flash_s0_corr_empty_ptr + 0, "
        "flash_s_corr_prod_phase, 0)"
    )
    stats_acquire = masked.index(stats_wait, early_pack)
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
        _emit_causal_resident_native_source(
            capability=(10, 0), seed_capability=(10, 3)
        ),
        _emit_causal_resident_native_source(
            capability=(999, 999), seed_capability=(10, 3)
        ),
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
    # Deadlock soak: repeated launches race the correction-warp handshake.
    for _ in range(7):
        repeated = compiled(q, k, v)
    torch.cuda.synchronize()
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize(
    ("num_kv", "repeat_count"),
    (
        (1024, 7),
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
    if (
        shape_policy is None
        or shape_policy.softmax_lowering
        is not FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
    ):
        pytest.skip("causal resident native softmax is unsupported on this target")

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


def test_causal_target_seed_match_ignores_conflicting_environment() -> None:
    with patch.dict(os.environ, {}, clear=True):
        causal_seed = cute_flash.flash_attention_seed_config(
            64,
            512,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 3),
        )
    assert causal_seed is not None
    with patch.dict(
        os.environ, {"HELION_CUTE_FLASH_WAIT_HINT": "10000000"}, clear=True
    ):
        causal_cfg = cute_flash.resolve_flash_config(
            64,
            512,
            causal_seed.config,
            is_causal=True,
            standard_causal_output=True,
        )
        policy = get_flash_target_policy((10, 3)).tuning.causal_policy(512)
        assert causal_cfg.wait_hint == 0
        assert cute_flash._flash_causal_resident_native_seed_matches(causal_cfg, policy)

    effective = cute_flash._flash_resident_softmax_config(causal_cfg)
    assert causal_cfg.exp2_packet == _DEG2_PACKET
    assert causal_cfg.stat_transport == "ring2"
    assert effective.exp2_packet == "1x1"
    assert effective.stat_transport == "single"
    assert effective.exp2_impl == "xu"


def test_bfloat16_hd128_degree2_packet_emits_dense_pass2_arguments() -> None:
    seed = next(
        seed
        for seed in cute_flash.flash_attention_seed_configs(
            128,
            2048,
            dtype=torch.bfloat16,
            standard_dense_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _DEG2_PACKET
    )
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            128,
            2048,
            seed.config,
            dtype=torch.bfloat16,
            standard_dense_output=True,
        )
    assert config.pipeline_family == "fa4_2cta"
    assert config.use_2cta_instrs
    assert config.softmax_disc
    assert config.stat_transport == "ring2"
    assert config.exp2_packet == _DEG2_PACKET
    assert config.e2e_schedule == "16/6"

    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=128,
        num_kv=2048,
        sequence_extent=262_144,
        num_bh=64,
        total_tiles=32_768,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.BFloat16",
        score_plan=dense_score_plan(128),
    )
    module = ast.Module(body=body, type_ignores=[])
    pass2_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fa4_disc_exp_convert_store_pipe"
    ]
    assert len(pass2_calls) == 2
    assert [ast.literal_eval(call.args[10]) for call in pass2_calls] == [
        config.e2e_offset0,
        config.e2e_offset,
    ]
    for call in pass2_calls:
        assert ast.literal_eval(call.args[8]) == 16
        assert ast.literal_eval(call.args[9]) == 6
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
        assert keywords == {"pair_batch": 8, "emu_batch": 3, "degree2": True}


def test_causal_hd128_resident_packet_has_structural_search_envelope() -> None:
    for num_kv in (2, 512, 1024, 1536, 2048, 4096, 6144):
        fragments = cute_flash.flash_autotune_fragments(
            128,
            num_kv,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            pipeline_family_override="fa4",
        )
        packet = fragments[cute_flash.FLASH_EXP2_PACKET_KEY]
        assert isinstance(packet, EnumFragment)
        assert packet.choices == ("1x1", _CAUSAL_HD128_RESIDENT_PACKET)
        assert packet.search_choices == ("1x1", _CAUSAL_HD128_RESIDENT_PACKET)

    for num_kv in (1, 513, 2049):
        fragments = cute_flash.flash_autotune_fragments(
            128,
            num_kv,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
        family = fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
        assert isinstance(family, EnumFragment)
        assert family.search_choices == ("ws_overlap",)
        packet = fragments[cute_flash.FLASH_EXP2_PACKET_KEY]
        assert isinstance(packet, EnumFragment)
        assert _CAUSAL_HD128_RESIDENT_PACKET not in packet.choices

    transferred = dict(_causal_hd128_resident_seed(1024).config)
    ordinary = {**transferred, cute_flash.FLASH_EXP2_PACKET_KEY: "1x1"}
    with patch.dict(os.environ, {}, clear=True):
        transferred_config = cute_flash.resolve_flash_config(
            128,
            2048,
            transferred,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
        ordinary_config = cute_flash.resolve_flash_config(
            128,
            2048,
            ordinary,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
    assert transferred_config.exp2_packet == _CAUSAL_HD128_RESIDENT_PACKET
    assert ordinary_config.exp2_packet == "1x1"


def test_causal_hd128_resident_seed_roundtrips_and_is_in_population() -> None:
    required = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
        cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
        cute_flash.FLASH_EXP2_PACKET_KEY: _CAUSAL_HD128_RESIDENT_PACKET,
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
        cute_flash.FLASH_DISC_PIPE_KEY: 2,
        cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
        cute_flash.FLASH_P_STORE_REP_KEY: 16,
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    }
    configs: list[dict[str, object]] = []
    for num_kv in (512, 1024, 1536, 2048, 4096, 6144):
        seed = _causal_hd128_resident_seed(num_kv)
        assert all(seed.config[key] == value for key, value in required.items())
        configs.append(seed.config)

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
            head_dim=128,
            num_kv=num_kv,
            dtype=torch.bfloat16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            is_causal=True,
            standard_causal_output=True,
        )
        spec.compiler_seed_configs = list(
            cute_flash.flash_attention_seed_configs(
                128,
                num_kv,
                dtype=torch.bfloat16,
                is_causal=True,
                standard_causal_output=True,
            )
        )
        config_gen = ConfigGeneration(spec)
        roundtrip = config_gen.unflatten(config_gen.flatten(seed))
        assert config_gen.unflatten(config_gen.flatten(roundtrip)) == roundtrip
        assert any(
            config == roundtrip for _flat, config in config_gen.seed_flat_config_pairs()
        )
        assert all(roundtrip.config[key] == value for key, value in required.items())
    assert all(config == configs[0] for config in configs[1:])


def test_causal_hd128_resident_packet_emits_prefetched_chunk2_route() -> None:
    module = _emit_causal_hd128_resident()
    masked_loops = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "flash_kv_mask_iter"
    ]
    unmasked_loops = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "flash_kv_unmask_iter"
    ]
    assert len(masked_loops) == len(unmasked_loops) == 2
    helper = "fa4_disc_exp_convert_store_resident3_013_prefetch2"
    assert all(helper not in ast.unparse(loop) for loop in masked_loops)
    for loop in unmasked_loops:
        source = ast.unparse(loop)
        calls = [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == helper
        ]
        assert len(calls) == 1
        call = calls[0]
        assert tuple(ast.unparse(arg) for arg in call.args[:3]) == (
            "flash_res_frg0",
            "flash_res_frg1",
            "flash_res_frg3",
        )
        assert (ast.literal_eval(call.args[11]), ast.literal_eval(call.args[12])) == (
            16,
            6,
        )
        assert {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        } == {"pair_batch": 8, "emu_batch": 3, "degree2": True}
        for chunk, fragment in (
            (0, "flash_res_frg0"),
            (1, "flash_res_frg1"),
            (2, "flash_res_rowmax_tmp"),
            (3, "flash_res_frg3"),
        ):
            assert f"[None, {chunk}, None, None], {fragment})" in source
        acquire = source.index("_corr_empty_ptr + flash_s_corr_prod_index")
        rowmax = source.index("flash_res_ld_shape =")
        alpha_store = source.index("] = flash_alpha")
        pass2 = source.index(helper)
        assert acquire < rowmax < alpha_store < pass2

    unproven = _emit_causal_hd128_resident(sequence_extent=524_287)
    assert helper not in ast.unparse(unproven)


def test_causal_hd128_resident_runtime_prefetches_chunk2_after_chunk0() -> None:
    function = _flash_runtime.fa4_disc_exp_convert_store_resident3_013_prefetch2
    source = inspect.getsource(function)
    module = ast.parse(source)
    copies = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cute"
        and node.func.attr == "copy"
    ]
    assert len(copies) == 1
    assert ast.unparse(copies[0].args[1]) == "tLDtS[None, 2, None, None]"

    consume = "_disc_resident_exp_store_rowsum("
    consumes = []
    offset = 0
    while (position := source.find(consume, offset)) >= 0:
        consumes.append(position)
        offset = position + len(consume)
    assert len(consumes) == 4
    prefetch = source.index("cute.copy(")
    pin = source.index("_disc_pin_frag(frg2)")
    load_fence = source.index("cute.arch.fence_view_async_tmem_load()")
    first_release = source.index("mbarrier_arrive(pfor_ptr_stage)")
    final_release = source.index("mbarrier_arrive(pfor2_ptr_stage)")
    assert (
        consumes[0]
        < prefetch
        < pin
        < consumes[1]
        < load_fence
        < consumes[2]
        < first_release
        < consumes[3]
        < final_release
    )


@onlyBackends(["cute"])
def test_causal_hd128_resident_seed_matches_sdpa() -> None:
    torch.manual_seed(107)
    q, k, v = (
        torch.randn(1, 1, 65_536, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    config = _causal_hd128_resident_seed(512).config
    bound = _causal_attention_output.bind((q, k, v))
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    out = bound(q, k, v)
    repeated = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = out.float() - expected.float()
    normalized_rmse = torch.sqrt(
        (diff * diff).mean(dtype=torch.float64)
        / (expected.float() * expected.float()).mean(dtype=torch.float64)
    )

    assert "fa4_disc_exp_convert_store_resident3_013_prefetch2" in code
    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    assert diff.abs().max().item() <= 0.015625
    assert normalized_rmse.item() < 0.003


def test_hybrid_packet_uses_degree1_only_for_unmasked_pass2() -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(_HYBRID_PACKET),
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
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


@pytest.mark.parametrize("num_kv", (1024, 4096, 5120, 6144, 7168, 8192))
def test_bfloat16_hybrid_packet_uses_audited_causal_routes(num_kv: int) -> None:
    config = next(
        seed.config
        for seed in cute_flash.flash_attention_seed_configs(
            64,
            num_kv,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _HYBRID_PACKET
    )
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            num_kv,
            config,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=1,
        total_tiles=num_kv // 2,
        cfg=resolved,
        has_lse=False,
        io_dtype="cutlass.BFloat16",
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
                    cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted",
                    cute_flash.FLASH_EPI_TMA_KEY: True,
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
    structural_seed = next(
        seed
        for seed in cute_flash.flash_attention_seed_configs(
            64,
            260,
            dtype=torch.float16,
            standard_dense_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _DEG2_PACKET
    )
    config = {
        **structural_seed.config,
        cute_flash.FLASH_STAT_TRANSPORT_KEY: "single_final",
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

    assert "degree2=True" in code
    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    assert diff.max().item() < 0.01
    assert normalized_rmse.item() < 0.002
    assert strict_failures.count_nonzero().item() / out.numel() < 1e-5


@onlyBackends(["cute"])
def test_bfloat16_hd128_degree2_seed_is_deterministic_and_matches_sdpa() -> None:
    config = next(
        seed.config
        for seed in cute_flash.flash_attention_seed_configs(
            128,
            256,
            dtype=torch.bfloat16,
            standard_dense_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _DEG2_PACKET
    )
    torch.manual_seed(105)
    first_args = tuple(
        torch.randn(1, 1, 32_768, 128, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    bound = _dense_attention_output.bind(first_args)
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    assert "degree2=True" in code

    for seed, scale in ((105, 1.0), (106, 2.0)):
        torch.manual_seed(seed)
        q, k, v = (
            torch.randn(1, 1, 32_768, 128, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        )
        q.mul_(scale)
        k.mul_(scale)
        out = bound(q, k, v)
        repeated = bound(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        diff = (out.float() - expected.float()).abs()
        normalized_rmse = torch.sqrt(
            (diff * diff).mean(dtype=torch.float64)
            / (expected.float() * expected.float()).mean(dtype=torch.float64)
        )
        assert torch.equal(out, repeated)
        assert torch.isfinite(out).all()
        assert diff.max().item() < 0.05
        assert normalized_rmse.item() < 0.003


@pytest.mark.parametrize(
    ("dtype", "head_dim", "is_causal", "packet", "family", "num_kv"),
    (
        pytest.param(
            torch.bfloat16,
            64,
            False,
            "8x2",
            "fa4_2cta",
            4,
            id="bf16-d64-dense-min",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            True,
            _HYBRID_PACKET,
            "fa4",
            2,
            id="bf16-d64-causal-min",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            False,
            _DEG2_PACKET,
            "fa4_2cta",
            4,
            id="bf16-d128-dense-min",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            True,
            _CAUSAL_HD128_RESIDENT_PACKET,
            "fa4",
            2,
            id="bf16-d128-causal-min",
        ),
        pytest.param(
            torch.float16,
            64,
            True,
            _DEG2_PACKET,
            "fa4",
            2,
            id="fp16-d64-causal-min",
        ),
        # One transferred-seed case at a length with no registered policy keeps
        # runtime coverage for the seed-transfer path without recompiling every
        # policy at a second length.
        pytest.param(
            torch.float16,
            64,
            True,
            _DEG2_PACKET,
            "fa4",
            768,
            id="fp16-d64-causal-unseen",
        ),
    ),
)
@onlyBackends(["cute"])
def test_specialized_flash_family_matches_sdpa_at_boundary_lengths(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    packet: str,
    family: str,
    num_kv: int,
) -> None:
    _run_specialized_flash_family_correctness(
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        packet=packet,
        family=family,
        num_kv=num_kv,
    )


def _run_specialized_flash_family_correctness(
    *,
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    packet: str,
    family: str,
    num_kv: int,
) -> None:
    sequence_extent = num_kv * 128
    configs = cute_flash.flash_attention_seed_configs(
        head_dim,
        num_kv,
        dtype=dtype,
        is_causal=is_causal,
        standard_dense_output=not is_causal,
        standard_causal_output=is_causal,
    )
    config = next(
        (
            seed.config
            for seed in configs
            if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == packet
            and seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == family
        ),
        None,
    )
    if config is None:
        fragments = cute_flash.flash_autotune_fragments(
            head_dim,
            num_kv,
            dtype=dtype,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
            pipeline_family_override=family,
        )
        config = {
            "block_sizes": [1, 128, 128],
            **{key: fragment.default() for key, fragment in fragments.items()},
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_EXP2_PACKET_KEY: packet,
        }

    torch.manual_seed(109)
    q, k, v = (
        torch.randn(
            1,
            1,
            sequence_extent,
            head_dim,
            dtype=dtype,
            device=DEVICE,
        )
        for _ in range(3)
    )
    kernel = _causal_attention_output if is_causal else _dense_attention_output
    bound = kernel.bind((q, k, v))
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    resolved = cute_flash.resolve_flash_config(
        head_dim,
        num_kv,
        active_config.config,
        dtype=dtype,
        is_causal=is_causal,
        standard_dense_output=not is_causal,
        standard_causal_output=is_causal,
    )
    code = bound.to_triton_code(active_config)
    outputs = [bound(q, k, v) for _ in range(3)]
    out = outputs[0]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal
    )
    diff = out.float() - expected.float()
    normalized_rmse = torch.sqrt(
        (diff * diff).mean(dtype=torch.float64)
        / (expected.float() * expected.float()).mean(dtype=torch.float64)
    )

    assert resolved.pipeline_family == family
    assert resolved.exp2_packet == packet
    assert resolved.use_2cta_instrs == (family == "fa4_2cta")
    if resolved.use_2cta_instrs:
        assert "cute_tcgen05_flash.CtaGroup.TWO" in code
    if packet == _CAUSAL_HD128_RESIDENT_PACKET:
        assert "fa4_disc_exp_convert_store_resident3_013_prefetch2" in code
    elif packet == _HYBRID_PACKET:
        assert "degree1=True" in code
        assert "degree2=True" in code
    elif packet == _DEG2_PACKET:
        assert "degree2=True" in code
    else:
        assert "pair_batch=8, emu_batch=2" in code
    assert all(torch.equal(out, repeated) for repeated in outputs[1:])
    assert torch.isfinite(out).all()
    assert diff.abs().max().item() < 0.05
    assert normalized_rmse.item() < 0.004


@pytest.mark.parametrize(
    ("dtype", "head_dim", "is_causal", "packet", "family", "num_kv"),
    (
        (torch.bfloat16, 64, False, "8x2", "fa4_2cta", 3),
        (torch.bfloat16, 64, True, _HYBRID_PACKET, "fa4", 3),
        (torch.bfloat16, 128, False, _DEG2_PACKET, "fa4_2cta", 3),
        (
            torch.bfloat16,
            128,
            True,
            _CAUSAL_HD128_RESIDENT_PACKET,
            "fa4",
            3,
        ),
    ),
)
def test_specialized_flash_family_rejects_adjacent_unaligned_length(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    packet: str,
    family: str,
    num_kv: int,
) -> None:
    fragments = cute_flash.flash_autotune_fragments(
        head_dim,
        num_kv,
        dtype=dtype,
        is_causal=is_causal,
        standard_dense_output=not is_causal,
        standard_causal_output=is_causal,
    )
    packet_fragment = fragments[cute_flash.FLASH_EXP2_PACKET_KEY]
    family_fragment = fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    assert isinstance(packet_fragment, EnumFragment)
    assert isinstance(family_fragment, EnumFragment)
    packet_choices = packet_fragment.search_choices or packet_fragment.choices
    family_choices = family_fragment.search_choices or family_fragment.choices
    assert packet not in packet_choices or family not in family_choices
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                cute_flash.FLASH_EXP2_PACKET_KEY: packet,
            },
            dtype=dtype,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
        )
    assert (resolved.pipeline_family, resolved.exp2_packet) != (family, packet)


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
def test_causal_whole_row_request_uses_safe_disc_pipeline() -> None:
    """Causal ring2 requests must not enter the unacknowledged whole-row path."""
    torch.manual_seed(110)
    sequence = 8192
    q, k, v = (
        torch.randn(1, 1, sequence, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )
    config = {
        "block_sizes": [1, 128, 128],
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_KV_STAGE_KEY: 6,
        cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
        cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
    }
    resolved = cute_flash.resolve_flash_config(
        64,
        sequence // 128,
        config,
        dtype=torch.float16,
        is_causal=True,
        standard_causal_output=True,
    )
    code, out = code_and_output(_causal_attention_output, (q, k, v), **config)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert resolved.softmax_disc
    assert "fa4_disc_exp_convert_store_pipe" in code
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.02)


@onlyBackends(["cute"])
def test_dense_ring2_whole_row_request_uses_single_transport() -> None:
    """The autotuner must not emit the whole-row ring2 wait cycle."""
    torch.manual_seed(107)
    sequence = 49_152
    q, k, v = (
        torch.randn(1, 1, sequence, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )
    config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4",
        "cute_flash_kv_stage": 6,
        "cute_flash_persistent": False,
        "cute_flash_softmax_disc": False,
        "cute_flash_split_p_arrive": True,
        "cute_flash_stat_transport": "ring2",
        "cute_flash_p_store_rep": 32,
        "cute_flash_s_load_rep": 32,
        "cute_flash_role_map": "fa4",
        "cute_flash_rescale_threshold": 8.0,
    }
    resolved = cute_flash.resolve_flash_config(
        64,
        sequence // 128,
        config,
        dtype=torch.float16,
        standard_dense_output=True,
    )

    bound = _dense_attention_output.bind((q, k, v))
    active_config = helion.Config(**config)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    out = bound(q, k, v)
    repeated = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert resolved.stat_transport == "single"
    assert resolved.kv_stage == 6
    assert resolved.split_p_arrive
    assert "mbar_ptr=flash_pfor2_ptr" in code
    assert torch.equal(out, repeated)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.01, rtol=0.02)


@onlyBackends(["cute"])
def test_persistent_legacy_degree1_final_only_is_accurate_across_work() -> None:
    torch.manual_seed(105)
    # 131_072 rows keep multiple work items per CTA of the num_SM-capped
    # persistent grid, so the stat handoff must carry phases across items.
    q = torch.randn(1, 1, 131_072, 64, dtype=torch.float16, device=DEVICE)
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
        # Whole-row PTX ring2 requests normalize to the single-slot protocol.
        # Exercise the CuTe-GEMM fallback, where ring2 remains legal.
        with patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}, clear=True):
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
def test_bfloat16_hybrid_packet_matches_sdpa_beyond_registered_length() -> None:
    torch.manual_seed(108)
    # 4098 is the smallest even num_kv above the largest registered causal
    # policy (4096), so the seed exercises the transferred long-length path.
    q, k, v = (
        torch.randn(1, 1, 524_544, 64, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    config = next(
        seed.config
        for seed in cute_flash.flash_attention_seed_configs(
            64,
            4098,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
        )
        if seed.config.get(cute_flash.FLASH_EXP2_PACKET_KEY) == _HYBRID_PACKET
    )
    code, out = code_and_output(_causal_attention_output, (q, k, v), **config)
    repeated = code_and_output(_causal_attention_output, (q, k, v), **config)[1]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert "degree1=True" in code
    assert "degree2=True" in code
    assert torch.equal(out, repeated)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.02)


def test_degree2_packet_normalizes_by_effective_schedule() -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(),
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
        )
    assert resolved.exp2_packet == _DEG2_PACKET
    assert resolved.e2e_schedule == "16/6"
    assert resolved.masked_e2e_schedule == "16/6"


def test_bfloat16_hd128_causal_keeps_16x6_without_degree2() -> None:
    overrides = _manual_config(
        **{
            cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "inherit",
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        }
    )
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            128,
            4096,
            overrides,
            dtype=torch.bfloat16,
            is_causal=True,
        )
    assert resolved.exp2_packet == "1x1"
    assert resolved.e2e_schedule == "16/6"
    assert (resolved.e2e_freq, resolved.e2e_res) == (16, 6)
    assert resolved.masked_e2e_schedule == "inherit"
    assert (resolved.masked_e2e_freq, resolved.masked_e2e_res) == (16, 6)

    fragments = cute_flash.flash_autotune_fragments(
        128,
        4096,
        dtype=torch.bfloat16,
        is_causal=True,
        pipeline_family_override="fa4",
    )
    schedule_fragment = fragments[cute_flash.FLASH_E2E_SCHEDULE_KEY]
    packet_fragment = fragments[cute_flash.FLASH_EXP2_PACKET_KEY]
    assert isinstance(schedule_fragment, EnumFragment)
    assert isinstance(packet_fragment, EnumFragment)
    assert "16/6" in schedule_fragment.choices
    assert _DEG2_PACKET not in packet_fragment.choices
    assert _DEG2_PACKET not in (packet_fragment.search_choices or ())


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


def test_hybrid_packet_normalizes_by_effective_schedule() -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(_HYBRID_PACKET),
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
        )
    assert resolved.exp2_packet == _HYBRID_PACKET
    assert resolved.e2e_schedule == "16/8"
    assert resolved.masked_e2e_schedule == "16/8"
    assert resolved.split_p_arrive
    assert resolved.causal_kv_order == "descending"
    assert resolved.causal_loop_split


def test_degree2_packet_canonicalizes_cadence_but_preserves_phase_offsets() -> None:
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
                    standard_causal_output=True,
                )
            )

    assert resolved[0] == resolved[1]
    assert resolved[0].e2e_offset == 14
    assert resolved[0].e2e_offset0 == 14


@pytest.mark.parametrize(
    "eligibility_fact",
    ("has_kv_tile_pruning", "requires_ws_overlap", "small_biased_candidate"),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_hybrid_packet_does_not_survive_ineligible_attention_fact(
    dtype: torch.dtype, eligibility_fact: str
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            512,
            _manual_config(_HYBRID_PACKET),
            dtype=dtype,
            is_causal=True,
            standard_causal_output=True,
            **{eligibility_fact: True},
        )
    assert resolved.exp2_packet == "1x1"


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
            standard_causal_output=True,
        )

        fragment = spec._flat_fields()[cute_flash.FLASH_EXP2_PACKET_KEY]
        assert isinstance(fragment, EnumFragment)
        assert fragment.choices == (packet,)
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
