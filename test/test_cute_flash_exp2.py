from __future__ import annotations

import ast
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
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
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


def test_degree2_polynomial_relative_error_bound() -> None:
    x = torch.linspace(0.0, 1.0, 100_001, dtype=torch.float32)
    c0, c1, c2 = _flash_runtime._POLY_EX2_DEG2
    approximate = (torch.tensor(c2) * x + torch.tensor(c1)) * x + torch.tensor(c0)
    relative_error = (approximate / torch.exp2(x) - 1.0).abs()

    assert relative_error.max().item() < 0.00173


def test_degree1_polynomial_relative_error_bound() -> None:
    x = torch.linspace(0.0, 1.0, 100_001, dtype=torch.float32)
    c0, c1 = _flash_runtime._POLY_EX2_DEG1
    approximate = torch.tensor(c1) * x + torch.tensor(c0)
    relative_error = (approximate / torch.exp2(x) - 1.0).abs()

    assert relative_error.max().item() < 0.02983


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
    ("packet", "num_kv", "sequence_extent", "total_tiles", "freq", "res", "emu"),
    (
        (_DEG1_SHORT_PACKET, 256, 32_768, 4_096, 8, 2, 2),
        (_DEG1_SHORT_PACKET, 512, 65_536, 8_192, 8, 2, 2),
        (_DEG1_SHORT_PACKET, 1024, 131_072, 16_384, 8, 2, 2),
        (_DEG1_PACKET, 2048, 262_144, 32_768, 16, 8, 4),
    ),
)
def test_degree1_packet_uses_whole_row_dense_pass2(
    packet: str,
    num_kv: int,
    sequence_extent: int,
    total_tiles: int,
    freq: int,
    res: int,
    emu: int,
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
            "degree1": True,
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
def test_hybrid_packet_runtime_matches_sdpa_beyond_previous_seed_cap() -> None:
    torch.manual_seed(103)
    q, k, v = (
        torch.randn(1, 1, 1_048_576, 64, dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    )

    code, out = code_and_output(
        _causal_attention_output,
        (q, k, v),
        **_hybrid_runtime_config(),
    )
    assert "degree1=True" in code
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
    ((_DEG2_PACKET, "16/6"), (_HYBRID_PACKET, "16/8")),
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
