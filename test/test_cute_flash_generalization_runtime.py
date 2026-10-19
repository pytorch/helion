from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _dense_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
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
def _dense_attention_with_lse(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty(
        [q_view.size(0), m_dim],
        dtype=torch.float32,
        device=q_in.device,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * math.log2(math.e)
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
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
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


def _run_and_check(
    *,
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    sequence_extent: int,
    num_heads: int = 1,
    config_values: dict[str, object] | None = None,
    repeat_count: int = 3,
    bound_out: list[helion.runtime.kernel.BoundKernel] | None = None,
) -> tuple[str, cute_flash.FlashAttentionConfig]:
    torch.manual_seed(
        1700
        + head_dim
        + sequence_extent
        + int(is_causal)
        + int(dtype is torch.bfloat16)
    )
    q, k, v = (
        torch.randn(
            1,
            num_heads,
            sequence_extent,
            head_dim,
            dtype=dtype,
            device=DEVICE,
        )
        for _ in range(3)
    )
    kernel = _causal_attention if is_causal else _dense_attention
    values = {} if config_values is None else config_values
    active_config = helion.Config(block_sizes=[1, 128, 128])
    active_config.config.update(values)
    bound = kernel.bind((q, k, v))
    if bound_out is not None:
        bound_out.append(bound)
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    output = bound(q, k, v)
    for _ in range(repeat_count - 1):
        assert torch.equal(output, bound(q, k, v))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=is_causal,
    )
    tolerance = 3e-2 if dtype is torch.bfloat16 else 1e-2

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)

    resolved = cute_flash.resolve_flash_config(
        head_dim,
        sequence_extent // 128,
        values,
        dtype=dtype,
        num_bh=num_heads,
        is_causal=is_causal,
        standard_dense_output=not is_causal,
        standard_causal_output=is_causal,
    )
    return code, resolved


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_odd_kv_required_tma_generic_fallback_initial_config_is_correct() -> None:
    fallback_kernel = helion.kernel(backend="cute", static_shapes=True)(
        _dense_attention.fn
    )
    torch.manual_seed(20260817)
    q, k, v = (
        torch.randn(1, 2, 384, 64, dtype=torch.float16, device=DEVICE) for _ in range(3)
    )
    with patch.object(cute_flash, "_flash_output_requires_tma", return_value=True):
        bound = fallback_kernel.bind((q, k, v))

    spec = bound.config_spec
    assert not spec.cute_flash_search_enabled
    assert spec.cute_attention_generic_fallback_enabled
    initial = spec.default_config()
    assert initial.config["block_sizes"] == [1, 64, 64]
    bound.set_config(initial)
    code = bound.to_triton_code(initial)
    output = bound(q, k, v)
    repeated = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert "_helion_flash_rt" not in code
    assert "_flash_scale_log2" not in code
    assert torch.isfinite(output).all()
    assert torch.equal(output, repeated)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_causal_two_cta_long_repeated_runtime_is_deterministic_and_correct() -> None:
    code, resolved = _run_and_check(
        dtype=torch.float16,
        head_dim=64,
        is_causal=True,
        sequence_extent=32_768,
        num_heads=4,
        repeat_count=32,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_causal",
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
            cute_flash.FLASH_P_STORE_REP_KEY: 16,
        },
    )

    assert resolved.pipeline_family == "fa4_2cta_causal"
    assert resolved.causal_two_cta
    assert resolved.p_store_repetition == 16
    assert "cute_tcgen05_flash.CtaGroup.TWO" in code
    assert "is_two_cta=True" in code


@pytest.mark.parametrize(
    ("dtype", "head_dim", "is_causal"),
    (
        pytest.param(torch.float16, 64, False, id="fp16-d64-dense"),
        pytest.param(torch.bfloat16, 64, True, id="bf16-d64-causal"),
        pytest.param(torch.bfloat16, 128, False, id="bf16-d128-dense"),
        pytest.param(torch.float16, 128, True, id="fp16-d128-causal"),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_odd_kv_extent_automatically_uses_ws_overlap(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
) -> None:
    code, resolved = _run_and_check(
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        sequence_extent=384,
    )

    assert resolved.pipeline_family == "ws_overlap"
    assert "flash_shared_storage" in code
    assert "flash_fa4_shared_storage" not in code


@pytest.mark.parametrize(
    ("dtype", "head_dim", "is_causal"),
    (
        pytest.param(torch.bfloat16, 64, False, id="bf16-d64-dense"),
        pytest.param(torch.float16, 128, True, id="fp16-d128-causal"),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_even_unclustered_kv_extent_uses_one_cta_fa4(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
) -> None:
    code, resolved = _run_and_check(
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        sequence_extent=768,
    )

    assert resolved.pipeline_family == "fa4"
    assert not resolved.use_2cta_instrs
    assert "flash_fa4_shared_storage" in code
    assert "flash_shared_storage" not in code
    assert "is_two_cta=True" not in code


@pytest.mark.parametrize(
    ("family", "kv_stage", "dtype", "is_causal"),
    (
        pytest.param("fa4", 5, torch.bfloat16, False, id="base-bf16-depth5"),
        pytest.param("fa4_2cta", 7, torch.float16, False, id="2cta-depth7"),
        pytest.param("fa4_tma_4d", 9, torch.float16, False, id="4d-depth9"),
        pytest.param(
            "fa4_clc_local_tma_4d",
            5,
            torch.float16,
            False,
            id="clc-local-4d-depth5",
        ),
        pytest.param("fa4", 7, torch.float16, True, id="causal-depth7"),
        pytest.param(
            "fa4_2cta_causal",
            9,
            torch.float16,
            True,
            id="causal-2cta-depth9",
        ),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_aliased_kv_depths_wrap_repeatedly_and_match_sdpa(
    family: str,
    kv_stage: int,
    dtype: torch.dtype,
    is_causal: bool,
) -> None:
    _code, resolved = _run_and_check(
        dtype=dtype,
        head_dim=64,
        is_causal=is_causal,
        sequence_extent=4096,
        num_heads=4,
        repeat_count=5,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
        },
    )

    assert resolved.pipeline_family == family
    assert resolved.kv_stage == kv_stage


@pytest.mark.parametrize(
    ("head_dim", "kv_stage"),
    (
        pytest.param(64, 12, id="d64-depth12"),
        pytest.param(128, 5, id="d128-depth5"),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_deepest_direct_output_aliased_kv_depth_matches_sdpa(
    head_dim: int,
    kv_stage: int,
) -> None:
    _code, resolved = _run_and_check(
        dtype=torch.float16,
        head_dim=head_dim,
        is_causal=False,
        sequence_extent=4096,
        num_heads=2,
        repeat_count=5,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
            cute_flash.FLASH_EPI_TMA_KEY: False,
            cute_flash.FLASH_EPI_STG_KEY: False,
        },
    )

    assert resolved.kv_stage == kv_stage


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_non_tile_aligned_extent_falls_back_and_matches_sdpa() -> None:
    torch.manual_seed(2819)
    q, k, v = (
        torch.randn(1, 2, 192, 64, dtype=torch.float16, device=DEVICE) for _ in range(3)
    )
    config = helion.Config(block_sizes=[1, 128, 128])
    bound = _dense_attention.bind((q, k, v))
    bound.set_config(config)
    code = bound.to_triton_code(config)
    output = bound(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert "flash_shared_storage" not in code
    assert "flash_fa4_shared_storage" not in code
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_dense_source_policy_variants_are_deterministic_and_match_sdpa() -> None:
    values: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_PERSISTENT_KEY: True,
        cute_flash.FLASH_PERSISTENT_CTAS_PER_SM_KEY: 1,
        cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted",
        cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
        cute_flash.FLASH_SP_ROW_SUM_KEY: "whole",
        cute_flash.FLASH_P_STORE_REP_KEY: 16,
        cute_flash.FLASH_SOFTMAX_SETUP_KEY: "stage_local",
        cute_flash.FLASH_EPI_TMA_KEY: True,
        cute_flash.FLASH_EPI_TMA_SETUP_KEY: "role_local",
    }
    code, resolved = _run_and_check(
        dtype=torch.float16,
        head_dim=64,
        is_causal=False,
        sequence_extent=768,
        num_heads=64,
        config_values=values,
    )

    assert resolved.persistent_loop == "counted"
    assert resolved.sp_row_sum == "whole"
    assert resolved.softmax_setup == "stage_local"
    assert resolved.epi_tma_setup == "role_local"
    assert "for flash_tile_iter in cutlass.range(flash_tile_count, unroll=1)" in code
    assert "fa4_sp_exp_convert_store_whole_rowsum" in code
    assert "flash_tiled_ld_coord" in code
    epilogue_role = code.index("gO_tma = cute.flat_divide")
    assert code.rfind("warp_idx", 0, epilogue_role) >= 0


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_dense_d128_counted_persistent_loop_is_deterministic_and_correct() -> None:
    code, resolved = _run_and_check(
        dtype=torch.float16,
        head_dim=128,
        is_causal=False,
        sequence_extent=768,
        num_heads=2,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted",
        },
    )

    assert resolved.pipeline_family == "fa4"
    assert resolved.persistent
    assert resolved.persistent_loop == "counted"
    assert "for flash_tile_iter in cutlass.range(flash_tile_count, unroll=1)" in code


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_dense_whole_row_path_with_lse_is_deterministic_and_correct() -> None:
    torch.manual_seed(2901)
    q, k, v = (
        torch.randn(1, 2, 768, 64, dtype=torch.float16, device=DEVICE) for _ in range(3)
    )
    values: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_PERSISTENT_KEY: False,
        cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
        cute_flash.FLASH_SP_ROW_SUM_KEY: "whole",
        cute_flash.FLASH_P_STORE_REP_KEY: 16,
        cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
        cute_flash.FLASH_EPI_TMA_KEY: True,
    }
    active_config = helion.Config.from_dict({"block_sizes": [1, 128, 128], **values})
    bound = _dense_attention_with_lse.bind((q, k, v))
    bound.set_config(active_config)
    code = bound.to_triton_code(active_config)
    outputs = [bound(q, k, v) for _ in range(3)]

    expected_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
    expected_lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    out, lse = outputs[0]

    assert "fa4_sp_exp_convert_store_whole_rowsum" in code
    assert all(
        torch.equal(out, repeated_out) and torch.equal(lse, repeated_lse)
        for repeated_out, repeated_lse in outputs[1:]
    )
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)


def test_stage_local_softmax_setup_canonicalizes_for_rep32() -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        6,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_P_STORE_REP_KEY: 32,
            cute_flash.FLASH_SOFTMAX_SETUP_KEY: "stage_local",
        },
        dtype=torch.float16,
        standard_dense_output=True,
    )

    assert resolved.p_store_repetition == 32
    assert resolved.softmax_setup == "shared"


def test_source_policy_variants_canonicalize_for_ws_overlap() -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        3,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted",
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
            cute_flash.FLASH_SP_ROW_SUM_KEY: "whole",
            cute_flash.FLASH_P_STORE_REP_KEY: 16,
            cute_flash.FLASH_SOFTMAX_SETUP_KEY: "stage_local",
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_EPI_TMA_SETUP_KEY: "role_local",
        },
        dtype=torch.float16,
        standard_dense_output=True,
    )

    assert resolved.pipeline_family == "ws_overlap"
    assert resolved.persistent_loop == "while"
    assert resolved.sp_row_sum == "fragment"
    assert resolved.softmax_setup == "shared"
    assert resolved.epi_tma_setup == "shared"


@pytest.mark.parametrize(
    ("head_dim", "dtype", "is_causal", "family", "kv_stage"),
    (
        pytest.param(80, torch.bfloat16, False, "fa4", 9, id="d80-dense-depth9"),
        pytest.param(96, torch.float16, True, "fa4", 7, id="d96-causal-depth7"),
        pytest.param(32, torch.bfloat16, False, "fa4_deep_1cta", 4, id="d32-deep-1cta"),
        pytest.param(112, torch.bfloat16, False, "ws_overlap", 2, id="d112-ws-overlap"),
        pytest.param(80, torch.float16, False, "fa4_local_tma", 6, id="d80-local-tma"),
        pytest.param(48, torch.bfloat16, True, "fa4", 12, id="d48-causal-depth12"),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_head_dims_beyond_64_and_128_match_sdpa(
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    family: str,
    kv_stage: int,
) -> None:
    """Every multiple of 16 up to 128 is a fused head dim: exact SMEM tiles
    (the swizzle the width admits), exact MMA extents, ``head_dim`` TMEM
    output columns; the direct-store ring depth follows the byte model."""
    _code, resolved = _run_and_check(
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        sequence_extent=2048,
        num_heads=2,
        repeat_count=3,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
        },
    )
    assert resolved.pipeline_family == family
    assert resolved.kv_stage == kv_stage
    assert resolved.corr_tile_size in (8, 16, 32, 64)
    assert head_dim % resolved.corr_tile_size == 0


@pytest.mark.parametrize(
    ("head_dim", "dtype", "is_causal", "kv_stage", "epi_stg"),
    (
        pytest.param(144, torch.bfloat16, False, 2, True, id="d144-dense-staged"),
        pytest.param(192, torch.float16, True, 1, True, id="d192-causal-staged"),
        pytest.param(256, torch.bfloat16, True, 1, False, id="d256-causal"),
        pytest.param(256, torch.float16, False, 1, False, id="d256-dense"),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_head_dims_above_128_run_the_single_tile_body(
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    kv_stage: int,
    epi_stg: bool,
) -> None:
    """Above width 128 the pair body's TMEM (two score buffers beside two
    accumulators) is full, so the resolver pins the single-query-tile body:
    one 256-column accumulator, separate K and V rings as deep as the byte
    model allows (one stage at 192 and 256), the staged output tile only
    where it leaves room for that ring (not at 256 with 128-row tiles)."""
    code, resolved = _run_and_check(
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        sequence_extent=2048,
        num_heads=2,
        repeat_count=3,
        # (A pair-body family request is rejected outright at these widths;
        # the single CTA's body is asked for by name, the default being its
        # cooperative pair.)
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_KV_STAGE_KEY: 3,
            cute_flash.FLASH_EPI_STG_KEY: True,
        },
    )
    assert resolved.pipeline_family == "ws_overlap"
    assert resolved.topology == "ws_overlap"
    assert resolved.kv_stage == kv_stage
    assert resolved.epi_stg == epi_stg
    assert "'topology': 'ws_overlap'" in code
    assert f"'head_dim': {head_dim}," in code


@pytest.mark.parametrize(
    ("head_dim", "dtype", "is_causal", "kv_stage"),
    (
        pytest.param(256, torch.bfloat16, True, 2, id="d256-causal"),
        pytest.param(256, torch.float16, False, 2, id="d256-dense"),
        pytest.param(192, torch.bfloat16, True, 3, id="d192-causal"),
        pytest.param(144, torch.bfloat16, False, 3, id="d144-dense"),
    ),
)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_head_dims_run_the_single_tile_cooperative_pair(
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    kv_stage: int,
) -> None:
    """``ws_2cta``: two adjacent 128-row query tiles of one batch head form a
    CTA pair that shares every K/V tile (64 rows per CTA, a 256-row
    cooperative MMA), the leader issuing the MMAs; the causal pair walks the
    union of its ranges and the lower tile masks the tile past its
    diagonal. The half tiles admit a deeper ring than the single CTA's."""
    code, resolved = _run_and_check(
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        sequence_extent=2048,
        num_heads=2,
        repeat_count=3,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_KV_STAGE_KEY: 3,
        },
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.topology == "ws_overlap"
    assert resolved.use_2cta_instrs
    assert resolved.kv_stage == kv_stage
    assert not resolved.persistent
    assert not resolved.epi_stg
    assert "'use_2cta_instrs': True" in code
    assert "_helion_cute_cluster_shape = (2, 1, 1)" in code
    assert f"'head_dim': {head_dim}," in code


@pytest.mark.parametrize("head_dim", (144, 160, 176, 208, 224, 240))
@pytest.mark.parametrize("is_causal", (False, True), ids=("dense", "causal"))
@pytest.mark.parametrize("family", ("ws_2cta", "ws_overlap"))
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_direct_store_at_widths_between_the_64_multiples(
    head_dim: int, is_causal: bool, family: str
) -> None:
    """The direct output store loads TMEM in ``Ld32x32bOp`` repetitions of
    gcd(head_dim, 64) columns: a 64-column repetition over a 144-wide tile
    partitioned to three repetitions and wrote 384 bytes into 288-byte rows
    (garbage leading columns in rows 1-31 of every warp, an overrun of the
    last row). Both single-tile bodies, dense and causal, every width above
    128 that is not a multiple of 64."""
    _code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=head_dim,
        is_causal=is_causal,
        sequence_extent=1024,
        num_heads=2,
        repeat_count=2,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_EPI_STG_KEY: False,
        },
    )
    assert resolved.pipeline_family == family
    assert not resolved.epi_stg


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_odd_tile_count_above_128_keeps_the_single_cta_body() -> None:
    """An odd query tile count has no pair for its last tile: the resolver
    keeps the single CTA's body (the pair family is not on the surface)."""
    code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=256,
        is_causal=True,
        sequence_extent=1152,
        num_heads=2,
        repeat_count=2,
    )
    assert resolved.pipeline_family == "ws_overlap"
    assert not resolved.use_2cta_instrs
    assert "'use_2cta_instrs': False" in code


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_cooperative_pair_persistent_cluster_loop_matches_sdpa() -> None:
    """On request the pair strides the cluster ids (one cluster per SM pair)
    over the pair work items and carries its pipelines across tiles: 32
    pairs per head at seq 4096, four heads, so every cluster walks more than
    one pair of different KV lengths (the causal pairs launch longest
    first)."""
    code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=256,
        is_causal=True,
        sequence_extent=4096,
        num_heads=4,
        repeat_count=3,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PERSISTENT_KEY: True,
        },
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.persistent
    assert "while flash_tile_id < _flash_total_tiles:" in code
    assert "'total_tiles': 64," in code


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_cooperative_pair_two_pair_cluster_matches_sdpa() -> None:
    """Two pairs per cluster sharing every K/V tile by TMA multicast across
    the pairs (four consecutive query tiles; the causal group walks to its
    highest tile)."""
    code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=256,
        is_causal=True,
        sequence_extent=2048,
        num_heads=2,
        repeat_count=3,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: 2,
        },
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.pair_cluster == 2
    assert "_helion_cute_cluster_shape = (4, 1, 1)" in code


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_cooperative_pair_residual_launch_group_matches_sdpa() -> None:
    """An explicit launch group that does not divide the heads (six heads in
    groups of four) takes FA4's residual group on the pair's decode."""
    code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=256,
        is_causal=True,
        sequence_extent=2048,
        num_heads=6,
        repeat_count=2,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 4,
        },
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.causal_lpt_swizzle == 4
    assert "flash_lpt_width = cutlass.Int32(2)" in code


@pytest.mark.parametrize("pair_cluster", (1, 2))
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_cooperative_pair_odd_pair_count_matches_sdpa(pair_cluster: int) -> None:
    """A tile count that is not a power of two (twelve tiles: six pairs,
    three two-pair clusters) with three heads: the launch group clamps to
    the grid, every pair's lower CTA skips the softmax of the tile above its
    diagonal, and the two-pair cluster's multicast covers the odd cluster
    count per head."""
    code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=256,
        is_causal=True,
        sequence_extent=1536,
        num_heads=3,
        repeat_count=3,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: pair_cluster,
        },
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.pair_cluster == pair_cluster
    assert f"'total_tiles': {3 * 12 // (2 * pair_cluster)}," in code
    assert "(flash_mma_tile_coord_v != 0) | (flash_kv != flash_m_tile_hi)" in code


# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
def test_cooperative_pair_persistent_two_pair_cluster_grid_is_the_resident_count() -> (
    None
):
    """The persistent two-pair cluster's grid holds the four-CTA clusters the
    device co-schedules (the driver's answer, 33 on a B200 rather than
    ``148 // 4 = 37``: a cluster lives inside one GPC and the surplus
    clusters re-ran the prologue as a second hardware wave): the emitted
    wrapper bakes ``min(item count x 4, clusters x 4)`` into its grid, the
    compiled-launcher key carries the count, and the kernel matches SDPA at
    an odd cluster count per head (twenty tiles, five clusters) over eight
    heads."""
    import linecache

    from helion.runtime import get_max_active_clusters
    from helion.runtime import get_num_sm
    from helion.runtime.cute.launcher import _cute_resident_clusters

    bound_out: list[helion.runtime.kernel.BoundKernel] = []
    code, resolved = _run_and_check(
        dtype=torch.bfloat16,
        head_dim=256,
        is_causal=True,
        sequence_extent=2560,
        num_heads=8,
        repeat_count=3,
        config_values={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: 2,
            cute_flash.FLASH_PERSISTENT_KEY: True,
        },
        bound_out=bound_out,
    )
    assert resolved.pair_cluster == 2 and resolved.persistent
    assert "'total_tiles': 40," in code
    assert "while flash_tile_id < _flash_total_tiles:" in code
    num_sm = get_num_sm(DEVICE)
    clusters = get_max_active_clusters(DEVICE, 4)
    assert 1 <= clusters <= num_sm // 4
    # The emitted wrapper: its grid is the smaller of the item count's CTAs
    # (forty items of four) and the resident clusters' CTAs.
    (bound,) = bound_out
    assert bound._run is not None
    cute_kernel = bound._run.__globals__[f"_helion_{bound.kernel.name}"]
    prefix = f"<helion_cute_launcher:{cute_kernel.__name__}_{id(cute_kernel):x}:"
    sources = [
        "".join(entry[2])
        for filename, entry in linecache.cache.items()
        if filename.startswith(prefix)
    ]
    assert len(sources) == 1, len(sources)
    assert f"    grid_x = cutlass.Int32({min(40 * 4, clusters * 4)})" in sources[0]
    # The compiled-launcher key carries the count (it varies with the GPC
    # floorsweeping of same-SM-count devices); the pair and CPU builds carry
    # nothing extra.
    (cache_key,) = cute_kernel._helion_cute_compiled_launchers
    assert cache_key[-1] == clusters
    assert _cute_resident_clusters((4, 1, 1), DEVICE) == clusters
    assert _cute_resident_clusters((2, 1, 1), DEVICE) is None
    assert _cute_resident_clusters((4, 1, 1), None) is None
