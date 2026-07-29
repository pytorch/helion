from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import patch_cute_mma_support
import helion.language as hl


def _config(block_sizes: list[int]) -> helion.Config:
    return helion.Config(
        block_sizes=block_sizes,
        l2_groupings=[1],
        loop_orders=[list(range(len(block_sizes) - 1))],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )


@helion.kernel(backend="cute")
def _pointwise_epilogue(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        shifted = torch.add(acc, 1.0, alpha=2.0)
        out[tile_m, tile_n] = (shifted * torch.sigmoid(shifted)).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _gelu_epilogue(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = torch.nn.functional.gelu(acc, approximate="tanh").to(
            x.dtype
        )
    return out


@helion.kernel(backend="cute")
def _masked_aux_epilogue(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        aux = hl.load(
            residual,
            [tile_m, tile_n],
            extra_mask=mask[tile_m, tile_n],
        )
        out[tile_m, tile_n] = (acc + aux).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _random_epilogue(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc + torch.rand_like(acc)).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _pair_swap_epilogue(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        pairs = acc.view(tile_m.block_size, tile_n.block_size // 2, 2)
        first, second = hl.split(pairs)
        swapped = hl.join(second, first).view(tile_m.block_size, tile_n.block_size)
        out[tile_m, tile_n] = swapped.to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _swapped_aux_indices(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc + residual[tile_n, tile_m]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _nonlocal_advanced_index(
    x: torch.Tensor, weight: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_k, tile_n], acc=acc)
        row = (acc.T > 0).to(torch.int64)
        column = torch.zeros_like(row)
        out[tile_m, tile_n] = (acc + source[row, column]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _projection_rotary(
    x: torch.Tensor,
    weight: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    h, _, d = weight.size()
    out = torch.empty([h, m, d], dtype=x.dtype, device=x.device)
    for tile_h, tile_m, tile_d in hl.tile([h, m, d]):
        acc = hl.zeros([tile_h, tile_m, tile_d], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_m, tile_k],
                weight[tile_h, tile_k, tile_d],
                acc=acc,
            )
        bias_tile = bias[tile_h.index[:, None], tile_d.index[None, :]]
        acc = acc + bias_tile[:, None, :]
        table_tile = table[tile_m, tile_d]
        table_pairs = table_tile.view(
            tile_m.block_size, 2, tile_d.block_size // 2
        ).permute(0, 2, 1)
        left, right = hl.split(table_pairs)
        pairs = acc.view(
            tile_h.block_size,
            tile_m.block_size,
            tile_d.block_size // 2,
            2,
        )
        x0, x1 = hl.split(pairs)
        perpendicular = hl.join(-x1, x0)
        result = (
            pairs * right[None, :, :, None] + perpendicular * left[None, :, :, None]
        )
        out[tile_h, tile_m, tile_d] = result.view(
            tile_h.block_size, tile_m.block_size, tile_d.block_size
        ).to(x.dtype)
    return out


def _code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], config: helion.Config
) -> str:
    with patch_cute_mma_support():
        bound = kernel.bind(args)
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        bound.set_config(config)
        return bound.to_triton_code(config)


def test_generic_pointwise_uses_inductor_semantics() -> None:
    x = torch.empty([128, 128], device="cuda", dtype=torch.bfloat16)
    code = _code(_pointwise_epilogue, (x, x), _config([128, 128, 32]))
    assert "cute.gemm" in code
    assert "tcgen05_epi_value" in code
    assert "tcgen05_chain_step" not in code


def test_generic_api_elementwise_uses_existing_codegen() -> None:
    x = torch.empty([128, 128], device="cuda", dtype=torch.bfloat16)
    code = _code(_gelu_epilogue, (x, x), _config([128, 128, 32]))
    assert "cute.gemm" in code
    assert "cute.math.tanh" in code
    assert "tcgen05_chain_step" not in code


def test_generic_elementwise_runtime() -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")
    x = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    weight = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    acc = x.float() @ weight.float()
    for kernel, expected in (
        (_pointwise_epilogue, (acc + 2.0) * torch.sigmoid(acc + 2.0)),
        (_gelu_epilogue, torch.nn.functional.gelu(acc, approximate="tanh")),
    ):
        bound = kernel.bind((x, weight))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        bound.set_config(_config([128, 128, 32]))
        torch.testing.assert_close(
            bound(x, weight), expected.to(x.dtype), atol=0.5, rtol=2e-2
        )


def test_rejected_masked_load_plan_fails_cleanly() -> None:
    x = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    weight = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    mask = torch.ones_like(x, dtype=torch.bool)
    config = _config([128, 128, 32])
    bound = _masked_aux_epilogue.bind((x, weight, residual, mask))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with pytest.raises(exc.BackendUnsupported, match="atomic viability check"):
        bound.to_triton_code(config)


def test_inputless_pointwise_plan_fails_cleanly() -> None:
    x = torch.empty([128, 128], device="cuda", dtype=torch.bfloat16)
    config = _config([128, 128, 32])
    bound = _random_epilogue.bind((x, x))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with pytest.raises(exc.BackendUnsupported, match="atomic viability check"):
        bound.to_triton_code(config)


def test_generic_pair_transform_uses_coordinate_fragment() -> None:
    x = torch.empty([128, 128], device="cuda", dtype=torch.bfloat16)
    code = _code(_pair_swap_epilogue, (x, x), _config([128, 128, 32]))
    assert "cute.gemm" in code
    assert "tcgen05_epi_scan" in code
    assert "split_smem" not in code
    assert "reshape_smem" not in code


def test_generic_pair_transform_runtime() -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")
    x = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    weight = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    bound = _pair_swap_epilogue.bind((x, weight))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_config([128, 128, 32]))
    actual = bound(x, weight)
    expected = (x @ weight).view(128, 64, 2).flip(-1).view(128, 128)
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)


def test_mismatched_aux_provenance_uses_direct_load() -> None:
    x = torch.empty([128, 128], device="cuda", dtype=torch.bfloat16)
    code = _code(_swapped_aux_indices, (x, x, x), _config([128, 128, 32]))
    assert "cute.gemm" in code
    assert "tcgen05_epi_load" in code
    assert "tcgen05_aux_tile" not in code


def test_mismatched_aux_provenance_runtime() -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")
    x = torch.randn([256, 128], device="cuda", dtype=torch.bfloat16)
    weight = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    residual = torch.randn([128, 256], device="cuda", dtype=torch.bfloat16)
    bound = _swapped_aux_indices.bind((x, weight, residual))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_config([128, 128, 32]))
    expected = (x.float() @ weight.float() + residual.T.float()).to(x.dtype)
    torch.testing.assert_close(
        bound(x, weight, residual), expected, atol=0.5, rtol=2e-2
    )


def test_direct_load_with_smaller_source_rejected_before_commit() -> None:
    x = torch.empty([256, 128], device="cuda", dtype=torch.bfloat16)
    weight = torch.empty([128, 128], device="cuda", dtype=torch.bfloat16)
    residual = torch.empty([64, 256], device="cuda", dtype=torch.bfloat16)
    config = _config([128, 128, 32])
    bound = _swapped_aux_indices.bind((x, weight, residual))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with pytest.raises(exc.BackendUnsupported, match="atomic viability check"):
        bound.to_triton_code(config)


def test_nonlocal_advanced_index_rejected_before_commit() -> None:
    x = torch.randn([128, 128], device="cuda", dtype=torch.bfloat16)
    source = torch.randn([2, 1], device="cuda", dtype=torch.bfloat16)
    config = _config([128, 128, 32])
    bound = _nonlocal_advanced_index.bind((x, x, source))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with pytest.raises(exc.BackendUnsupported, match="atomic viability check"):
        bound.to_triton_code(config)


def test_projection_rotary_is_generic_fragment_epilogue() -> None:
    dtype = torch.bfloat16
    x = torch.empty([128, 128], device="cuda", dtype=dtype)
    for head_dim in (64, 128):
        weight = torch.empty([1, 128, head_dim], device="cuda", dtype=dtype)
        table = torch.empty([128, head_dim], device="cuda", dtype=dtype)
        bias = torch.empty([1, head_dim], device="cuda", dtype=dtype)
        code = _code(
            _projection_rotary,
            (x, weight, table, bias),
            _config([1, 128, head_dim, 32]),
        )
        assert "cute.gemm" in code
        assert "tcgen05_epi_scan" in code
        assert "tcgen05_epi_load" in code
        assert "split_smem" not in code
        assert "permute_smem" not in code
        assert len(code) < 100_000


@pytest.mark.parametrize("head_dim", [64, 128])
def test_projection_rotary_runtime(head_dim: int) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")
    dtype = torch.bfloat16
    heads = 2
    m = 256
    k = 128
    x = torch.randn([m, k], device="cuda", dtype=dtype)
    weight = torch.randn([heads, k, head_dim], device="cuda", dtype=dtype)
    table = torch.randn([m, head_dim], device="cuda", dtype=dtype)
    bias = torch.randn([heads, head_dim], device="cuda", dtype=dtype)
    bound = _projection_rotary.bind((x, weight, table, bias))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_config([1, 128, head_dim, 32]))
    actual = bound(x, weight, table, bias)

    acc = torch.einsum("mk,hkd->hmd", x.float(), weight.float())
    pairs = (acc + bias.float()[:, None, :]).view(heads, m, head_dim // 2, 2)
    table_pairs = table.float().view(m, 2, head_dim // 2).permute(0, 2, 1)
    left, right = table_pairs.unbind(-1)
    perpendicular = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
    expected = (
        pairs * right[None, :, :, None] + perpendicular * left[None, :, :, None]
    ).view(heads, m, head_dim)
    torch.testing.assert_close(actual, expected.to(dtype), atol=1.0, rtol=2e-2)
