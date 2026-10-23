"""Fused block-scaled preparation: eligibility, entry and numerics."""

from __future__ import annotations

import inspect
import re

from examples.nvfp4_gemm import _nvfp4_w4a4_matmul_kernel
from examples.nvfp4_gemm import pack_fp4
from examples.nvfp4_gemm import pack_fp4_last_dim
from examples.nvfp4_gemm import quantize_fp4_e2m1
from examples.nvfp4_gemm import reference_nvfp4_w4a4_matmul
from examples.nvfp4_gemm import swizzle_fp8_scales
from examples.nvfp4_gemm import swizzled_scale_offsets
import pytest
import torch

from test.test_cute_block_scaled_mma import _bind_native
from test.test_cute_block_scaled_mma import _native_config
from test.test_cute_block_scaled_mma import _wrapper_plan
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.block_scaled_config import normalize_block_scaled_config
from helion._compiler.cute.block_scaled_fused import fused_geometry
from helion._compiler.cute.block_scaled_prepare import preparation_source
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig

pytestmark = skipUnlessBackends(["cute"])


def _fused_plan(shape: tuple[int, int, int], **config: object) -> dict[str, object]:
    bound, _inputs = _bind_native(shape)
    settings: dict[str, object] = {
        "cute_scaled_tile_k": 256,
        "cute_scaled_tile_n": 64,
        "cute_scaled_cluster_m": 1,
        "cute_scaled_prepare": "fused",
        **config,
    }
    with _cpu_target():
        return _wrapper_plan(bound.to_code(_native_config(**settings)))


def test_w4a4_sources_are_read_directly() -> None:
    """K-major activations, a row-contiguous weight and swizzled scales: the
    consumer takes the sources, no preparation kernel is emitted."""
    plan = _fused_plan((1024, 128, 1024))
    assert plan["prepare"] == "fused"
    assert plan["fused"] == {"lhs": ("direct", 1024), "rhs": ("transpose", 1024)}
    source = preparation_source(plan)
    assert "_prepare" not in source
    assert "_consumer.fused_entry(a, b, sa, sb, output, alpha, stream)" in source


def test_fused_falls_back_to_the_tiled_preparation() -> None:
    # Strides that are not multiples of 16 bytes (n = 9) cannot be TMA'd.
    plan = _fused_plan((7, 5, 9))
    assert plan["prepare"] == "tiled" and plan["fused"] is None
    # K padded to the K tile differs from the scale layout's padding.
    plan = _fused_plan((128, 5, 128))
    assert plan["prepare"] == "tiled" and plan["fused"] is None
    # The 2-CTA pair keeps the preparation kernel.
    plan = _fused_plan((1024, 128, 1024), cute_scaled_cluster_m=2)
    assert plan["prepare"] == "tiled" and plan["fused"] is None
    # A transposed 256-row N tile exceeds the TMA swizzle width, and 64 rows
    # by a 64-wide K tile leave the eight transform warps half idle.
    plan = _fused_plan(
        (1024, 128, 1024), cute_scaled_tile_n=256, cute_scaled_tile_k=128
    )
    assert plan["prepare"] == "tiled" and plan["fused"] is None
    plan = _fused_plan((1024, 128, 1024), cute_scaled_tile_k=64)
    assert plan["prepare"] == "tiled" and plan["fused"] is None
    plan = _fused_plan((1024, 128, 1024), cute_scaled_tile_n=128, cute_scaled_tile_k=64)
    assert plan["prepare"] == "fused"


def test_geometry_requires_the_plain_affine_recipes() -> None:
    direct = tuple(
        f"(cutlass.Int64(cutlass.Int32(row)) * 512 + cutlass.Int64(cutlass.Int32(group)) * 8 + cutlass.Int64({b}) * 1)"
        for b in range(8)
    )
    transposed = tuple(
        f"(cutlass.Int64(cutlass.Int32(group)) * 2048 + cutlass.Int64({b}) * 256 + cutlass.Int64(cutlass.Int32(row)) * 1)"
        for b in range(8)
    )
    swizzled = (
        "(cutlass.Int64(cutlass.Int32((cutlass.Int32((cutlass.Int32((cutlass.Int32(row) // 128)) * 16)) "
        "+ cutlass.Int32((cutlass.Int32(group) // 4)))) * 512 + cutlass.Int32((cutlass.Int32(row) % 32)) * 16 "
        "+ cutlass.Int32((cutlass.Int32((cutlass.Int32(row) % 128)) // 32)) * 4 + cutlass.Int32(group) % 4) * 1)"
    )
    common = {
        "lhs_scale_address": swizzled,
        "rhs_scale_address": swizzled,
        "lhs_scale_span": 256 * 64,
        "rhs_scale_span": 256 * 64,
        "m": 200,
        "n": 256,
        "k": 1024,
        "bn": 128,
        "bk": 256,
        "cluster_m": 1,
    }
    assert fused_geometry(
        lhs_byte_addresses=direct, rhs_byte_addresses=transposed, **common
    ) == {"lhs": ("direct", 512), "rhs": ("transpose", 256)}
    # A plain (unswizzled) scale recipe is not the consumer's layout.
    plain = dict(
        common, lhs_scale_address="(cutlass.Int64(row) * 64 + cutlass.Int64(group))"
    )
    assert (
        fused_geometry(
            lhs_byte_addresses=direct, rhs_byte_addresses=transposed, **plain
        )
        is None
    )
    # A scale span short of the padded extents.
    short = dict(common, rhs_scale_span=256 * 64 - 1)
    assert (
        fused_geometry(
            lhs_byte_addresses=direct, rhs_byte_addresses=transposed, **short
        )
        is None
    )


def test_fused_is_an_nvfp4_only_choice() -> None:
    config: dict[str, object] = {
        "cute_scaled_mma": True,
        "cute_scaled_prepare": "fused",
    }
    normalize_block_scaled_config(
        config, available=True, capability=(10, 0), fix_invalid=False
    )
    assert config["cute_scaled_prepare"] == "fused"
    with pytest.raises(InvalidConfig):
        normalize_block_scaled_config(
            dict(config),
            available=True,
            capability=(10, 0),
            fix_invalid=False,
            mxfp8=True,
        )
    normalize_block_scaled_config(
        config, available=True, capability=(10, 0), fix_invalid=True, mxfp8=True
    )
    assert config["cute_scaled_prepare"] == "tiled"


@pytest.mark.parametrize(
    "bn,bk,stages,persistent",
    [
        (64, 256, 4, False),
        (64, 128, 3, True),
        (128, 256, 3, False),
        (128, 128, 4, True),
    ],
)
def test_cuda_fused_matches_the_reference_with_signed_scales(
    bn: int, bk: int, stages: int, persistent: bool
) -> None:
    """Signed E4M3 scales exercise the in-kernel sign fold of both sides; the
    shapes leave M, N and K tails for the TMA zero fill."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native NVFP4 required")
    m, k, n = 200, 768, 320
    torch.manual_seed(0)
    a_packed = pack_fp4_last_dim(
        quantize_fp4_e2m1(torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    w_packed = pack_fp4(
        quantize_fp4_e2m1(torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    act_scale = swizzle_fp8_scales(
        (torch.rand(m, k // 16, device=DEVICE) * 2 - 1).to(torch.float8_e4m3fn)
    )
    weight_scale = swizzle_fp8_scales(
        (torch.rand(n, k // 16, device=DEVICE) * 2 - 1).to(torch.float8_e4m3fn)
    )
    expected = reference_nvfp4_w4a4_matmul(a_packed, w_packed, act_scale, weight_scale)
    args = (
        a_packed.view(m, k // 16, 8),
        w_packed.view(k // 16, 8, n),
        act_scale.reshape(-1),
        weight_scale.reshape(-1),
        torch.empty(m, n, dtype=torch.bfloat16, device=DEVICE),
        1.0,
    )
    outputs = {}
    for prepare in ("tiled", "fused"):
        config = _native_config(
            cute_scaled_tile_n=bn,
            cute_scaled_tile_k=bk,
            cute_scaled_cluster_m=1,
            cute_scaled_stages=stages,
            cute_scaled_persistent=persistent,
            cute_scaled_prepare=prepare,
        )
        kernel = helion.kernel(
            _nvfp4_w4a4_matmul_kernel.fn,
            backend="cute",
            static_shapes=True,
            config=config,
        )
        assert (
            _wrapper_plan(kernel.bind(args).to_triton_code(config))["prepare"]
            == prepare
        )
        kernel(*args)
        torch.cuda.synchronize()
        outputs[prepare] = args[4].clone()
    # The fused consumer feeds the MMA the same bits as the prepared workspace.
    torch.testing.assert_close(outputs["fused"], outputs["tiled"], rtol=0, atol=0)
    torch.testing.assert_close(
        outputs["fused"].float(), expected.float(), rtol=2e-2, atol=2e-2
    )


def test_fused_consumer_waits_for_the_preceding_grid_under_pdl() -> None:
    """The fused kernel launches with the PDL attribute like the two-kernel
    form; its TMA producer must execute griddepcontrol.wait before the first
    source load (the preceding grid in the stream may still be writing the
    packed operands or scales), not only when a preparation kernel exists."""
    # Imported here: the module needs the CuTe DSL, which the test's backend
    # skip does not guard at collection time.
    from helion._compiler.cute.block_scaled_runtime import Sm100BlockScaledGemm

    source = inspect.getsource(Sm100BlockScaledGemm.kernel)
    assert "self.pdl and not self.fused" not in source
    assert re.search(
        r"if cutlass\.const_expr\(self\.pdl\):\n(?:[ \t]*#[^\n]*\n)*"
        r"\s*cute\.arch\.griddepcontrol_wait\(\)",
        source,
    )
    assert "use_pdl=self.pdl" in inspect.getsource(Sm100BlockScaledGemm._launch)


def _poisoned_scales(rows: int, groups: int) -> torch.Tensor:
    """Signed scales in the swizzled layout with every padding byte set to
    0xFF (a NaN in E4M3)."""
    logical = (torch.rand(rows, groups, device=DEVICE) * 2 - 1).to(torch.float8_e4m3fn)
    swizzled = swizzle_fp8_scales(logical)
    valid = torch.zeros(swizzled.numel(), dtype=torch.bool, device=DEVICE)
    row = torch.arange(rows, device=DEVICE, dtype=torch.int64)[:, None]
    col = torch.arange(groups, device=DEVICE, dtype=torch.int64)[None, :]
    valid[swizzled_scale_offsets(row, col, groups).reshape(-1)] = True
    swizzled.view(torch.uint8)[~valid] = 0xFF
    return swizzled


@pytest.mark.parametrize(("bn", "bk"), [(64, 256), (128, 128)])
def test_cuda_fused_masks_the_sources_padding_scales(bn: int, bk: int) -> None:
    """K = 736 (a 16-byte-aligned K-contiguous side, two padding groups in
    the last quad) and 200 x 320 (padding rows): NaN scale bytes there must
    not reach the MMA (the preparation kernel writes zeros; the fused
    transform masks the K tail, and the padding rows' NaNs stay in rows the
    epilogue never stores)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native NVFP4 required")
    m, k, n = 200, 736, 320
    torch.manual_seed(1)
    a_packed = pack_fp4_last_dim(
        quantize_fp4_e2m1(torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    w_packed = pack_fp4(
        quantize_fp4_e2m1(torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    act_scale = _poisoned_scales(m, k // 16)
    weight_scale = _poisoned_scales(n, k // 16)
    expected = reference_nvfp4_w4a4_matmul(a_packed, w_packed, act_scale, weight_scale)
    assert torch.isfinite(expected.float()).all()
    args = (
        a_packed.view(m, k // 16, 8),
        w_packed.view(k // 16, 8, n),
        act_scale.reshape(-1),
        weight_scale.reshape(-1),
        torch.empty(m, n, dtype=torch.bfloat16, device=DEVICE),
        1.0,
    )
    outputs = {}
    for prepare in ("tiled", "fused"):
        config = _native_config(
            cute_scaled_tile_n=bn,
            cute_scaled_tile_k=bk,
            cute_scaled_cluster_m=1,
            cute_scaled_stages=3,
            cute_scaled_persistent=False,
            cute_scaled_prepare=prepare,
        )
        kernel = helion.kernel(
            _nvfp4_w4a4_matmul_kernel.fn,
            backend="cute",
            static_shapes=True,
            config=config,
        )
        assert (
            _wrapper_plan(kernel.bind(args).to_triton_code(config))["prepare"]
            == prepare
        )
        kernel(*args)
        torch.cuda.synchronize()
        outputs[prepare] = args[4].clone()
    assert torch.isfinite(outputs["fused"].float()).all()
    torch.testing.assert_close(outputs["fused"], outputs["tiled"], rtol=0, atol=0)
    torch.testing.assert_close(
        outputs["fused"].float(), expected.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(("prepare", "offset"), [("linear", 4), ("fused", 8)])
def test_cuda_under_aligned_source_is_staged(prepare: str, offset: int) -> None:
    """A K-contiguous side read as 8-byte words (linear preparation) or by
    TMA (fused) from a storage offset below its alignment: the launcher
    clones it into an aligned buffer and the result matches the aligned
    call bit for bit."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native NVFP4 required")
    m, k, n = 128, 512, 256
    torch.manual_seed(2)
    a_packed = pack_fp4_last_dim(
        quantize_fp4_e2m1(torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    storage = torch.zeros(m * k // 2 + 16, dtype=torch.uint8, device=DEVICE)
    shifted = storage[offset : offset + m * k // 2].view(m, k // 2)
    shifted.copy_(a_packed.view(torch.uint8))
    assert shifted.data_ptr() % 16 == offset
    a_shifted = shifted.view(torch.float4_e2m1fn_x2)
    w_packed = pack_fp4(
        quantize_fp4_e2m1(torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE))
    ).view(torch.float4_e2m1fn_x2)
    act_scale = swizzle_fp8_scales(
        (torch.rand(m, k // 16, device=DEVICE) + 0.5).to(torch.float8_e4m3fn)
    )
    weight_scale = swizzle_fp8_scales(
        (torch.rand(n, k // 16, device=DEVICE) + 0.5).to(torch.float8_e4m3fn)
    )
    config = _native_config(
        cute_scaled_tile_n=64,
        cute_scaled_tile_k=256,
        cute_scaled_cluster_m=1,
        cute_scaled_stages=3,
        cute_scaled_persistent=False,
        cute_scaled_prepare=prepare,
    )
    kernel = helion.kernel(
        _nvfp4_w4a4_matmul_kernel.fn, backend="cute", static_shapes=True, config=config
    )
    outputs = []
    for a in (a_packed, a_shifted):
        out = torch.empty(m, n, dtype=torch.bfloat16, device=DEVICE)
        kernel(
            a.view(m, k // 16, 8),
            w_packed.view(k // 16, 8, n),
            act_scale.reshape(-1),
            weight_scale.reshape(-1),
            out,
            1.0,
        )
        torch.cuda.synchronize()
        outputs.append(out)
    plan = _wrapper_plan(
        kernel.bind(
            (
                a_packed.view(m, k // 16, 8),
                w_packed.view(k // 16, 8, n),
                act_scale.reshape(-1),
                weight_scale.reshape(-1),
                outputs[0],
                1.0,
            )
        ).to_triton_code(config)
    )
    assert plan["prepare"] == prepare
    if prepare == "linear":
        assert plan["lhs_qword_address"] is not None
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
