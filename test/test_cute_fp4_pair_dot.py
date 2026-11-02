from __future__ import annotations

import ast

from examples.nvfp4_gemv import _nvfp4_gemv_fp4in_body
from examples.nvfp4_gemv import reference_nvfp4_gemv_fp4in
from examples.nvfp4_gemv import swizzle_fp8_scales
import pytest
import torch

from test.test_cute_tile_packing import _gemv_args

import helion
from helion._compiler.cute.fuse_fp4_pair_dot import fuse_fp4_pair_dot
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _config(groups: int, threads: int, **extra: object) -> helion.Config:
    return helion.Config(
        block_sizes=[1, groups],
        num_threads=[1, threads],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["blocked", "strided"],
        **extra,
    )


@pytest.mark.parametrize(
    ("threads", "extra"),
    [(64, {}), (32, {"cute_lane_unroll": 8}), (32, {"cute_tiles_per_block": 4})],
)
def test_fp4_group_dot_replaces_the_byte_chain(threads: int, extra: dict) -> None:
    """The sixteen decode/multiply/add steps of one group become one exact
    f16x2 dot of the two merged 8-byte vectors; the result is unchanged."""
    rows, kbytes = 256, 2048
    args, expected = _gemv_args(rows, kbytes)
    config = _config(256, threads, **extra)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True, config=config
    )
    source = kernel.bind(args).to_triton_code(config)
    assert "_cute_float4_e2m1fn_x16_dot_float32(_helion_literal_vec_" in source
    assert "_cute_float4_e2m1fn_x2_to_float32(" not in source
    kernel(*args)
    torch.testing.assert_close(args[4].float(), expected.float(), rtol=0, atol=0)


def test_fp4_group_dot_is_exact_at_the_largest_products() -> None:
    """Every product is 36 (6 x 6) or -36: the sixteen-term sums reach the
    f16 quarter-resolution limit the helper relies on, and the kernel's f32
    result still equals the reference exactly."""
    rows, kbytes = 64, 2048
    groups = kbytes // 8
    # 0x7 encodes +6.0 and 0xF encodes -6.0 in E2M1; alternate signs by row.
    weight = torch.full((rows, kbytes), 0x77, dtype=torch.uint8, device=DEVICE)
    weight[1::2] = 0xFF
    weight[2::4, : kbytes // 2] = 0xF7
    weight = weight.view(torch.float4_e2m1fn_x2)
    x = torch.full((kbytes,), 0x77, dtype=torch.uint8, device=DEVICE)
    x = x.view(torch.float4_e2m1fn_x2)
    wscale = swizzle_fp8_scales(
        torch.ones(rows, groups, dtype=torch.float8_e4m3fn, device=DEVICE)
    )
    xscale = swizzle_fp8_scales(
        torch.ones(1, groups, dtype=torch.float8_e4m3fn, device=DEVICE)
    )
    expected = reference_nvfp4_gemv_fp4in(weight, x, wscale, xscale)
    out = torch.empty(rows, dtype=torch.bfloat16, device=DEVICE)
    args = (
        weight.view(rows, groups, 8),
        x.view(groups, 8),
        wscale.reshape(-1),
        xscale.reshape(-1),
        out,
        1.0,
    )
    config = _config(256, 32)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True, config=config
    )
    assert "_cute_float4_e2m1fn_x16_dot_float32(" in kernel.bind(args).to_triton_code(
        config
    )
    kernel(*args)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
    assert out.float().abs().max().item() == 256 * 16 * 36


def _chain(base: str) -> str:
    lines = [
        f"{base}",
        "W = cute.arch.load(p, ir.VectorType.get([8], cutlass.Uint8.mlir_type))",
        "X = cute.arch.load(q, ir.VectorType.get([8], cutlass.Uint8.mlir_type))",
    ]
    acc = "acc"
    for b in range(8):
        lines += [
            f"lw{b} = cutlass.Uint8(W[{b}])",
            f"pw{b} = _cute_float4_e2m1fn_x2_to_float32(lw{b})",
            f"wlo{b} = pw{b}[0]",
            f"whi{b} = pw{b}[1]",
            f"lx{b} = cutlass.Uint8(X[{b}])",
            f"px{b} = _cute_float4_e2m1fn_x2_to_float32(lx{b})",
            f"xlo{b} = px{b}[0]",
            f"xhi{b} = px{b}[1]",
            f"p{b}a = wlo{b} * xlo{b}",
            f"s{b}a = {acc} + p{b}a",
            f"p{b}b = whi{b} * xhi{b}",
            f"s{b}b = s{b}a + p{b}b",
        ]
        acc = f"s{b}b"
    lines.append(f"out = {acc} * scale")
    return "\n".join(lines)


def test_pass_requires_a_zero_start() -> None:
    fused = fuse_fp4_pair_dot(ast.parse(_chain("acc = cutlass.Float32(0.0)")).body)
    text = ast.unparse(ast.Module(body=fused, type_ignores=[]))
    assert "s7b = cutlass.Float32(_cute_float4_e2m1fn_x16_dot_float32(W, X))" in text
    assert "_cute_float4_e2m1fn_x2_to_float32" not in text
    assert text.count("cute.arch.load(") == 2
    # A running sum that does not start at zero rounds differently when
    # regrouped: the chain is left alone.
    kept = fuse_fp4_pair_dot(ast.parse(_chain("acc = other")).body)
    assert "_cute_float4_e2m1fn_x16_dot_float32" not in ast.unparse(
        ast.Module(body=kept, type_ignores=[])
    )
    # A zero that is overwritten before the chain is not a zero either.
    kept = fuse_fp4_pair_dot(
        ast.parse(_chain("acc = cutlass.Float32(0.0)\nacc = other")).body
    )
    assert "_cute_float4_e2m1fn_x16_dot_float32" not in ast.unparse(
        ast.Module(body=kept, type_ignores=[])
    )
