from __future__ import annotations

import ast
import re

from examples.nvfp4_gemv import _nvfp4_gemv_bf16in_body
from examples.nvfp4_gemv import _nvfp4_gemv_fp4in_body
from examples.nvfp4_gemv import swizzled_scale_numel
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

_CONFIG = helion.Config(
    block_sizes=[1, 256],
    num_threads=[0, 128],
    cute_vector_widths=[1, 1],
    cute_lane_layouts=["blocked", "strided"],
)


def _scalar_byte_loads(source: str, tensor: str) -> int:
    return len(
        re.findall(
            rf"cute\.arch\.load\({tensor}\.iterator[^\n]*cutlass\.Uint8\)", source
        )
    )


def _vector_loads(source: str, tensor: str, width: int, carrier: str) -> int:
    return len(
        re.findall(
            rf"cute\.arch\.load\({tensor}\.iterator[^\n]*ir\.VectorType\.get\(\[{width}\], {carrier}\.mlir_type\)\)",
            source,
        )
    )


def _render(fn: object, inputs: tuple[object, ...], config: helion.Config) -> str:
    kernel = helion.kernel(
        fn,  # pyrefly: ignore[bad-argument-type]
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with _cpu_target():
        bound = kernel._bind_isolated(inputs)
        return bound.to_code(config)


def test_bf16in_gemv_merges_qword_and_activation_loads() -> None:
    """``w[row, k, byte]`` for the eight bytes of a group becomes one 8-byte
    load; the sixteen bf16 activations of the group become two 16-byte loads
    (the launcher's 16-byte base alignment bounds a piece)."""
    rows, groups = 8192, 256
    inputs = (
        torch.empty((rows, groups, 8), dtype=torch.float4_e2m1fn_x2),
        torch.empty((groups, 16), dtype=torch.bfloat16),
        torch.empty((swizzled_scale_numel(rows, groups),), dtype=torch.float8_e4m3fn),
        torch.empty((rows,), dtype=torch.bfloat16),
        1.0,
    )
    source = _render(_nvfp4_gemv_bf16in_body, inputs, _CONFIG)
    assert _vector_loads(source, "weight_fp4x2", 8, "cutlass.Uint8") == 1
    assert _scalar_byte_loads(source, "weight_fp4x2") == 0
    assert _vector_loads(source, "x_values", 8, "cutlass.Uint16") == 2
    assert ").load()" not in source
    assert (
        "cutlass.Uint16(_helion_literal_vec_2[7]).bitcast(cutlass.BFloat16)" in source
    )
    # The masked scale gather keeps its scalar form.
    assert _scalar_byte_loads(source, "weight_scale") == 1


def test_fp4in_gemv_merges_both_packed_operands() -> None:
    rows, groups = 8192, 256
    inputs = (
        torch.empty((rows, groups, 8), dtype=torch.float4_e2m1fn_x2),
        torch.empty((groups, 8), dtype=torch.float4_e2m1fn_x2),
        torch.empty((swizzled_scale_numel(rows, groups),), dtype=torch.float8_e4m3fn),
        torch.empty((swizzled_scale_numel(1, groups),), dtype=torch.float8_e4m3fn),
        torch.empty((rows,), dtype=torch.bfloat16),
        1.0,
    )
    source = _render(_nvfp4_gemv_fp4in_body, inputs, _CONFIG)
    assert _vector_loads(source, "weight_fp4x2", 8, "cutlass.Uint8") == 1
    assert _vector_loads(source, "x_fp4x2", 8, "cutlass.Uint8") == 1
    assert _scalar_byte_loads(source, "weight_fp4x2") == 0
    assert _scalar_byte_loads(source, "x_fp4x2") == 0
    # The byte extracts of both vectors fold into the exact group dot.
    assert (
        "_cute_float4_e2m1fn_x16_dot_float32(_helion_literal_vec_1, _helion_literal_vec_2)"
        in source
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _strided_bytes(w: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The literal dim has stride K: the eight loads are not contiguous."""
    rows, _, groups = w.shape
    block_m = hl.register_block_size(1, 1)
    block_k = hl.register_block_size(16, groups)
    for tile_m in hl.tile(rows, block_size=block_m):
        row = tile_m.begin
        acc = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(groups, block_size=block_k):
            contrib = hl.zeros([tile_k], dtype=torch.float32)
            for byte in hl.static_range(8):
                lo, hi = hl.float4_e2m1fn_x2_to_float32(w[row, byte, tile_k])
                contrib = contrib + lo * x[tile_k, byte * 2].to(torch.float32)
                contrib = contrib + hi * x[tile_k, byte * 2 + 1].to(torch.float32)
            acc = acc + contrib.sum()
        out[row] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_between(w: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A store between two literal loads of one group: the run stays scalar
    up to the store."""
    rows, groups, _ = w.shape
    block_m = hl.register_block_size(1, 1)
    block_k = hl.register_block_size(16, groups)
    for tile_m in hl.tile(rows, block_size=block_m):
        row = tile_m.begin
        for tile_k in hl.tile(groups, block_size=block_k):
            lo0, hi0 = hl.float4_e2m1fn_x2_to_float32(w[row, tile_k, 0])
            out[row, tile_k] = (lo0 + hi0).to(torch.bfloat16)
            lo1, hi1 = hl.float4_e2m1fn_x2_to_float32(w[row, tile_k, 1])
            x[row, tile_k] = (lo1 - hi1).to(torch.bfloat16)
    return out


def test_non_unit_literal_stride_stays_scalar() -> None:
    inputs = (
        torch.empty((64, 8, 256), dtype=torch.float4_e2m1fn_x2),
        torch.empty((256, 16), dtype=torch.bfloat16),
        torch.empty((64,), dtype=torch.bfloat16),
    )
    source = _render(_strided_bytes.fn, inputs, _CONFIG)
    assert _scalar_byte_loads(source, "w") == 8
    assert _vector_loads(source, "w", 8, "cutlass.Uint8") == 0
    # The activations are still contiguous along the literal and merge.
    assert _vector_loads(source, "x", 8, "cutlass.Uint16") == 2


def test_store_between_literal_loads_stays_scalar() -> None:
    inputs = (
        torch.empty((64, 256, 8), dtype=torch.float4_e2m1fn_x2),
        torch.empty((64, 256), dtype=torch.bfloat16),
        torch.empty((64, 256), dtype=torch.bfloat16),
    )
    source = _render(_store_between.fn, inputs, _CONFIG)
    assert _scalar_byte_loads(source, "w") == 2
    assert "ir.VectorType" not in source


def test_rebound_pointer_name_between_loads_stays_scalar() -> None:
    """``indices`` is rebound between two literal loads: the same pointer
    text names another address, so the run is not hoisted."""
    from helion._compiler.cute.literal_offset_loads import _Merger
    from helion._compiler.cute.literal_offset_loads import _TensorFacts

    facts = {"w": _TensorFacts("cutlass.Uint8", {0: 8, 1: 1}, 16)}
    load = (
        "cute.arch.load(w.iterator + cutlass.Int32(indices) * "
        "cutlass.Int32(w.layout.stride[0]) + cutlass.Int32({byte}) * "
        "cutlass.Int32(w.layout.stride[1]), cutlass.Uint8)"
    )
    lines = [f"a{byte} = {load.format(byte=byte)}" for byte in range(4)]
    rebound = [*lines[:2], "indices = indices + 1", *lines[2:]]
    body = _Merger(facts).block(ast.parse("\n".join(rebound)).body)
    rewritten = ast.unparse(ast.Module(body=body, type_ignores=[]))
    assert "ir.VectorType" not in rewritten
    # Without the rebinding the four bytes merge into one 32-bit load.
    body = _Merger(facts).block(ast.parse("\n".join(lines)).body)
    rewritten = ast.unparse(ast.Module(body=body, type_ignores=[]))
    assert rewritten.count("ir.VectorType.get([4], cutlass.Uint8.mlir_type)") == 1
    assert "a3 = cutlass.Uint8(_helion_literal_vec_1[3])" in rewritten
