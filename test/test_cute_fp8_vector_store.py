from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl

# CuTe's generated launcher resolves kernel annotations and DSL primitives in
# module globals. Keep these optional imports available there while allowing
# the pure AST tests in this file to run without the SDK.
if importlib.util.find_spec("cutlass") is not None:
    import cutlass
    import cutlass.cute as cute


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _quantize_rows(x: torch.Tensor, dtype: torch.dtype, limit: int) -> torch.Tensor:
    rows, width = x.shape
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    for row in hl.tile(rows):
        columns = hl.arange(width)
        data = x[row, columns].float()
        maximum = torch.clamp(data.abs().amax(-1), min=1)
        encoded = (data / maximum[:, None]).to(dtype)
        if limit >= width:
            out[row, columns] = encoded
        else:
            hl.store(out, [row, columns], encoded, extra_mask=columns[None, :] < limit)
    return out


def _config(bound: Any, vec_width: int) -> helion.Config:
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    spec = bound.config_spec
    return helion.Config(
        block_sizes=[16],
        num_threads=[
            32 // vec_width if block == reduction else 8
            for block in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            vec_width if block == reduction else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_lane_layouts=[
            "strided" if block == reduction else "blocked"
            for block in spec.cute_lane_layouts.valid_block_ids()
        ],
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("vec_width", [2, 4, 8, 16])
def test_persistent_row_store_is_one_fp8_packet(
    dtype: torch.dtype, vec_width: int
) -> None:
    # A partial outer row tile must predicate the packet as a whole.
    args = (torch.empty((37, 32), dtype=torch.bfloat16), dtype, hl.constexpr(32))
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_quantize_rows, args)
        code = bound.to_code(_config(bound, vec_width))
    helper = (
        "_cute_store_fp8e4m3fn_vec"
        if dtype is torch.float8_e4m3fn
        else "_cute_store_fp8e5m2_vec"
    )
    assert code.count(f"{helper}(") == 1, code
    assert "_helion_persistent_branch_vec_store" not in code, code
    assert ".store(cutlass.Float8" not in code, code
    # The packet collects the unconverted lanes; the helper converts them.
    assert "= cutlass.Float8" not in code, code


@pytest.mark.parametrize("limit", [16, 17])
def test_lane_mask_keeps_partial_packet_stores(limit: int) -> None:
    args = (
        torch.empty((37, 32), dtype=torch.bfloat16),
        torch.float8_e4m3fn,
        hl.constexpr(limit),
    )
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_quantize_rows, args)
        code = bound.to_code(_config(bound, 8))
    assert "_helion_persistent_branch_vec_store" not in code, code
    # An explicit lane mask retains scalar stores, even at a packet boundary.
    assert "_cute_store_fp8e4m3fn_vec(" not in code, code
    assert ".store(cutlass.Float8E4M3FN(" in code, code


@pytest.mark.parametrize("dtype_name", ["Float8E4M3FN", "Float8E5M2"])
@pytest.mark.parametrize("source_dtype_name", ["Float32", "Float16", "BFloat16"])
def test_float_helper_emits_vector_conversion_and_integer_packet(
    dtype_name: str, source_dtype_name: str
) -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir.dialects import arith

    from helion._compiler.cute import vec_utils
    from helion._compiler.cute._mlir_compat import ir

    helper = (
        vec_utils.store_fp8e4m3fn_vec
        if dtype_name == "Float8E4M3FN"
        else vec_utils.store_fp8e5m2_vec
    )
    stored = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        source_dtype = getattr(cutlass, source_dtype_name)
        with (
            ir.InsertionPoint(module.body),
            patch.object(
                vec_utils.cute.arch,
                "store",
                side_effect=lambda pointer, value: stored.append(value),
            ),
        ):
            values = [
                source_dtype(
                    arith.constant(
                        source_dtype.mlir_type,
                        ir.FloatAttr.get(source_dtype.mlir_type, value),
                    )
                )
                for value in (0.0, -0.0, 1.0625, -1.0625)
            ]
            helper(None, values)
        source = str(module)
        assert f"vector<4x{source_dtype.mlir_type}>" in source
        assert "arith.truncf" in source
        assert f"vector<4x{getattr(cutlass, dtype_name).mlir_type}>" in source
        assert len(stored) == 1 and type(stored[0]) is cutlass.Uint32


@pytest.mark.parametrize(
    ("lanes", "stored_type"),
    [(2, "i16"), (4, "i32"), (8, "i64"), (16, "vector<2xi64>")],
)
def test_byte_packets_store_exactly_their_lane_bytes(
    lanes: int, stored_type: str
) -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir.dialects import arith

    from helion._compiler.cute import vec_utils
    from helion._compiler.cute._mlir_compat import ir

    stored = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with (
            ir.InsertionPoint(module.body),
            patch.object(
                vec_utils.cute.arch,
                "store",
                side_effect=lambda pointer, value: stored.append(value),
            ),
        ):
            # Float64 lanes convert one at a time and pack the bytes.
            values = [
                cutlass.Float64(
                    arith.constant(
                        cutlass.Float64.mlir_type,
                        ir.FloatAttr.get(cutlass.Float64.mlir_type, 1.0),
                    )
                )
                for _ in range(lanes)
            ]
            vec_utils.store_fp8e4m3fn_vec(None, values)
        (value,) = stored
        raw = value if isinstance(value, ir.Value) else value.ir_value()
        assert str(raw.type) == stored_type


def test_fp64_helper_keeps_scalar_conversion_precision() -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir.dialects import arith

    from helion._compiler.cute import vec_utils
    from helion._compiler.cute._mlir_compat import ir

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body), patch.object(vec_utils.cute.arch, "store"):
            values = [
                cutlass.Float64(
                    arith.constant(
                        cutlass.Float64.mlir_type,
                        ir.FloatAttr.get(cutlass.Float64.mlir_type, 1.0625),
                    )
                )
                for _ in range(2)
            ]
            vec_utils.store_fp8e4m3fn_vec(None, values)
        source = str(module)
        assert "vector<" not in source
        assert source.count("arith.truncf") == 2
        assert "f64 to f8E4M3FN" in source


@pytest.mark.parametrize(
    ("dtype_name", "midpoint", "rounded"),
    [("Float8E4M3FN", 1.0625, 1.125), ("Float8E5M2", 1.125, 1.25)],
)
@pytest.mark.parametrize("source_dtype_name", ["Float32", "Float16", "BFloat16"])
def test_literal_fp32_helper_keeps_direct_fp8_rounding(
    dtype_name: str, midpoint: float, rounded: float, source_dtype_name: str
) -> None:
    cutlass = pytest.importorskip("cutlass")

    from helion._compiler.cute import vec_utils
    from helion._compiler.cute._mlir_compat import ir

    helper = (
        vec_utils.store_fp8e4m3fn_vec
        if dtype_name == "Float8E4M3FN"
        else vec_utils.store_fp8e5m2_vec
    )
    stored_lanes = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with (
            ir.InsertionPoint(module.body),
            patch.object(
                vec_utils,
                "store_u8_vec",
                side_effect=lambda pointer, values: stored_lanes.extend(values),
            ),
        ):
            value = midpoint + 2.0**-26
            source_dtype = getattr(cutlass, source_dtype_name)
            helper(None, [source_dtype(value), source_dtype(-value)])
            # Uint8 lanes bitcast the FP8 constants emitted by the scalar
            # constructor. Materializing the wrapper as FP32 instead rounds
            # back to the exact midpoint, which then rounds down in FP8.
            actual = [
                ir.FloatAttr(
                    byte.value.owner.operands[0].owner.attributes["value"]
                ).value
                for byte in stored_lanes
            ]
        assert actual == [rounded, -rounded]
        source = str(module)
        assert "f32" not in source
        assert "arith.truncf" not in source


@pytest.mark.parametrize("source_dtype", ["bf16", "f16"])
@pytest.mark.parametrize("destination_dtype", ["f8E4M3FN", "f8E5M2"])
def test_half_scalar_and_vector_lower_to_same_native_rounding(
    source_dtype: str, destination_dtype: str
) -> None:
    pytest.importorskip("cutlass")
    from cutlass._mlir import passmanager

    from helion._compiler.cute._mlir_compat import ir

    # Keep both conversions live through byte stores. The CPU lowering does
    # not create a CUDA context, load a GPU module or execute either kernel.
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(
            f"""
module {{
  gpu.module @kernels {{
    gpu.func @kernel(%x: {source_dtype}, %v: vector<4x{source_dtype}>, %out: !llvm.ptr<1>) kernel {{
      %a = arith.truncf %x : {source_dtype} to {destination_dtype}
      %b = arith.truncf %v : vector<4x{source_dtype}> to vector<4x{destination_dtype}>
      %ia = arith.bitcast %a : {destination_dtype} to i8
      %ib = vector.bitcast %b : vector<4x{destination_dtype}> to vector<1xi32>
      llvm.store %ia, %out : i8, !llvm.ptr<1>
      llvm.store %ib, %out : vector<1xi32>, !llvm.ptr<1>
      gpu.return
    }}
  }}
}}
"""
        )
        manager: Any = passmanager
        manager.PassManager.parse(
            "builtin.module(nvvm-attach-target{chip=sm_100},core-to-llvm-prepare,"
            "one-shot-convert-to-llvm{index-bitwidth=64})"
        ).run(module.operation)
        conversions = [
            line for line in str(module).splitlines() if "nvvm.cvt.packfloat" in line
        ]
        assert len(conversions) >= 2
        assert all("rn>" in line and "satfinite>" in line for line in conversions)
        if source_dtype == "f16":
            assert all(
                "from = #nvvm.packfloat_type<f16x2>" in line for line in conversions
            )
        else:
            # Current CUTLASS first widens BF16 exactly to FP32. A later
            # direct-BF16 instruction path must preserve this scalar contract.
            assert "bf16 to f32" in str(module)
            assert "vector<4xbf16> to vector<4xf32>" in str(module)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _convert_rows(
    x: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(x.shape, dtype=dtype, device=x.device)
    maxima = torch.empty((x.size(0),), dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(1))
        values = x[row, columns]
        maxima[row] = values.abs().amax(-1)
        output[row, columns] = values.to(dtype)
    return output, maxima


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("vec_width", [2, 4, 8, 16])
def test_cuda_fp8_conversion_packets_match_scalar(
    dtype: torch.dtype, vec_width: int
) -> None:
    values = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-10,
            -(2.0**-10),
            2.0**-17,
            -(2.0**-17),
            448.0,
            -448.0,
            57344.0,
            -57344.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            -float("nan"),
        ],
        dtype=torch.float32,
    )

    def with_neighbors(boundaries: tuple[float, ...]) -> torch.Tensor:
        points = []
        for boundary in boundaries:
            point = torch.tensor(boundary, dtype=torch.float32)
            points += [
                torch.nextafter(point, torch.tensor(-float("inf"))),
                point,
                torch.nextafter(point, torch.tensor(float("inf"))),
            ]
        points_tensor = torch.stack(points)
        return torch.cat((points_tensor, -points_tensor))

    # An exact E4M3FN midpoint and the E4M3FN/E5M2 overflow boundaries, each
    # with its neighboring FP32 values and both signs.
    values = torch.cat((values, with_neighbors((1.0625, 464.0, 61440.0))))
    assert values.numel() == 32
    # E5M2 midpoints: a normal one and two between subnormals.
    ties = with_neighbors((1.125, 3 * 2.0**-17, 7 * 2.0**-17))
    ties = torch.cat((ties, torch.zeros(32 - ties.numel())))
    x = torch.stack((values, ties)).repeat(19, 1)[:37].to(DEVICE)
    args = (x, dtype)
    # Isolated bindings keep each pinned configuration independent.
    scalar_bound = _convert_rows._bind_isolated(args)
    scalar, scalar_maxima = scalar_bound.compile_config(_config(scalar_bound, 1))(*args)
    packet_bound = _convert_rows._bind_isolated(args)
    packet, packet_maxima = packet_bound.compile_config(
        _config(packet_bound, vec_width)
    )(*args)
    assert torch.equal(packet.view(torch.uint8), scalar.view(torch.uint8))
    # CuTe's cvt.rn.satfinite saturates finite overflow and infinities, and
    # canonicalizes both NaN signs to 0x7f, for both formats. PyTorch's cast
    # differs by format and version, so spell those bytes out in the oracle.
    maximum = torch.finfo(dtype).max
    reference_bytes = torch.where(
        torch.isnan(x),
        0x7F,
        x.clamp(-maximum, maximum).to(dtype).view(torch.uint8),
    ).to(torch.uint8)
    assert torch.equal(packet.view(torch.uint8), reference_bytes)
    torch.testing.assert_close(
        packet_maxima, scalar_maxima, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("destination_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("vec_width", [2, 4, 8, 16])
def test_cuda_half_packet_conversion_matches_every_scalar_encoding(
    source_dtype: torch.dtype, destination_dtype: torch.dtype, vec_width: int
) -> None:
    from helion._compiler.cute import vec_utils
    from helion.runtime import default_cute_launcher

    destination = (
        cutlass.Float8E4M3FN
        if destination_dtype is torch.float8_e4m3fn
        else cutlass.Float8E5M2
    )
    helper = (
        vec_utils.store_fp8e4m3fn_vec
        if destination_dtype is torch.float8_e4m3fn
        else vec_utils.store_fp8e5m2_vec
    )

    @cute.kernel
    def convert(inputs: cute.Tensor, output: cute.Tensor, packed: cutlass.Constexpr):
        base = (
            cutlass.Int32(cute.arch.block_idx()[0]) * (128 * vec_width)
            + cutlass.Int32(cute.arch.thread_idx()[0]) * vec_width
        )
        if packed:
            values = []
            for lane in cutlass.range_constexpr(vec_width):
                values.append(inputs[base + lane])
            helper(output.iterator + base, values)
        else:
            for lane in cutlass.range_constexpr(vec_width):
                output[base + lane] = destination(inputs[base + lane]).bitcast(
                    cutlass.Uint8
                )

    # All 65,536 source encodings cover every rounding transition, signed
    # zero, subnormal, finite overflow, infinity, and NaN sign/payload.
    x = (
        torch.arange(65536, dtype=torch.int32, device=DEVICE)
        .to(torch.uint16)
        .view(source_dtype)
    )
    original = torch.empty_like(x, dtype=torch.uint8)
    packed = torch.empty_like(original)
    grid = (x.numel() // (128 * vec_width),)
    default_cute_launcher(convert, grid, x, original, False, block=(128, 1, 1))
    default_cute_launcher(convert, grid, x, packed, True, block=(128, 1, 1))
    assert torch.equal(packed, original)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _quantize_twice(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    second = torch.empty(x.shape, dtype=torch.float8_e5m2, device=x.device)
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(1))
        data = x[row, columns].float()
        scaled = data / torch.clamp(data.abs().amax(-1), min=1)[:, None]
        first[row, columns] = scaled.to(torch.float8_e4m3fn)
        second[row, columns] = (scaled * 2).to(torch.float8_e5m2)
    return first, second


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("vec_width", [2, 4, 8, 16])
def test_cuda_second_and_per_lane_fp8_packets_match_scalar(vec_width: int) -> None:
    # Every FP8 store of a row loop gets its own packet, and Float64 lanes
    # take the per-lane conversion; both must store exactly their own bytes.
    x = torch.randn((37, 32), dtype=torch.bfloat16, device=DEVICE) * 8
    scalar_bound = _quantize_twice._bind_isolated((x,))
    expected = scalar_bound.compile_config(_config(scalar_bound, 1))(x)
    packet_bound = _quantize_twice._bind_isolated((x,))
    config = _config(packet_bound, vec_width)
    code = packet_bound.to_code(config)
    assert code.count("_cute_store_fp8e4m3fn_vec(") == 1, code
    assert code.count("_cute_store_fp8e5m2_vec(") == 1, code
    actual = packet_bound.compile_config(config)(x)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))

    wide = x.double()
    args = (wide, torch.float8_e4m3fn)
    scalar_bound = _convert_rows._bind_isolated(args)
    scalar, _ = scalar_bound.compile_config(_config(scalar_bound, 1))(*args)
    packet_bound = _convert_rows._bind_isolated(args)
    config = _config(packet_bound, vec_width)
    assert "_cute_store_fp8e4m3fn_vec(" in packet_bound.to_code(config)
    packet, _ = packet_bound.compile_config(config)(*args)
    assert torch.equal(packet.view(torch.uint8), scalar.view(torch.uint8))
