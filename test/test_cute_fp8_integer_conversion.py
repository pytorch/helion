"""Integer and bool values converted to an 8-bit float.

The CuTe DSL converts an integer to a float with one ``arith.sitofp`` /
``uitofp``, and MLIR folds into it the float casts and identities (``x * 1.0``)
that follow: into an 8-bit float, the NVVM backend rejects the conversion (a
bool, an Int32) or, for an 8-bit integer, lowers it as a reinterpretation of
its bits.  The pointwise lowering and the computed fragment's add 0.0 where an
integer or a bool becomes a float in a kernel that holds 8-bit floats.  The
kernels render on CPU; the ``cuda`` tests run them against eager, with
per-element stores and with fp8 packets.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _convert_rows(
    x: torch.Tensor, codes: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # One fp8 row each from a bool, from that bool's Float32, and from an Int8.
    rows, width = x.shape
    from_bool = torch.empty(x.shape, dtype=like.dtype, device=x.device)
    from_float = torch.empty(x.shape, dtype=like.dtype, device=x.device)
    from_int = torch.empty(x.shape, dtype=like.dtype, device=x.device)
    for row in hl.tile(rows):
        data = x[row, :].to(torch.float32)
        keep = data > torch.amax(data, dim=-1, keepdim=True) * 0.5
        from_bool[row, :] = keep.to(like.dtype)
        from_float[row, :] = keep.to(torch.float32).to(like.dtype)
        from_int[row, :] = codes[row, :].to(like.dtype)
    return from_bool, from_float, from_int


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _negated_codes(codes: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # The loop rebinds the integer's float: a zero comes out as -0.0.
    out = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    for tile in hl.tile(codes.size(0)):
        value = codes[tile].to(torch.float32)
        for _ in range(1):
            value = -value
        out[tile] = value.to(like.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _looped_codes(codes: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # A nested loop converts the integer's float, which it does not rebind.
    out = torch.empty([4, codes.size(0)], dtype=like.dtype, device=codes.device)
    for tile in hl.tile(codes.size(0)):
        value = codes[tile].to(torch.float32)
        for i in hl.grid(4):
            out[i, tile] = value.to(like.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _identity_codes(
    codes: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Identities MLIR folds between the integer's float and the conversion.
    times_one = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    over_one = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    minus_zero = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    for tile in hl.tile(codes.size(0)):
        times_one[tile] = (codes[tile] * 1.0).to(like.dtype)
        over_one[tile] = (codes[tile] / 1.0).to(like.dtype)
        minus_zero[tile] = (codes[tile] - 0.0).to(like.dtype)
    return times_one, over_one, minus_zero


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chained_codes(codes: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    out = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    for tile in hl.tile(codes.size(0)):
        out[tile] = codes[tile].to(torch.float16).to(torch.float32).to(like.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_rows(
    codes: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Free hl.arange subscripts: the computed fragment lowers both stores, a
    # bool's and an Int8's conversion to fp8.
    eye = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    out = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    for _ in hl.grid(1):
        index = hl.arange(16)
        rows, cols = index[:, None], index[None, :]
        eye[rows, cols] = (rows == cols).to(like.dtype)
        out[rows, cols] = codes[rows, cols].to(like.dtype)
    return eye, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _split_codes(codes: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # hl.split re-reads its halves' load and builds their casts again.
    n, d = codes.shape
    out = torch.empty([n, d // 2], dtype=like.dtype, device=codes.device)
    for tile_n in hl.tile(n):
        pair = codes[tile_n, :].to(torch.float32).reshape([tile_n, d // 2, 2])
        lo, _ = hl.split(pair)
        out[tile_n, :] = lo.to(like.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _full_codes(
    codes: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # hl.full of an integer element, as a Float32 then converted to fp8, and
    # straight to fp8.
    n = codes.size(0)
    widened = torch.empty([n], dtype=like.dtype, device=codes.device)
    direct = torch.empty([n], dtype=like.dtype, device=codes.device)
    for tile in hl.tile(n):
        widened[tile] = hl.full([tile], codes[0], dtype=torch.float32).to(like.dtype)
        direct[tile] = hl.full([tile], codes[1], dtype=like.dtype)
    return widened, direct


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _full_of_a_size(codes: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # torch.full of a dynamic size, an Int32 scalar of the kernel.
    n = codes.size(0)
    out = torch.empty([n], dtype=like.dtype, device=codes.device)
    for tile in hl.tile(n):
        out[tile] = torch.full([tile], codes.size(1), dtype=torch.float32).to(
            like.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_fill(
    codes: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A computed fragment fills ``out`` with an Int32 element through a
    # Float32 (the ``eye`` store forms the fragment).
    eye = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    out = torch.empty(codes.shape, dtype=like.dtype, device=codes.device)
    for _ in hl.grid(1):
        index = hl.arange(16)
        rows, cols = index[:, None], index[None, :]
        eye[rows, cols] = (rows == cols).to(like.dtype)
        out[rows, cols] = hl.full([16, 16], codes[0, 0], dtype=torch.float32).to(
            like.dtype
        )
    return eye, out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _scalars_of_a_size(
    codes: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A dynamic size, an Int32 scalar of the kernel, through a scalar tensor
    # and stored straight to fp8.
    n = codes.size(0)
    widened = torch.empty([n], dtype=like.dtype, device=codes.device)
    stored = torch.empty([n], dtype=like.dtype, device=codes.device)
    for tile in hl.tile(n):
        widened[tile] = torch.scalar_tensor(codes.size(1), dtype=torch.float32).to(
            like.dtype
        )
        stored[tile] = codes.size(1)
    return widened, stored


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _offset_codes(codes: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + codes[tile].to(torch.float32)
    return out


def _config(bound: BoundKernel, vec_width: int) -> helion.Config:
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


def _args(device: torch.device | str) -> tuple[torch.Tensor, ...]:
    x = torch.randn((37, 32), dtype=torch.bfloat16, device=device)
    codes = torch.randint(-100, 100, (37, 32), dtype=torch.int8, device=device)
    return x, codes


@skipIfRefEager("ref-eager mode generates no code")
@pytest.mark.parametrize("vec_width", [1, 8])
def test_integer_and_bool_values_reach_fp8_through_float32(vec_width: int) -> None:
    x, codes = _args("cpu")
    like = torch.empty(1, dtype=torch.float8_e4m3fn)
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_convert_rows, (x, codes, like))
        code = bound.to_code(_config(bound, vec_width))
    # The 0.0 added to the bool's Float32 (twice: the direct conversion and
    # the one through an explicit Float32) and to the Int8's: no integer
    # reaches an fp8 conversion, with or without packets.
    assert code.count(" + cutlass.Float32(0.0)") == 3, code
    assert code.count("_cute_store_fp8e4m3fn_vec(") == (3 if vec_width > 1 else 0)


def _render(kernel: Kernel[torch.Tensor], args: tuple[object, ...]) -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, args)
        return bound.to_code(bound.config_spec.default_config())


@skipIfRefEager("ref-eager mode generates no code")
def test_the_zero_is_added_where_an_integer_becomes_a_float() -> None:
    like = torch.empty(1, dtype=torch.float8_e4m3fn)
    codes = torch.empty(64, dtype=torch.int8)
    # What follows the conversion follows the 0.0 too: a nested loop's use,
    # identities MLIR folds, a loop rebinding the float (a zero it negates
    # stays -0.0) and other float casts.
    for kernel, conversions in (
        (_looped_codes, 1),
        (_identity_codes, 3),
        (_negated_codes, 1),
        (_chained_codes, 1),
    ):
        code = _render(kernel, (codes, like))
        barriers = re.findall(
            r"= cutlass\.(Float\w+)\(load\w*\) \+ cutlass\.\1\(0\.0\)", code
        )
        assert len(barriers) == conversions, code
    # The computed fragment's conversions take the 0.0 as well.
    code = _render(_fragment_rows, (torch.empty(16, 16, dtype=torch.int8), like))
    assert "fragment_value_" in code, code
    assert code.count(" + cutlass.Float32(0.0)") == 2, code
    # A kernel that holds no 8-bit float converts as before.
    code = _render(_offset_codes, (codes, torch.empty(64)))
    assert "= cutlass.Float32(load_1)\n" in code, code
    assert " + cutlass.Float32(0.0)" not in code, code


@skipIfRefEager("ref-eager mode generates no code")
def test_split_and_full_add_the_zero_too() -> None:
    like = torch.empty(1, dtype=torch.float8_e4m3fn)
    # hl.split's re-read of each half converts the Int8 again.
    code = _render(_split_codes, (torch.empty(16, 8, dtype=torch.int8), like))
    for half in ("split_lo", "split_hi"):
        assert re.search(
            rf"{half} = cutlass\.Float32\(.*\.load\(\)\) \+ cutlass\.Float32\(0\.0\)\n",
            code,
        ), code
    # hl.full of an Int32 element, to a Float32 and to fp8.
    code = _render(_full_codes, (torch.empty(16, dtype=torch.int32), like))
    assert "= cutlass.Float32(load) + cutlass.Float32(0.0)\n" in code, code
    assert (
        "cutlass.Float8E4M3FN(cutlass.Float32(load_1) + cutlass.Float32(0.0))" in code
    ), code
    # torch.full of a dynamic size.
    code = _render(_full_of_a_size, (torch.empty(16, 8, dtype=torch.int8), like))
    assert "= cutlass.Float32(codes_size_1) + cutlass.Float32(0.0)\n" in code, code


@skipIfRefEager("ref-eager mode generates no code")
def test_fragment_fills_and_scalar_stores_add_the_zero_too() -> None:
    like = torch.empty(1, dtype=torch.float8_e4m3fn)
    # A computed fragment's fill of an Int32 element.
    code = _render(_fragment_fill, (torch.empty(16, 16, dtype=torch.int32), like))
    assert "= cutlass.Float32(fragment_buffer[0]) + cutlass.Float32(0.0)\n" in code
    # A dynamic size through torch.scalar_tensor, and stored to an fp8 tensor.
    code = _render(_scalars_of_a_size, (torch.empty(16, 8, dtype=torch.int8), like))
    assert "= cutlass.Float32(codes_size_1) + cutlass.Float32(0.0)\n" in code, code
    assert (
        ".store(cutlass.Float8E4M3FN(cutlass.Float32(codes_size_1) + "
        "cutlass.Float32(0.0)))" in code
    ), code


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_cuda_rebound_and_chained_integer_floats_match_eager(
    dtype: torch.dtype,
) -> None:
    codes = torch.randint(-100, 100, (64,), dtype=torch.int8, device=DEVICE)
    codes[::5] = 0
    like = torch.empty(1, dtype=dtype, device=DEVICE)
    negated = _negated_codes(codes, like)
    expected = (-codes.float()).to(dtype)
    assert torch.equal(negated.view(torch.uint8), expected.view(torch.uint8))
    chained = _chained_codes(codes, like)
    expected = codes.half().float().to(dtype)
    assert torch.equal(chained.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("codes_dtype", [torch.int8, torch.int32])
def test_cuda_looped_and_folded_integer_floats_match_eager(
    dtype: torch.dtype, codes_dtype: torch.dtype
) -> None:
    codes = torch.arange(-128, 128, device=DEVICE).to(codes_dtype)
    like = torch.empty(1, dtype=dtype, device=DEVICE)
    expected = codes.float().to(dtype).view(torch.uint8)
    looped = _looped_codes(codes, like)
    assert torch.equal(looped.view(torch.uint8), expected.expand(4, -1))
    for out in _identity_codes(codes, like):
        assert torch.equal(out.view(torch.uint8), expected)


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_cuda_fragment_conversions_match_eager(dtype: torch.dtype) -> None:
    codes = torch.arange(-128, 128, device=DEVICE).to(torch.int8).view(16, 16)
    like = torch.empty(1, dtype=dtype, device=DEVICE)
    eye, out = _fragment_rows(codes, like)
    expected = torch.eye(16, device=DEVICE).to(dtype)
    assert torch.equal(eye.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(out.view(torch.uint8), codes.to(dtype).view(torch.uint8))


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_cuda_split_and_full_conversions_match_eager(dtype: torch.dtype) -> None:
    codes = torch.arange(-64, 64, device=DEVICE).to(torch.int8).view(16, 8)
    like = torch.empty(1, dtype=dtype, device=DEVICE)
    out = _split_codes(codes, like)
    expected = codes[:, ::2].float().to(dtype)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    values = torch.arange(-100, -84, dtype=torch.int32, device=DEVICE)
    widened, direct = _full_codes(values, like)
    for out, value in ((widened, values[0]), (direct, values[1])):
        expected = value.float().to(dtype).expand(16)
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    out = _full_of_a_size(codes, like)
    expected = torch.full([16], 8.0, device=DEVICE).to(dtype)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    for out in _scalars_of_a_size(codes, like):
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    values = torch.full([16, 16], -100, dtype=torch.int32, device=DEVICE)
    _, out = _fragment_fill(values, like)
    expected = torch.full([16, 16], -100.0, device=DEVICE).to(dtype)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("vec_width", [1, 8])
def test_cuda_fp8_from_integers_and_bools_matches_eager(
    dtype: torch.dtype, vec_width: int
) -> None:
    x, codes = _args(DEVICE)
    like = torch.empty(1, dtype=dtype, device=DEVICE)
    bound = _convert_rows._bind_isolated((x, codes, like))
    outputs = bound.compile_config(_config(bound, vec_width))(x, codes, like)
    data = x.float()
    keep = data > data.amax(-1, keepdim=True) * 0.5
    for out, expected in zip(
        outputs,
        (keep.to(dtype), keep.float().to(dtype), codes.to(dtype)),
        strict=True,
    ):
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
