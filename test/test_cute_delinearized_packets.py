"""CuTe packet loads for de-linearized subscripts ``x[e // d, e % d]``.

A flat index ``e = row * d + tile.index`` read through rows of ``d``
contiguous elements (the planes of an NCHW tensor viewed as ``[N, H*W, C]``)
is contiguous across an aligned V-wide chunk of lanes when the row offset and
``d`` are multiples of V: the chunk stays inside one row, so ``e // d`` is one
row for the packet and ``e % d`` advances with stride 1.  The packet is
guarded by the mask terms over ``e`` evaluated at the lane base, and the
per-lane quotient and remainder become one per chunk.  An offset or a ``d``
that is not a multiple of V keeps the scalar loads.
"""

from __future__ import annotations

import ast
import operator
import re
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._compiler.cute.memory_ops import _cute_lane_affine_alignment
from helion._compiler.cute.memory_ops import _cute_lane_uniform_predicate
from helion._compiler.cute.memory_ops import _cute_tile_unroll_uniform_definitions
from helion._compiler.cute.memory_ops import _CuteLaneIndexFacts
from helion._compiler.cute.memory_ops import _CuteVloopScope
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def split_rows(
    x: torch.Tensor, splits: hl.constexpr, shift: hl.constexpr
) -> torch.Tensor:
    # x: [N, H*W, C], a permuted view of an NCHW tensor (H*W has stride 1).
    # Split s reads the flat (n, h, w) elements s * rows + shift onwards in a
    # device loop, as BatchNorm's split Welford partials do.
    batch, hw, channels = x.shape
    total = batch * hw
    rows = (total + splits - 1) // splits
    out = torch.empty([splits, rows, channels], dtype=x.dtype, device=x.device)
    for tile_s, tile_c in hl.tile([splits, channels]):
        for tile_r in hl.tile(rows):
            r = tile_s.index[:, None] * rows + tile_r.index[None, :] + shift
            valid = (r < total) & (tile_r.index[None, :] < rows)
            out[tile_s, tile_r, tile_c] = hl.load(
                x, [r // hw, r % hw, tile_c], extra_mask=valid[:, :, None]
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def split_rows_grid(x: torch.Tensor, splits: hl.constexpr) -> torch.Tensor:
    # The same read with the flat axis tiled by the launch grid.
    batch, hw, channels = x.shape
    total = batch * hw
    rows = (total + splits - 1) // splits
    out = torch.empty([splits, rows, channels], dtype=x.dtype, device=x.device)
    for tile_s, tile_r, tile_c in hl.tile([splits, rows, channels]):
        r = tile_s.index[:, None] * rows + tile_r.index[None, :]
        out[tile_s, tile_r, tile_c] = hl.load(
            x, [r // hw, r % hw, tile_c], extra_mask=(r < total)[:, :, None]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def float_shift_rows(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    # e = tile.index + int(8 * shift) through rows of 16: with a float shift
    # (0.5) the row offset is no multiple of 8, so an 8-wide chunk can
    # straddle two rows.
    out = torch.empty([shifts.size(0), 64], dtype=x.dtype, device=x.device)
    for tile_r, tile_l in hl.tile([shifts.size(0), 64]):
        s = (shifts[tile_r] * 8).to(torch.int32)
        e = tile_l.index[None, :] + s[:, None]
        out[tile_r, tile_l] = hl.load(x, [e // 16, e % 16])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def wrapped_descending_rows(x: torch.Tensor) -> torch.Tensor:
    # e = (-i - 2**31) in the 32-bit index type, widened, plus 2 i: lane 0's
    # term is -2**31 and each later lane's wraps to 2**31 - i, so e jumps
    # from -2**31 to 2**31 + 1 between the first two columns.
    out = torch.empty_like(x)
    for tile_r, tile_l in hl.tile(x.size()):
        i = tile_l.index[None, :] + tile_r.index[:, None] * 0
        e = (-i + (-2147483648)).to(torch.int64) + i + i
        out[tile_r, tile_l] = hl.load(x, [(e // 16) % 3, e % 16])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def shifted_columns(x: torch.Tensor, shift: hl.constexpr) -> torch.Tensor:
    # Columns ``tile.index + shift`` with a negative literal shift.
    out = torch.zeros_like(x)
    for tile0, tile1 in hl.tile(x.size()):
        out[tile0, tile1] = hl.load(
            x, [tile0, tile1.index + shift], extra_mask=(tile1.index >= -shift)[None, :]
        )
    return out


_FLOAT_SHIFT_CONFIG = helion.Config(
    block_sizes=[1, 64], num_threads=[0, 8], cute_vector_widths=[1, 8]
)
_SHIFTED_COLUMNS_CONFIG = helion.Config(
    block_sizes=[1, 2048], num_threads=[0, 512], cute_vector_widths=[1, 4]
)
_WRAPPED_DESCENDING_CONFIG = helion.Config(
    block_sizes=[1, 4], num_threads=[0, 1], cute_vector_widths=[1, 4]
)


def _float_shift_args(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    full = torch.randn(64, 32, device=device).to(torch.bfloat16)
    shifts = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device)
    return full[:, :16], shifts


def _rows(batch: int, hw: int, device: str = "cpu") -> torch.Tensor:
    planes = torch.randn(batch, 16, hw, device=device).to(torch.bfloat16)
    return planes.permute(0, 2, 1)


def _config(kernel: object, vec_width: int) -> helion.Config:
    if kernel is split_rows_grid:
        # A 64-row grid tile of 8 threads.
        return helion.Config(
            block_sizes=[1, 64, 1],
            num_threads=[1, 8, 1],
            cute_vector_widths=[1, vec_width, 1],
        )
    # A 128-row tile of the device loop over 16 threads.
    return helion.Config(
        block_sizes=[1, 1, 128],
        num_threads=[1, 0, 16],
        cute_vector_widths=[1, 1, vec_width],
    )


def _args(
    kernel: object, batch: int, hw: int, shift: int, device: str = "cpu"
) -> tuple[object, ...]:
    x = _rows(batch, hw, device)
    return (x, 4) if kernel is split_rows_grid else (x, 4, shift)


def _code(kernel: object, batch: int, hw: int, vec_width: int, shift: int = 0) -> str:
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, _args(kernel, batch, hw, shift))
        return bound.to_code(_config(kernel, vec_width))


def _vector_loop(code: str) -> ast.For:
    """The single constexpr V-loop of the generated kernel."""
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("vec_lane_")
    ]
    assert len(loops) == 1, code
    return loops[0]


def _packets(code: str) -> list[ast.Call]:
    """The hoisted ``_tile_unroll_vec_* = cute.arch.load(...)`` packet loads."""
    return [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_tile_unroll_vec_")
        and isinstance(node.value, ast.Call)
    ]


def _scalar_x_loads(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and "x.iterator" in ast.unparse(call.func.value)
    ]


@pytest.mark.parametrize(
    ("kernel", "lane"),
    [(split_rows, "2"), (split_rows_grid, "1")],
    ids=["device_loop", "grid"],
)
def test_rows_load_one_guarded_packet_per_chunk(kernel: object, lane: str) -> None:
    # 192 = 24 * 8 and the split offsets s * 192 are multiples of 8.
    code = _code(kernel, 4, 192, 8)
    (packet,) = _packets(code)
    assert (
        ast.unparse(packet.args[1])
        == "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"
    )
    select = packet.args[0]
    assert isinstance(select, ast.IfExp), ast.unparse(packet)
    pointer, guard, anchor = (
        ast.unparse(node) for node in (select.body, select.test, select.orelse)
    )
    # One row and the in-row offset of the chunk's first element.
    row = rf"\(indices_0 \* 192 \+ lane_base_{lane}( \+ 0)?\) // 192"
    assert re.search(
        rf"cutlass\.Int32\({row}\)\) \* cutlass\.Int32\(x\.layout\.stride\[0\]\)",
        pointer,
    ), pointer
    assert re.search(
        rf"_cute_python_mod\(cutlass\.Int32\(indices_0 \* 192 \+ lane_base_{lane}( \+ 0)?\), cutlass\.Int32\(192\)\)",
        pointer,
    ), pointer
    # The masks over ``e`` guard the whole packet at the lane base, and a
    # masked thread reads the first element: its row may not exist.
    assert re.search(rf"cutlass\.Int32\({row}\) < 4", guard), guard
    assert re.search(
        rf"operator\.lt\(indices_0 \* 192 \+ lane_base_{lane}( \+ 0)?, 768\)", guard
    ), guard
    assert anchor.count("cutlass.Int32(0) *") == 3, anchor
    assert f"vec_lane_{lane}" not in ast.unparse(packet), ast.unparse(packet)
    vloop = _vector_loop(code)
    body = ast.unparse(vloop)
    assert not _scalar_x_loads(vloop), body
    # The per-lane quotient and remainder of the mask become one per chunk.
    assert re.search(
        rf"= \(v_\d+ \+ lane_base_{lane}( \+ v_\d+)?\) // v_\d+\n", body
    ), body
    assert re.search(
        rf"= _cute_python_mod\(cutlass\.Int32\(v_\d+ \+ lane_base_{lane}( \+ v_\d+)?\), cutlass\.Int32\(v_\d+\)\) \+ vec_lane_{lane}\n",
        body,
    ), body


@pytest.mark.parametrize(
    ("batch", "hw", "vec_width", "shift", "packets"),
    [
        # 196 = 24.5 * 8: an aligned chunk of 8 can straddle two rows.
        (8, 196, 8, 0, 0),
        # ... but not a chunk of 4.
        (8, 196, 4, 0, 1),
        # A shift of 4 leaves the chunks of 8 off the row grid.
        (4, 192, 8, 4, 0),
        (4, 192, 8, 8, 1),
    ],
)
def test_packets_need_divisor_and_offset_multiples_of_the_width(
    batch: int, hw: int, vec_width: int, shift: int, packets: int
) -> None:
    code = _code(split_rows, batch, hw, vec_width, shift)
    assert len(_packets(code)) == packets, code
    assert len(_scalar_x_loads(_vector_loop(code))) == 1 - packets, code


_LANE = _CuteLaneIndexFacts(
    "index", "base", "vi", 4, True, frozenset(("offset",)), frozenset(("row",))
)


@pytest.mark.parametrize(
    ("predicate", "uniform"),
    [
        # A quotient row: the chunk never crosses a multiple of 16.
        ("(index + 8 * row) // 16 < 3", True),
        ("cutlass.Int32((index - offset) // 8) == row", True),
        ("(index + 2) // 16 < 3", False),
        ("index // 6 < 3", False),
        # The in-row offset advances with the lane: a bound that is a
        # multiple of the width keeps the chunk on one side.
        ("_cute_python_mod(index + 8 * row, 16) < 12", True),
        ("_cute_python_mod(index + 8 * row, 16) < 14", False),
        ("_cute_python_mod(index, 18) < 12", False),
        ("index // 4 * 4 < index", False),
        # A divisor a 32-bit cast wraps: 4294967280 is -16 as an Int32.
        ("_cute_python_mod(index, cutlass.Int32(4294967280)) < 0", False),
        ("index // cutlass.Int32(4294967280) < 0", False),
    ],
)
def test_quotient_and_remainder_uniformity(predicate: str, uniform: bool) -> None:
    node = ast.parse(predicate, mode="eval").body
    assert _cute_lane_uniform_predicate(node, _LANE) == uniform
    if not uniform:
        return
    namespace = {
        "operator": operator,
        "cutlass": SimpleNamespace(Int32=int),
        "_cute_python_mod": operator.mod,
    }
    for offset in (-8, 0, 12):
        for row in (-3, 0, 5):
            for base in range(-64, 64, 4):
                results = {
                    eval(
                        predicate,
                        {**namespace, "index": base + vi, "offset": offset, "row": row},
                    )
                    for vi in range(4)
                }
                assert len(results) == 1, (predicate, offset, row, base)


@pytest.mark.parametrize(
    ("expression", "form"),
    [
        # A cast of a product of a value the facts do not know to be an
        # integer (a loaded float) is one unknown integer.
        ("index + cutlass.Int32(load * 8)", (1, 1)),
        ("index + cutlass.Int32(row * 8)", (1, 8)),
        ("index + load * 8", None),
        ("index + cutlass.Int32(load) * 8", (1, 8)),
        # Negative literals and negations.
        ("index + -512", (1, 512)),
        ("operator.add(index, -512)", (1, 512)),
        ("index + (-8)", (1, 8)),
        ("operator.neg(row * 4) + index", (1, 4)),
        ("-index", (-1, 0)),
        # A cast keeps the form of ascending lanes, but descending ones may
        # wrap after the first lane (-index - 2**31 as an Int32).
        ("cutlass.Int64(index + row * 8)", (1, 8)),
        ("cutlass.Int64(-index + (-2147483648)) + index + index", None),
    ],
)
def test_lane_affine_alignment_needs_integer_factors(
    expression: str, form: tuple[int, int] | None
) -> None:
    node = ast.parse(expression, mode="eval").body
    assert _cute_lane_affine_alignment(node, _LANE) == form


def test_a_float_shift_keeps_the_scalar_loads() -> None:
    """``s = int(shifts[r] * 8)``: the shift is a multiple of 8 only for an
    integer ``shifts[r]``, so the remainder ``(tile.index + s) % 16`` is not
    proved contiguous and the site keeps its scalar loads."""
    with mock_cuda_unavailable():
        bound = cpu_bind(float_shift_rows, _float_shift_args())
        code = bound.to_code(_FLOAT_SHIFT_CONFIG)
    assert not _packets(code), code
    assert _scalar_x_loads(_vector_loop(code)), code


def test_a_cast_of_descending_lanes_keeps_the_scalar_loads() -> None:
    """The index arithmetic of ``wrapped_descending_rows`` wraps after the
    first column of a packet, so its widened sum is no contiguous run: the
    site keeps its scalar loads."""
    with mock_cuda_unavailable():
        bound = cpu_bind(wrapped_descending_rows, (torch.empty(3, 16),))
        code = bound.to_code(_WRAPPED_DESCENDING_CONFIG)
    assert not _packets(code), code
    assert _scalar_x_loads(ast.parse(code)), code


def test_without_lane_facts_only_the_blocks_own_index_takes_the_lane_base() -> None:
    """A scope without lane facts puts the packet at the block's own lane
    base, which only the block's own index takes: a coordinate shifted from
    it keeps the scalar loads."""
    vloop = ast.parse(
        "for vec_lane_1 in cutlass.range_constexpr(4):\n    col = indices_1 + 3"
    ).body[0]
    assert isinstance(vloop, ast.For)
    scope = _CuteVloopScope(
        list(vloop.body),
        frozenset(("vec_lane_1", "indices_1")),
        [],
        [vloop],
        vloop,
        index_var="indices_1",
    )
    assert _cute_tile_unroll_uniform_definitions(scope, ["row", "col"], 1, None) is None
    hoist = _cute_tile_unroll_uniform_definitions(scope, ["row", "indices_1"], 1, None)
    assert hoist is not None and hoist.lane_base_expr is None


def test_a_negative_column_shift_loads_packets() -> None:
    """``tile.index + -512`` (a multiple of the width) addresses an aligned,
    contiguous packet at the lane base shifted by -512."""
    with mock_cuda_unavailable():
        bound = cpu_bind(shifted_columns, (torch.empty(8, 2048), -512))
        code = bound.to_code(_SHIFTED_COLUMNS_CONFIG)
    (packet,) = _packets(code)
    assert re.search(r"lane_base_1 \+ -512", ast.unparse(packet)), ast.unparse(packet)


def _reference(x: torch.Tensor, splits: int, shift: int) -> torch.Tensor:
    batch, hw, channels = x.shape
    total = batch * hw
    rows = (total + splits - 1) // splits
    flat = x.reshape(total, channels)
    index = torch.arange(splits * rows, device=x.device) + shift
    valid = index < total
    out = torch.zeros(splits * rows, channels, dtype=x.dtype, device=x.device)
    out[valid] = flat[index[valid]]
    return out.view(splits, rows, channels)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("kernel", "batch", "hw", "vec_width", "shift"),
    [
        # Packets; the shift masks the tail of the last split.
        (split_rows, 4, 192, 8, 8),
        # Packets of 4 over rows of 196, and a 392-row loop with a tail tile.
        (split_rows, 8, 196, 4, 0),
        # 196 is no multiple of 8: scalar loads.
        (split_rows, 8, 196, 8, 0),
        # Packets in the launch grid.
        (split_rows_grid, 4, 192, 8, 0),
    ],
    ids=["tail", "width4", "scalar", "grid"],
)
def test_rows_match_eager(
    kernel: object, batch: int, hw: int, vec_width: int, shift: int
) -> None:
    args = _args(kernel, batch, hw, shift, DEVICE)
    bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
    out = bound.compile_config(_config(kernel, vec_width))(*args)
    x = args[0]
    assert isinstance(x, torch.Tensor)
    torch.testing.assert_close(out, _reference(x, 4, shift), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_float_shift_rows_match_eager() -> None:
    x, shifts = _float_shift_args(DEVICE)
    out = float_shift_rows.bind((x, shifts)).compile_config(_FLOAT_SHIFT_CONFIG)(
        x, shifts
    )
    e = torch.arange(64, device=DEVICE)[None, :] + (shifts * 8).to(torch.int32)[:, None]
    torch.testing.assert_close(out, x[e // 16, e % 16], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wrapped_descending_rows_match_eager() -> None:
    x = torch.randn(3, 16, device=DEVICE).to(torch.bfloat16)
    out = wrapped_descending_rows.bind((x,)).compile_config(_WRAPPED_DESCENDING_CONFIG)(
        x
    )
    i = torch.arange(16, device=DEVICE)
    # -i - 2**31 in 32 bits is -2**31 at i = 0 and 2**31 - i after.
    e = torch.where(i == 0, -(2**31), 2**31 + i)
    expected = x[(e // 16) % 3, e % 16][None, :].expand(3, 16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_negative_column_shift_matches_eager() -> None:
    x = torch.randn(8, 2048, device=DEVICE)
    out = shifted_columns.bind((x, -512)).compile_config(_SHIFTED_COLUMNS_CONFIG)(
        x, -512
    )
    expected = torch.zeros_like(x)
    expected[:, 512:] = x[:, :-512]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
