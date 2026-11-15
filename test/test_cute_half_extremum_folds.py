"""A V-loop's maximum or minimum of bf16 / f16 lanes folded as packed words.

``helion/_compiler/cute/half_extremum_folds.py`` (the pass and why the
result stays bit for bit) with its device helper
``device.half_extrema.half_packet_extremum_f32``.

CPU: the pass on AST snippets (which folds move after their loop, what keeps
a fold in place) and renders of a softmax-statistics row kernel over bf16
and f16 rows.  GPU (``cuda``): those kernels against the pass switched off,
bit for bit, on rows of NaNs, signed zeros, infinities, subnormals and
ordinary values.
"""

from __future__ import annotations

import ast
import contextlib
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import half_extremum_folds
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

PACKED = "_cute_half_packet_extremum_f32("


def _run(source: str, *, capability: tuple[int, int] | None = (10, 0)) -> str:
    body = ast.parse(textwrap.dedent(source)).body
    body = half_extremum_folds.pack_half_extremum_folds(
        body, target_device_capability=capability
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _same(source: str) -> str:
    return ast.unparse(ast.parse(textwrap.dedent(source)))


# --------------------------------------------------------------------------
# The pass on AST snippets
# --------------------------------------------------------------------------

# A loaded packet: an SSA vector, which no store can change.
_LOAD = (
    "packet = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))\n"
)
_MAX_FOLD = f"""\
{_LOAD}acc = cutlass.Float32(float('-inf'))
cache = cute.make_rmem_tensor(16, cutlass.Float32)
for v in cutlass.range_constexpr(16):
    load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
    v_0 = cutlass.Float32(load)
    cache[v] = cutlass.Float32(v_0)
    acc = cute.arch.fmax(acc, cutlass.Float32(load), nan=True)
parts.append(acc)
"""


def test_a_bf16_max_fold_moves_after_its_loop() -> None:
    expected = """\
        packet = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        acc = cutlass.Float32(float('-inf'))
        cache = cute.make_rmem_tensor(16, cutlass.Float32)
        for v in cutlass.range_constexpr(16):
            load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
            v_0 = cutlass.Float32(load)
            cache[v] = cutlass.Float32(v_0)
        acc = cute.arch.fmax(acc, _cute_half_packet_extremum_f32(packet, 16, cutlass.BFloat16, 'max'), nan=True)
        parts.append(acc)
        """
    assert _run(_MAX_FOLD) == _same(expected)


@pytest.mark.parametrize("capability", [(7, 5), None], ids=["sm75", "unknown"])
def test_targets_without_packed_half_extrema_keep_the_fold(
    capability: tuple[int, int] | None,
) -> None:
    assert _run(_MAX_FOLD, capability=capability) == _same(_MAX_FOLD)


def test_an_f16_min_fold_through_an_fp32_local_with_swapped_operands() -> None:
    source = """\
        packet = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        for v in cutlass.range_constexpr(8):
            load = cutlass.Uint16(packet[v]).bitcast(cutlass.Float16)
            v_0 = cutlass.Float32(load)
            acc = cute.arch.fmin(cutlass.Float32(v_0), acc, nan=True)
        """
    expected = """\
        packet = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        for v in cutlass.range_constexpr(8):
            load = cutlass.Uint16(packet[v]).bitcast(cutlass.Float16)
            v_0 = cutlass.Float32(load)
        acc = cute.arch.fmin(_cute_half_packet_extremum_f32(packet, 8, cutlass.Float16, 'min'), acc, nan=True)
        """
    assert _run(source) == _same(expected)


def test_both_extrema_of_a_lane_fold() -> None:
    # The packet is an alias of the loaded one, as a pipelined load's is.
    source = """\
        piped = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        packet = piped
        for v in cutlass.range_constexpr(2):
            x = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
            hi = cute.arch.fmax(hi, cutlass.Float32(x), nan=True)
            lo = cute.arch.fmin(lo, cutlass.Float32(x), nan=True)
        """
    expected = """\
        piped = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        packet = piped
        for v in cutlass.range_constexpr(2):
            x = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
        hi = cute.arch.fmax(hi, _cute_half_packet_extremum_f32(packet, 2, cutlass.BFloat16, 'max'), nan=True)
        lo = cute.arch.fmin(lo, _cute_half_packet_extremum_f32(packet, 2, cutlass.BFloat16, 'min'), nan=True)
        """
    assert _run(source) == _same(expected)


@pytest.mark.parametrize(
    "source",
    [
        # An odd extent leaves a half word.
        _MAX_FOLD.replace("range_constexpr(16)", "range_constexpr(15)"),
        # The running value is read by another statement of the loop.
        _MAX_FOLD.replace(
            "    cache[v] = cutlass.Float32(v_0)\n", "    cache[v] = acc + v_0\n"
        ),
        # Not the loop's own lane.
        _MAX_FOLD.replace("packet[v]", "packet[v + 1]"),
        # The fold does not propagate NaN.
        _MAX_FOLD.replace(", nan=True)", ")"),
        # The packet is passed to a call in the loop.
        _MAX_FOLD.replace(
            "    cache[v] = cutlass.Float32(v_0)\n", "    touch(packet)\n"
        ),
        # An fp32 lane.
        _MAX_FOLD.replace("cutlass.BFloat16", "cutlass.Float32"),
        # A lane loop, not a V-loop.
        _MAX_FOLD.replace("cutlass.range_constexpr(16)", "range(16)"),
        # A lane may skip the fold.
        _MAX_FOLD.replace(
            "    cache[v] = cutlass.Float32(v_0)\n",
            "    if v_0 > 0:\n        continue\n",
        ),
        # The lane is assigned after the fold: the fold reads the previous
        # trip's lane (or the one from before the loop).
        _MAX_FOLD.replace(
            "    load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)\n", ""
        ).replace(
            "    acc = cute.arch.fmax(acc, cutlass.Float32(load), nan=True)\n",
            "    acc = cute.arch.fmax(acc, cutlass.Float32(load), nan=True)\n"
            "    load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)\n",
        ),
        # The fp32 local widens the previous trip's lane: its lane is
        # assigned after it.
        _MAX_FOLD.replace(
            "    load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)\n"
            "    v_0 = cutlass.Float32(load)\n",
            "    v_0 = cutlass.Float32(load)\n"
            "    load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)\n",
        ).replace("cutlass.Float32(load), nan=True", "cutlass.Float32(v_0), nan=True"),
        # A branch may rebind the lane between its assignment and the fold.
        _MAX_FOLD.replace(
            "    cache[v] = cutlass.Float32(v_0)\n",
            "    if flag:\n        load = other\n",
        ),
        # The body rebinds the lane index.
        _MAX_FOLD.replace(
            "    load = cutlass.Uint16(packet[v])",
            "    v = 0\n    load = cutlass.Uint16(packet[v])",
        ),
        # A packet of unknown origin may be storage another name writes.
        _MAX_FOLD.replace(_LOAD, ""),
        # A register fragment written through an alias: the moved fold would
        # read lanes the later trips overwrite.
        _MAX_FOLD.replace(
            _LOAD,
            "packet = cute.make_rmem_tensor(16, cutlass.Uint16)\nalias = packet\n",
        ).replace(
            "    cache[v] = cutlass.Float32(v_0)\n",
            "    alias[15 - v] = cutlass.Uint16(0)\n",
        ),
        # So may a resident load's fragment.
        _MAX_FOLD.replace(
            _LOAD,
            "packet = _cute_resident_load_vector(t, offset, 16)\nalias = packet\n",
        ).replace(
            "    cache[v] = cutlass.Float32(v_0)\n",
            "    alias[15 - v] = cutlass.Uint16(0)\n",
        ),
        # One of the packet's values is not a load.
        _MAX_FOLD.replace(_LOAD, _LOAD + "if flag:\n    packet = other\n"),
    ],
    ids=[
        "odd_extent",
        "accumulator_read",
        "shifted_lane",
        "no_nan_propagation",
        "packet_escapes",
        "fp32_lane",
        "lane_loop",
        "early_exit",
        "lane_assigned_after_the_fold",
        "local_widens_the_previous_lane",
        "lane_rebound_in_a_branch",
        "lane_index_rebound",
        "packet_not_loaded",
        "fragment_written_through_an_alias",
        "resident_fragment_written_through_an_alias",
        "packet_rebound_to_another_value",
    ],
)
def test_folds_that_keep_their_place(source: str) -> None:
    assert _run(source) == _same(source)


def test_a_resident_fragment_read_only_by_element_is_a_loaded_packet() -> None:
    # A resident load gives a register fragment.  While the body only reads
    # its elements, no store reaches it and the fold moves; listed, passed
    # to a call, aliased or stored into, it may be written.
    source = _MAX_FOLD.replace(
        _LOAD, "packet = _cute_resident_load_vector(t, offset, 16)\n"
    )
    assert PACKED in _run(source)
    for use in ("frags = [packet]", "touch(packet)", "alias = packet", "packet[0] = 1"):
        written = source.replace("acc = cutlass", f"{use}\nacc = cutlass", 1)
        assert half_extremum_folds.loaded_packets(ast.parse(written).body) == set()


def test_a_packet_read_back_from_a_list_of_loaded_packets_folds() -> None:
    # The loads-first lane unroll stashes each lane's loaded packet in a list
    # and reads it back: still a loaded packet.  A list that also holds a
    # register fragment may hand the fold storage the loop writes.
    source = """\
        loads = []
        for lane in range(2):
            packet = cute.arch.load(ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
            loads.append(packet)
        for lane in range(2):
            packet = loads[lane]
            for v in cutlass.range_constexpr(16):
                load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
                acc = cute.arch.fmax(acc, cutlass.Float32(load), nan=True)
        """
    assert PACKED in _run(source)
    mixed = textwrap.dedent(source).replace(
        "loads = []\n", "loads = [cute.make_rmem_tensor(16, cutlass.Uint16)]\n"
    )
    assert _run(mixed) == _same(mixed)


def test_a_packet_of_odd_length_folds_lane_by_lane() -> None:
    # Its last lane has no partner in a 32-bit word: even for an even
    # ``count``, the helper folds lane by lane rather than bitcast the packet
    # to words.
    half_extrema = pytest.importorskip("helion._compiler.cute.device.half_extrema")

    class VectorType:
        element_type = SimpleNamespace(width=16)
        shape = (3,)

        def __init__(self, mlir_type: object) -> None:
            pass

        @staticmethod
        def isinstance(mlir_type: object) -> bool:
            return True

    cutlass = SimpleNamespace(
        extract_mlir_values=lambda packet: [SimpleNamespace(type=None)],
        Float32=lambda value: value,
        Uint16=lambda value: SimpleNamespace(bitcast=lambda half: value),
    )
    cute = SimpleNamespace(arch=SimpleNamespace(fmax=lambda a, b, nan: max(a, b)))
    with (
        patch.object(half_extrema, "cutlass", cutlass),
        patch.object(half_extrema, "cute", cute),
        patch.object(half_extrema.ir, "VectorType", VectorType),
        patch.object(half_extrema.vector, "bitcast", side_effect=AssertionError),
    ):
        packet = [3.0, 7.0, 5.0]
        assert half_extrema.half_packet_extremum_f32(packet, 2, None, "max") == 7.0


# --------------------------------------------------------------------------
# Renders
# --------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _softmax_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    row_max_out = torch.empty([m], dtype=torch.float32, device=x.device)
    row_sum_out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(32, n)
    for tile_m in hl.tile(m):
        row_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            vals = x[tile_m, tile_n].to(torch.float32)
            block_max = torch.amax(vals, dim=-1)
            block_sum = torch.sum(torch.exp(vals - block_max[:, None]), dim=-1)
            new_max = torch.maximum(row_max, block_max)
            row_sum = row_sum * torch.exp(row_max - new_max) + block_sum * torch.exp(
                block_max - new_max
            )
            row_max = new_max
        row_max_out[tile_m] = row_max
        row_sum_out[tile_m] = row_sum
    return row_max_out, row_sum_out


_CONFIG = helion.Config(
    block_sizes=[512, 1],
    num_threads=[32, 0],
    cute_vector_widths=[16, 8],
    cute_pair_lane_math=True,
)


@contextlib.contextmanager
def _pass_off() -> Iterator[None]:
    with patch.object(
        half_extremum_folds, "pack_half_extremum_folds", lambda body, **kwargs: body
    ):
        yield


def _render(dtype: torch.dtype, config: helion.Config = _CONFIG) -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_softmax_stats, (torch.empty(4, 2048, dtype=dtype),))
        return bound.to_code(config)


@pytest.mark.parametrize(
    ("dtype", "half"),
    [(torch.bfloat16, "cutlass.BFloat16"), (torch.float16, "cutlass.Float16")],
    ids=["bf16", "f16"],
)
def test_softmax_stats_fold_their_block_max_as_words(
    dtype: torch.dtype, half: str
) -> None:
    code = _render(dtype)
    assert f"{PACKED}_tile_unroll_vec_0_0, 16, {half}, 'max')" in code
    assert "cute.arch.fmax(_helion_vfold_acc_0, cutlass.Float32(load)" not in code
    with _pass_off():
        plain = _render(dtype)
    assert PACKED not in plain
    assert (
        "cute.arch.fmax(_helion_vfold_acc_0, cutlass.Float32(load), nan=True)" in plain
    )


def test_fp32_rows_keep_their_fold() -> None:
    code = _render(torch.float32)
    assert PACKED not in code


def test_the_knob_default_keeps_the_fold() -> None:
    values = {k: v for k, v in _CONFIG.config.items() if k != "cute_pair_lane_math"}
    code = _render(torch.bfloat16, helion.Config(**values))
    assert PACKED not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_max(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m] = torch.amax(x[tile_m, :], dim=-1)
    return out


@pytest.mark.parametrize(
    ("dtype", "offered"),
    [(torch.bfloat16, True), (torch.float16, True), (torch.float32, False)],
    ids=["bf16", "f16", "fp32"],
)
def test_a_half_precision_max_offers_the_knob(
    dtype: torch.dtype, offered: bool
) -> None:
    # No exp, division or fp32 arithmetic: only the half2 fold can use the
    # knob, so an fp32 row's maximum does not offer it.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_row_max, (torch.empty(4, 2048, dtype=dtype),))
    keys = {name for name, _, _ in bound.config_spec.flat_key_layout()}
    assert ("cute_pair_lane_math" in keys) is offered


# --------------------------------------------------------------------------
# GPU: the packed folds against the pass switched off, bit for bit
# --------------------------------------------------------------------------


def _rows(dtype: torch.dtype) -> torch.Tensor:
    """Rows of ordinary values, of -0.0 and +0.0 maxima, of NaNs (one per
    row and one per lane word), of infinities and of subnormals."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(8, 2048, generator=generator)
    rows[1] = -torch.rand(2048, generator=generator)
    rows[1, ::3] = -0.0
    rows[1, 1::7] = 0.0
    rows[2] = -0.0
    rows[3, 1000] = float("nan")
    rows[4, ::2] = float("nan")
    rows[5, 17] = float("inf")
    rows[5, 18] = float("-inf")
    rows[6] = float("-inf")
    tiny = torch.finfo(dtype).smallest_normal
    rows[7] = torch.rand(2048, generator=generator) * tiny * -1.0
    rows[7, 5] = tiny / 4
    return rows.to(dtype)


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_cuda_packed_folds_match_the_scalar_fold_bit_for_bit(
    dtype: torch.dtype,
) -> None:
    x = _rows(dtype).to(DEVICE)

    def outputs() -> list[torch.Tensor]:
        bound = _softmax_stats._bind_isolated((x,))
        return list(bound.compile_config(_CONFIG)(x))

    packed = outputs()
    with _pass_off():
        plain = outputs()
    for a, b in zip(packed, plain, strict=True):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    row_max = packed[0].cpu()
    assert row_max[3].isnan() and row_max[4].isnan()
    assert row_max[2].item() == 0.0 and str(row_max[2].item()) == "-0.0"
    assert row_max[1].item() == 0.0 and str(row_max[1].item()) == "0.0"
