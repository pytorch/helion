"""A flattened gather under a window mask (an unfold, ``out[n, k]
= flat[n + channels * (k - pad)]`` inside the sequence) loads predicated
packets along its lane tile: the flat-gather matcher sees through
``tile.index[:, None]``, and the V-loop sink compares the mask at the chunk
base where that is proved the same for every lane; a mask that is not one
value per packet, and a view the packets cannot follow, keep scalar loads,
and no comparison outside the predicate of a load the sink widens moves.
The sink's second try is not the gathers' alone: a tile load under a mask
over a lane quotient takes it too.  CPU tests of the generated code, and
``cuda``-marked tests that the packet and the scalar forms write what eager
computes."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.flat_gather_lane import FLAT_GATHER_FACTOR_SPECIALIZATION_KEY
from helion._compiler.cute.flat_gather_lane import _is_broadcast_view
from helion._compiler.cute.flat_gather_lane import match_flat_gather_lane
from helion._compiler.cute.lane_invariant_division import uniform_load_predicates
from helion._compiler.cute.sink_vector_loops import _guards_recorded_load
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.unfold_kernels import register_tile_config
from helion._testing.unfold_kernels import sliding_windows
from helion._testing.unfold_kernels import sliding_windows_into
from helion._testing.unfold_kernels import sliding_windows_reference
import helion.language as hl
from helion.language import memory_ops

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

# 8 threads of one 8-lane bf16 packet each on 64 flat elements, the 3 window
# slots a per-thread lane loop over one 4-slot tile.
_TILE = register_tile_config(threads=8, window_block=4, staged=False)


def _code(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    config: dict[str, object],
    **settings: object,
) -> str:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(
            helion.kernel(
                kernel.fn, backend="cute", **{"static_shapes": True, **settings}
            ),
            args,
        )
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config.from_dict(config))
        )


def _windows(batch: int = 2, seq: int = 8, channels: int = 16) -> torch.Tensor:
    return torch.empty(batch, seq, channels, dtype=torch.bfloat16)


def _guard_terms(guard: ast.expr) -> list[str]:
    """Flatten only `and`, preserving every guard term in evaluation order."""
    if isinstance(guard, ast.BoolOp) and isinstance(guard.op, ast.And):
        return [term for child in guard.values for term in _guard_terms(child)]
    return [ast.unparse(guard)]


def _check_window_packets(code: str) -> None:
    packets = [
        call
        for call in ast.walk(ast.parse(code))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_cute_load_v16b_if"
    ]
    assert len(packets) == 4
    for slot, packet in enumerate(packets):
        assert ast.unparse(packet.args[0]).startswith("flat.iterator + ")
        assert _guard_terms(packet.args[1]) == [
            f"v_11_u{slot}",
            "lane_base_0 - tile_offset_0 < 256",
            f"mask_1_u{slot}",
            f"cutlass.Int32(lane_base_0 + v_15_u{slot}) < 256",
        ]


def test_a_window_gather_loads_one_predicated_packet_per_slot() -> None:
    code = _code(sliding_windows, (_windows(), 3, 1), _TILE)
    # One packet per unrolled slot of the 4-slot tile, issued before the
    # stores; no scalar load is left.
    _check_window_packets(code)
    assert ".load()" not in code
    # The gather's address carries the lane tile's index as its own term
    # (``i[:, None]`` on a tile index traces to ``load(tile_index, ...)``).
    assert "cutlass.Int32(lane_base_0) * cutlass.Int32(flat.layout.stride[0])" in code
    # The window mask reads the sequence position of the chunk base, and the
    # gather's own bound compares the base: one value per packet.
    assert "v_1 = lane_base_0 // v_0" in code
    assert "indices_0" not in code[: code.index("for vec_lane_0 in")]
    last_load = code.rindex("_cute_load_v16b_if(")
    assert last_load < code.index("(out.iterator + ")


@pytest.mark.parametrize(
    ("args", "settings"),
    [
        # 12 channels: ``(i // 12) % seq`` changes inside an 8-lane chunk.
        ((_windows(channels=12), 3, 1), {}),
        # Dynamic shapes: the bounds are kernel arguments, not multiples of
        # the packet the generated code can read.
        ((_windows(), 3, 1), {"static_shapes": False}),
    ],
    ids=["divisor_not_a_multiple", "dynamic_shapes"],
)
def test_a_mask_that_is_not_one_value_per_packet_keeps_scalar_loads(
    args: tuple[object, ...], settings: dict[str, object]
) -> None:
    code = _code(sliding_windows, args, _TILE, **settings)
    assert "_cute_load_v16b_if(" not in code
    assert code.count(".load()") == 1
    # Nothing sank, so the knob left no trace: the knob-off code.
    off = {**_TILE, "cute_vloop_sink": False}
    assert code == _code(sliding_windows, args, off, **settings)


def test_the_gathers_of_views_keep_scalar_loads() -> None:
    buffer = torch.empty(600, dtype=torch.bfloat16)
    out = torch.empty(256, 3, dtype=torch.bfloat16)
    for flat in (buffer[3:259], buffer[:512:2]):
        # A base 6 bytes into its storage, and a stride of two elements.
        code = _code(sliding_windows_into, (flat, out, 8, 16, 1), _TILE)
        assert "_cute_load_v16b_if(" not in code
    aligned = _code(sliding_windows_into, (buffer[8:264], out, 8, 16, 1), _TILE)
    assert aligned.count("_cute_load_v16b_if(flat.iterator") == 4


def _launch(
    kernel: Kernel[Any], args: tuple[object, ...], config: dict[str, object]
) -> tuple[str, torch.Tensor]:
    """The code ``kernel`` runs at ``config`` on the GPU, and its result."""
    pinned = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict(config),
    )
    bound = pinned.bind(args)
    code = bound.to_code(bound.config_spec.normalized_config(pinned.configs[0]))
    return code, pinned(*args)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("shape", "dtype", "window", "config", "packets"),
    [
        # 1920 bf16 elements in tiles of 256 (one warp of 8-lane packets):
        # seven full tiles and a partial eighth, 9 slots of a 16-slot tile.
        (
            (3, 40, 16),
            torch.bfloat16,
            9,
            register_tile_config(threads=32, window_block=16, staged=False),
            True,
        ),
        # fp32: four-element packets, three slots of a four-slot tile.
        (
            (2, 20, 16),
            torch.float32,
            3,
            register_tile_config(threads=32, window_block=4, vec=4, staged=False),
            True,
        ),
        # 12 channels: the mask differs inside a packet, the loads stay
        # scalar.
        (
            (3, 40, 12),
            torch.bfloat16,
            9,
            register_tile_config(threads=32, window_block=16, staged=False),
            False,
        ),
    ],
    ids=["bf16", "fp32", "mask_inside_a_packet"],
)
def test_window_gathers_match_eager(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    window: int,
    config: dict[str, object],
    packets: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    pad = window // 2
    code, out = _launch(sliding_windows, (x, window, pad), config)
    assert ("_cute_load_v16b_if(" in code) == packets
    expected = sliding_windows_reference(x.view(-1), shape[1], shape[2], window, pad)
    assert torch.equal(out, expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("flat_view", "packets"),
    [
        # The input 16 bytes into its buffer: packets from an aligned base.
        (slice(8, 1928), True),
        # 6 bytes in, and every other element: scalar loads.
        (slice(3, 1923), False),
        (slice(0, 3840, 2), False),
    ],
    ids=["aligned_offset", "unaligned_offset", "strided"],
)
def test_window_gathers_of_views_match_eager(flat_view: slice, packets: bool) -> None:
    elements, window, pad, seq, channels = 1920, 9, 4, 40, 16
    source = (torch.arange(2 * elements + 64, device=DEVICE) % 1021 - 510).to(
        torch.bfloat16
    )
    flat = source[flat_view]
    # A value no window holds: an element the kernel did not write keeps it.
    out = torch.full((elements, window), 12345.0, device=DEVICE, dtype=torch.bfloat16)
    config = register_tile_config(threads=32, window_block=16, staged=False)
    code, _ = _launch(sliding_windows_into, (flat, out, seq, channels, pad), config)
    assert ("_cute_load_v16b_if(" in code) == packets
    assert torch.equal(out, sliding_windows_reference(flat, seq, channels, window, pad))


@helion.kernel(backend="cute", static_shapes=True)
def _gather_from_starts(
    flat: torch.Tensor, starts: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(out.size(1)):
        index = starts[:, None] + tile_m.index[None, :]
        out[:, tile_m] = flat[index]
    return out


def test_only_a_tile_index_subscript_is_a_broadcast_view() -> None:
    """``starts[:, None]`` and ``tile_m.index[None, :]`` both trace to an
    unmasked ``hl.load`` with a broadcasting subscript; only the tile index
    is a view the matcher may look through, the other reads memory."""
    with cpu_render_target():
        bound = cpu_bind(
            _gather_from_starts,
            (
                torch.empty(4096),
                torch.zeros(8, dtype=torch.int32),
                torch.empty(8, 64),
            ),
        )
    views = {
        str(node.args[0]): _is_broadcast_view(node)
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.op == "call_function" and node.target is memory_ops.load
    }
    assert views == {"starts": False, "tile_index": True, "flat": False}


@helion.kernel(backend="cute", static_shapes=True)
def _column_sums_of_a_gather(x: torch.Tensor) -> torch.Tensor:
    """Column sums of ``x`` read as a gather of its flattened storage: the
    root tile's index, subscripted inside the row loop, is the gather's
    lane."""
    m, n = x.shape
    flat = x.view(-1)
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            rows = tile_m.index[:, None] * n
            gathered = hl.load(flat, [rows + tile_n.index[None, :]])
            acc = acc + gathered.to(torch.float32).sum(dim=0)
        out[tile_n] = acc
    return out


# One warp of 8-lane bf16 packets on the 256 columns, 16 rows per trip.
_COLUMN_SUMS = {
    "block_sizes": [256, 16],
    "num_threads": [32, 1],
    "cute_vector_widths": [8, 1],
    "cute_vloop_sink": True,
}


def test_a_gather_in_a_device_loop_along_a_root_tile_loads_packets() -> None:
    """A root tile's index subscripted inside a device loop traces to the
    same ``hl.load``, so the matcher reads these gathers too: the gather of
    a reduction nest along the root V-loop loads one packet per row, under
    its bound at the chunk base."""
    x = torch.empty(64, 256, dtype=torch.bfloat16)
    code = _code(_column_sums_of_a_gather, (x,), _COLUMN_SUMS)
    (packet,) = [
        line for line in code.splitlines() if "_cute_load_v16b_if(flat.iterator" in line
    ]
    assert "cutlass.Int32(lane_base_0) * cutlass.Int32(flat.layout.stride[0])" in packet
    assert "lane_base_0) < 16384" in packet
    assert ".load()" not in code


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_column_sums_of_a_gather_match_eager() -> None:
    # Small integers: the fp32 sums are exact in any order.
    x = (torch.arange(64 * 256, device=DEVICE) % 13 - 6).reshape(64, 256)
    x = x.to(torch.bfloat16)
    code, out = _launch(_column_sums_of_a_gather, (x,), _COLUMN_SUMS)
    assert "_cute_load_v16b_if(flat.iterator" in code
    assert torch.equal(out, x.to(torch.float32).sum(dim=0))


@helion.kernel(backend="cute", static_shapes=True)
def _column_sums_of_even_groups(x: torch.Tensor, group: int) -> torch.Tensor:
    """Column sums of ``x`` over the columns in an even group of ``group``:
    a tile load, no gather, under a mask whose lane term is a quotient."""
    group = hl.specialize(group)
    m, n = x.shape
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            in_group = ((tile_n.index // group) % 2 == 0)[None, :]
            keep = (tile_m.index >= 0)[:, None] & in_group
            kept = hl.load(x, [tile_m, tile_n], extra_mask=keep)
            acc = acc + kept.to(torch.float32).sum(dim=0)
        out[tile_n] = acc
    return out


def test_a_tile_load_under_a_quotient_mask_loads_packets() -> None:
    """The sink's second try takes any V-loop it declined, not the gathers'
    alone: with the lane quotient dividing the chunk base (groups of 16
    columns, a multiple of the 8 lanes) the mask is one value per vector and
    the tile load one predicated packet per row.  Groups of 12 keep the mask
    per lane, and the code that of the knob off."""
    x = torch.empty(64, 256, dtype=torch.bfloat16)
    code = _code(_column_sums_of_even_groups, (x, 16), _COLUMN_SUMS)
    assert code.count("_cute_load_v16b_if(") == 1
    assert ".load()" not in code
    quotients = {
        line.split(" = ")[1].split(" // ")[0]
        for line in code.splitlines()
        if " // " in line and not line.lstrip().startswith("#")
    }
    assert quotients == {"lane_base_0"}
    code = _code(_column_sums_of_even_groups, (x, 12), _COLUMN_SUMS)
    assert "_cute_load_v16b_if(" not in code
    off = {**_COLUMN_SUMS, "cute_vloop_sink": False}
    assert code == _code(_column_sums_of_even_groups, (x, 12), off)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_column_sums_of_even_groups_match_eager() -> None:
    # Small integers: the fp32 sums are exact in any order.
    x = (torch.arange(64 * 256, device=DEVICE) % 13 - 6).reshape(64, 256)
    x = x.to(torch.bfloat16)
    code, out = _launch(_column_sums_of_even_groups, (x, 16), _COLUMN_SUMS)
    assert "_cute_load_v16b_if(" in code
    kept = (torch.arange(256, device=DEVICE) // 16) % 2 == 0
    assert torch.equal(out, (x.to(torch.float32) * kept).sum(dim=0))


@helion.kernel(backend="cute", static_shapes=True)
def _every_third_row(flat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A gather along ``tile_n`` at a row the packets can follow, under a
    mask that keeps every third element: not one value per packet."""
    n, window = out.shape
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        keep = (i % 3 == 0)[:, None] & (k >= 0)[None, :]
        offset = i[:, None] + 16 * k[None, :]
        out[tile_n, tile_k] = hl.load(flat, [offset], extra_mask=keep)
    return out


def test_a_flat_gather_load_that_sinks_nothing_is_the_knob_off_code() -> None:
    """The load lowering marks its flat-gather address as code the knob
    shaped: the address takes that form here (an aligned base, a row that
    is a multiple of the packet), the sink declines the per-lane mask, and
    the knob leaves no trace."""
    args = (
        torch.empty(304, dtype=torch.bfloat16),
        torch.empty(256, 3, dtype=torch.bfloat16),
    )
    code = _code(_every_third_row, args, _TILE)
    assert "_cute_load_v16b_if(" not in code
    assert code == _code(_every_third_row, args, {**_TILE, "cute_vloop_sink": False})


@helion.kernel(backend="cute", static_shapes=True)
def _fold_windows(
    cols: torch.Tensor, flat: torch.Tensor, seq: int, channels: int, pad: int
) -> torch.Tensor:
    """The unfold's scatter: ``flat[n + channels * (k - pad)] = cols[n, k]``
    for source positions inside the sequence."""
    seq, channels, pad = (
        hl.specialize(seq),
        hl.specialize(channels),
        hl.specialize(pad),
    )
    n, window = cols.shape
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        hl.store(flat, [offset], cols[tile_n, tile_k], extra_mask=in_seq)
    return flat


def test_a_flat_gather_store_that_sinks_nothing_is_the_knob_off_code() -> None:
    """The store lowering marks its flat-gather address as code the knob
    shaped, as the load lowering does: nothing sinks here (the loads of
    ``cols`` run along the window), so the knob leaves no trace."""
    args = (
        torch.empty(256, 3, dtype=torch.bfloat16),
        torch.empty(256, dtype=torch.bfloat16),
        8,
        16,
        1,
    )
    code = _code(_fold_windows, args, _TILE)
    assert code == _code(_fold_windows, args, {**_TILE, "cute_vloop_sink": False})


@helion.kernel(backend="cute", static_shapes=True)
def _lane_terms_through_conversions(
    flat: torch.Tensor, out: torch.Tensor, scattered: torch.Tensor
) -> torch.Tensor:
    """Accesses along ``tile_n`` in program order: a gather whose lane term
    is the tile's index, one with that index widened to ``int64``, and three
    with it converted to ``uint8`` and back, which wraps it at 256: stores
    with the conversion before and after the subscript, and a gather."""
    for tile_n, tile_k in hl.tile([512, 3]):
        i = tile_n.index
        row = 256 * tile_k.index[None, :]
        plain = hl.load(flat, [i[:, None] + row])
        widened = hl.load(flat, [i.to(torch.int64)[:, None] + row])
        early = i.to(torch.uint8).to(torch.int32)[:, None] + row
        hl.store(scattered, [early], plain)
        late = i[:, None].to(torch.uint8).to(torch.int32) + row
        hl.store(scattered, [late], widened)
        out[tile_n, tile_k] = hl.load(flat, [late])
    return out


def test_a_lane_term_a_conversion_can_wrap_is_no_flat_gather() -> None:
    """The matcher's callers address the lane with the block's own index, so
    a lane term is a view of the tile index only through conversions that
    hold every index value."""
    with cpu_render_target():
        bound = cpu_bind(
            _lane_terms_through_conversions,
            (torch.empty(1024), torch.empty(512, 3), torch.empty(768)),
        )
    matched = [
        match_flat_gather_lane(node.args[1][0], bound.env) is not None
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.op == "call_function"
        and node.target in (memory_ops.load, memory_ops.store)
        and str(node.args[0]) in ("flat", "scattered")
    ]
    assert matched == [True, True, False, False, False]


@helion.kernel(backend="cute", static_shapes=True)
def _store_through_a_wrapping_cast(
    flat: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Gathers ``flat[i + 256 * k]`` and stores it at ``uint8(i) + 256 * k``
    for the rows from 256 on, whose index the conversion wraps: ``out[m] =
    flat[m + 256]``."""
    for tile_n, tile_k in hl.tile([512, 3]):
        i = tile_n.index[:, None]
        row = 256 * tile_k.index[None, :]
        value = hl.load(flat, [i + row])
        wrapped = i.to(torch.uint8).to(torch.int32) + row
        hl.store(out, [wrapped], value, extra_mask=i >= 256)
    return out


def test_a_store_through_a_wrapping_cast_keeps_its_own_address() -> None:
    config = register_tile_config(threads=8, window_block=4, vec=4, staged=False)
    args = (torch.empty(1024), torch.empty(768))
    code = _code(_store_through_a_wrapping_cast, args, config)
    # The gather beside it still loads packets along the lane tile.
    assert code.count("_cute_load_v16b_if(flat.iterator") == 4
    # The store's address is the lifted index, which holds the conversion,
    # not the lane tile's own index.
    assert "= cutlass.Uint8(indices_0" in code
    stores = [line for line in code.splitlines() if "(out.iterator + " in line]
    assert len(stores) == 4
    assert not any("indices_0" in line for line in stores)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_store_through_a_wrapping_cast_matches_eager() -> None:
    torch.manual_seed(0)
    flat = torch.randn(1024, device=DEVICE)
    # A value no input holds: an element the kernel did not write keeps it.
    out = torch.full((768,), 12345.0, device=DEVICE)
    config = register_tile_config(threads=8, window_block=4, vec=4, staged=False)
    code, _ = _launch(_store_through_a_wrapping_cast, (flat, out), config)
    assert "_cute_load_v16b_if(flat.iterator" in code
    assert torch.equal(out, flat[256:])


@helion.kernel(backend="cute", static_shapes=True)
def _windows_of_int_arguments(
    flat: torch.Tensor, out: torch.Tensor, seq: int, channels: int, pad: int
) -> torch.Tensor:
    """``sliding_windows_into`` with its sizes left as ``int`` arguments,
    which key the bound kernel by type alone."""
    n, window = out.shape
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        out[tile_n, tile_k] = hl.load(flat, [offset], extra_mask=in_seq)
    return out


def test_a_root_tile_gather_pins_no_int_argument_residue() -> None:
    """The flat-gather factor specialization keys a bound kernel on the
    packet residue of the ``int`` arguments in a gather's row.  A gather
    along a root tile's ``tile.index[:, None]`` registers none: its load
    stays scalar whatever the residue (the mask is proved on the generated
    code, where ``channels`` is an argument), so calls that differ in
    ``channels`` or ``pad`` keep sharing one bound kernel."""
    args = (
        torch.empty(256, dtype=torch.bfloat16),
        torch.empty(256, 3, dtype=torch.bfloat16),
        8,
        16,
        1,
    )
    with cpu_render_target():
        bound = cpu_bind(_windows_of_int_arguments, args)
    specializations = bound.env.runtime_input_specializations
    assert FLAT_GATHER_FACTOR_SPECIALIZATION_KEY not in specializations
    code = _code(_windows_of_int_arguments, args, _TILE)
    assert "_cute_load_v16b_if(" not in code
    assert code.count(".load()") == 1


@helion.kernel(backend="cute", static_shapes=True)
def _windows_plus_wide(
    x: torch.Tensor, wide: torch.Tensor, window: int, pad: int
) -> torch.Tensor:
    """``sliding_windows`` plus the same windows of an fp64 tensor, whose
    loads have no packet form."""
    window, pad = hl.specialize(window), hl.specialize(pad)
    _, seq, channels = x.shape
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n, window], dtype=x.dtype, device=x.device)
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        value = hl.load(flat, [offset], extra_mask=in_seq)
        other = hl.load(wide, [offset], extra_mask=in_seq)
        out[tile_n, tile_k] = value + other.to(x.dtype)
    return out


def test_only_the_predicates_of_the_packet_loads_compare_the_chunk_base() -> None:
    """The fp32 gather sinks under its mask compared at the chunk base; the
    fp64 gather beside it stays a scalar load per lane, and its mask, the
    same comparisons, keeps comparing each lane's own index."""
    config = register_tile_config(threads=8, window_block=4, vec=4, staged=False)
    args = (torch.empty(2, 8, 16), torch.empty(256, dtype=torch.float64), 3, 1)
    code = _code(_windows_plus_wide, args, config)
    _check_window_packets(code)
    scalars = [
        select
        for select in ast.walk(ast.parse(code))
        if isinstance(select, ast.IfExp)
        and isinstance(select.body, ast.Call)
        and isinstance(select.body.func, ast.Attribute)
        and select.body.func.attr == "load"
        and ast.unparse(select.body.func.value).startswith("wide.iterator + ")
    ]
    assert len(scalars) == 4
    for slot, scalar in enumerate(scalars):
        assert _guard_terms(scalar.test) == [
            f"v_11_u{slot}",
            f"indices_0_u{slot} - tile_offset_0 < 256",
            f"mask_1_u{slot}",
            f"cutlass.Int32(v_16_u{slot}) < 256",
        ]
        assert "lane_base_0" not in ast.unparse(scalar)


# A vector loop of 8 lanes at a base that is a multiple of 8.
_VECTOR = """
offsets_base_0 = cutlass.Int32(cute.arch.block_idx()[0]) * _BLOCK_SIZE_0
lane_base_0 = offsets_base_0 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
for vec_lane_0 in cutlass.range_constexpr(8):
    indices_0 = lane_base_0 + cutlass.Int32(vec_lane_0)
{body}
"""
_LOAD = "(x.iterator + indices_0).load()"


def _uniform(body: str) -> str | None:
    """``uniform_load_predicates``' copy of the vector loop of ``_VECTOR``
    (None when the proof changed nothing), a select counting as a load's
    predicate when its body is the load ``_LOAD``."""
    module = ast.parse(_VECTOR.format(body=textwrap.indent(body, "    ")))
    (loop,) = [stmt for stmt in module.body if isinstance(stmt, ast.For)]
    copies = uniform_load_predicates(
        module.body,
        constexpr_values={"_BLOCK_SIZE_0": 8192},
        load_predicate=lambda select: ast.unparse(select.body) == _LOAD,
    )
    copy = copies.get(id(loop))
    return None if copy is None else ast.unparse(copy)


def test_a_load_predicate_is_a_select_of_the_recorded_load_itself() -> None:
    """What the sink hands the early proof as a load's predicate: a select
    of a load it recorded, in either spelling, and a literal fill.  With
    arithmetic on the loaded value or in the fill the select is a float
    select like any other, and a load the sink did not record is not its to
    widen."""
    recorded: Any = {"x.iterator + indices_0": object()}

    def guards(select: str) -> bool:
        node = ast.parse(select, mode="eval").body
        assert isinstance(node, ast.IfExp)
        return _guards_recorded_load(node, loads=recorded, rename_groups={})

    assert guards(f"{_LOAD} if indices_0 < 8192 else 0.0")
    assert guards(
        "cute.arch.load(x.iterator + indices_0, cutlass.Float32)"
        " if indices_0 < 8192 else 0.0"
    )
    assert guards(f"{_LOAD} if indices_0 < 8192 else cutlass.Float32(float('-inf'))")
    assert not guards(f"a * b + {_LOAD} if indices_0 < 8192 else 0.0")
    assert not guards(f"{_LOAD} if indices_0 < 8192 else a * b")
    assert not guards("(y.iterator + indices_0).load() if indices_0 < 8192 else 0.0")


def test_the_early_proof_rewrites_load_predicates_only() -> None:
    """Beside the load's predicate, the same comparison in a mask statement
    and in a select of float arithmetic (the forms whose rewrite can change
    FMA contraction, ``cute_uniform_lane_compares``) is left per lane."""
    result = _uniform(
        f"loaded = {_LOAD} if indices_0 < 8192 else 0.0\n"
        "inside = indices_0 < 8192\n"
        "scaled = a * b if indices_0 < 8192 else 0.0\n"
    )
    assert result is not None
    assert f"loaded = {_LOAD} if lane_base_0 < 8192 else 0.0" in result
    assert "inside = indices_0 < 8192" in result
    assert "scaled = a * b if indices_0 < 8192 else 0.0" in result
    # Without a load's predicate there is nothing to rewrite.
    assert _uniform("scaled = a * b if indices_0 < 8192 else 0.0\n") is None


def test_the_early_proof_follows_a_lane_loop_that_keeps_the_offset() -> None:
    result = _uniform(
        f"for lane_1 in range(4):\n    loaded = {_LOAD} if indices_0 < 8192 else 0.0\n"
    )
    assert result is not None
    assert "if lane_base_0 < 8192 else" in result
    # A loop that moves the offset on leaves it no multiple of the width in
    # its later iterations: the predicate keeps the lane.
    assert (
        _uniform(
            "for lane_1 in range(4):\n"
            f"    loaded = {_LOAD} if indices_0 < 8192 else 0.0\n"
            "    indices_0 = indices_0 + 1\n"
        )
        is None
    )
