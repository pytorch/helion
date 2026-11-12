"""CPU tests: ``cute_tail_peel`` runs a device loop's complete tiles apart
from its partial last tile and loads their lanes in 8- or 16-byte packets;
the autotuner offers it where such a loop exists, with witnesses on one-warp
seeds."""

from __future__ import annotations

import ast
import re
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.peel_vector_tails import TensorFacts
from helion._compiler.cute.peel_vector_tails import _Packetizer
from helion._compiler.cute.peel_vector_tails import peel_vector_tails
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tail_peel_kernels import STATS_CONFIG
from helion._testing.tail_peel_kernels import online_softmax_stats
from helion._testing.tail_peel_kernels import padded_rows
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

_PACKET = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"


def _packet_names(code: str) -> list[str]:
    """The variables the rendered kernel ``code`` loads packets into."""
    return re.findall(
        r"^\s*(_?tail_packet\w*) = cute\.arch\.load\(", code, re.MULTILINE
    )


def _code(logits: torch.Tensor, **knobs: object) -> str:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(online_softmax_stats, (logits,))
        config = helion.Config(**{**STATS_CONFIG, **knobs})
        return bound.to_code(bound.config_spec.normalized_config(config))


def _vocab_loops(code: str) -> list[str]:
    return re.findall(r"for tile_offset_1 in range\(([^)]*\)[^)]*\)[^)]*\))\)", code)


def test_complete_tiles_load_packets_and_the_last_tile_keeps_its_masks() -> None:
    # 1003 columns: three complete 256-column tiles, then 235 columns.
    logits = padded_rows(4, 1003, "cpu")
    off = _code(logits)
    on = _code(logits, cute_tail_peel=True)
    assert len(_vocab_loops(off)) == 1
    assert _PACKET not in off
    loops = _vocab_loops(on)
    assert len(loops) == 2
    assert "cutlass.Int32(768)" in loops[0] and "cutlass.Int32(768)" in loops[1]
    complete, partial = on.split("for tile_offset_1 in range(", 2)[1:]
    # The complete tiles load each lane's eight columns as one packet and
    # compare no column against the row's end.
    assert _PACKET in complete
    assert "< 1003" not in complete
    assert ").load()" not in complete
    # The last tile keeps its masked scalar loads.
    assert "< 1003" in partial
    assert ").load() if mask_1" in partial


def test_a_row_the_tiles_divide_is_left_alone() -> None:
    logits = padded_rows(4, 1024, "cpu")
    assert _code(logits, cute_tail_peel=True) == _code(logits)


def test_rows_without_aligned_packets_are_left_alone() -> None:
    # Unpadded rows of 1003 bf16 start at 2-byte alignment: no packet, so
    # the loop is not split either.
    logits = torch.randn(4, 1003, dtype=torch.bfloat16)
    assert _code(logits, cute_tail_peel=True) == _code(logits)


def test_rows_whose_base_alignment_is_unproven_are_left_alone() -> None:
    # Padded rows of 1008 columns sliced from their second column start 2
    # bytes off a 16-byte boundary, which the strides alone do not show: no
    # packet, so the loop is not split.
    logits = torch.randn(4, 1008, dtype=torch.bfloat16)[:, 1:1004]
    assert _code(logits, cute_tail_peel=True) == _code(logits)


@pytest.mark.parametrize(
    ("dtype", "width", "packet"),
    [
        (torch.bfloat16, 4, "[4], cutlass.Uint16"),
        (torch.float32, 4, "[4], cutlass.Uint32"),
        (torch.float32, 2, "[2], cutlass.Uint32"),
    ],
    ids=["bf16_8_bytes", "fp32_16_bytes", "fp32_8_bytes"],
)
def test_lanes_load_eight_or_sixteen_byte_packets(
    dtype: torch.dtype, width: int, packet: str
) -> None:
    logits = torch.randn(4, 1008, dtype=dtype)[:, :1003]
    code = _code(
        logits,
        block_sizes=[1, 32 * width],
        cute_vector_widths=[width, width],
        cute_tail_peel=True,
    )
    assert _packet_names(code)
    assert f"ir.VectorType.get({packet}.mlir_type)" in code


@helion.kernel(backend="cute", static_shapes=True)
def _row_sum_from(x: torch.Tensor, start: hl.constexpr) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(x.size(0)):
        acc = hl.zeros([tile_r], dtype=torch.float32)
        for tile_v in hl.tile(start, x.size(1)):
            acc = acc + x[tile_r, tile_v].to(torch.float32).sum(-1)
        out[tile_r] = acc
    return out


def test_a_loop_start_aligns_the_lanes_to_its_gcd_with_the_step() -> None:
    # A tile offset of ``start + k * step`` is a multiple of gcd(start,
    # step): 8 from a start of 8 takes 8-lane packets, 4 from a start of 4
    # only 4-lane ones.
    rows = padded_rows(4, 1003, "cpu")

    def code(start: int, width: int, **knobs: object) -> str:
        with cpu_render_target(reduction_smem_budget_bytes=232448):
            bound = cpu_bind(_row_sum_from, (rows, start))
            config = helion.Config(
                block_sizes=[1, 32 * width],
                num_threads=[1, 32],
                cute_vector_widths=[width, width],
                **knobs,
            )
            return bound.to_code(bound.config_spec.normalized_config(config))

    assert _packet_names(code(8, 8, cute_tail_peel=True))
    assert code(4, 8, cute_tail_peel=True) == code(4, 8)
    assert _packet_names(code(4, 4, cute_tail_peel=True))


@helion.kernel(backend="cute", static_shapes=True)
def _sum_and_squares(tail_packet: torch.Tensor) -> torch.Tensor:
    """Two device loops over each row of an input named like the packets."""
    out = torch.empty(
        [tail_packet.size(0)], dtype=torch.float32, device=tail_packet.device
    )
    for tile_r in hl.tile(tail_packet.size(0)):
        total = hl.zeros([tile_r], dtype=torch.float32)
        for tile_v in hl.tile(tail_packet.size(1)):
            total = total + tail_packet[tile_r, tile_v].to(torch.float32).sum(-1)
        squares = hl.zeros([tile_r], dtype=torch.float32)
        for tile_v in hl.tile(tail_packet.size(1)):
            v = tail_packet[tile_r, tile_v].to(torch.float32)
            squares = squares + (v * v).sum(-1)
        out[tile_r] = total + squares
    return out


def test_packet_names_are_the_kernels_own() -> None:
    # Each split loop's packets take names no other variable of the kernel
    # has, its arguments included.
    rows = padded_rows(4, 1003, "cpu")
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_sum_and_squares, (rows,))
        config = helion.Config(
            block_sizes=[1, 256, 256],
            num_threads=[1, 32, 32],
            cute_vector_widths=[8, 8, 8],
            cute_tail_peel=True,
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
    names = _packet_names(code)
    assert len(names) == 2, code
    assert len(set(names)) == 2 and "tail_packet" not in names, code


@pytest.mark.parametrize("pid_type", ["persistent_blocked", "persistent_interleaved"])
def test_a_persistent_launch_splits_the_loops_of_each_program_id(pid_type: str) -> None:
    # The device loop runs inside the loop over virtual program ids.
    logits = padded_rows(4, 1003, "cpu")
    code = _code(logits, pid_type=pid_type, cute_tail_peel=True)
    assert "for virtual_pid in" in code
    assert len(_vocab_loops(code)) == 2
    assert _PACKET in code
    assert code != _code(logits, pid_type=pid_type)


_FACTS = {
    name: TensorFacts(
        dtype="cutlass.BFloat16",
        carrier="cutlass.Uint16",
        itemsize=2,
        strides={0: 1008, 1: 1},
        alignment=16,
    )
    for name in ("x", "y")
}
_LOAD = (
    "load = (x.iterator + cutlass.Int32(indices_0) * cutlass.Int32(x.layout.stride[0])"
    " + cutlass.Int32(indices_1) * cutlass.Int32(x.layout.stride[1])).load()"
    " if mask_1 else cutlass.BFloat16(0)"
)


def _packets(*body: str, before: tuple[str, ...] = ()) -> int:
    """The packets the complete tiles' lane loop gets, its lane offset bound
    first, then the statements ``before``, then in the lane loop the lane
    index, the true mask and ``body``."""
    source = (
        "lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n"
        + "".join(f"{statement}\n" for statement in before)
        + textwrap.dedent(
            """
            for vec_lane_1 in cutlass.range_constexpr(8):
                indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                mask_1 = True
            """
        )
        + "".join(f"    {statement}\n" for statement in body)
    )
    names = iter(range(100))
    packetizer = _Packetizer(
        _FACTS, {"tile_offset_1": 256}, {}, lambda name: f"{name}_{next(names)}"
    )
    packetizer.block(ast.parse(source).body)
    return packetizer.count


@pytest.mark.parametrize(
    ("before", "packets"),
    [
        ((), 1),
        (("w = _cute_inline_asm_elementwise('mov.b32 $0, $1;', v, is_pure=True)",), 1),
        (
            (
                "_cute_inline_asm_elementwise('st.global.b16 [$0], $1;', x, is_pure=False)",
            ),
            0,
        ),
        (("indices_0 = indices_0 + 1",), 0),
        (("indices_1 = indices_1 + cutlass.Int32(1)",), 0),
        (("mask_1 = indices_1 < 1003",), 0),
        (("cute.arch.sync_threads()",), 0),
        (("cute.arch.cluster_wait()",), 0),
        (("x[indices_0, indices_1] = cutlass.BFloat16(1)",), 0),
        (("y[indices_0, indices_1] = cutlass.BFloat16(1)",), 0),
        (("_helion_vmerge_cache_0[vec_lane_1] = cutlass.Float32(0.0)",), 1),
    ],
    ids=[
        "plain",
        "pure_asm",
        "impure_asm",
        "row_index_bound_in_the_loop",
        "lane_offset_bound_again",
        "mask_bound_again",
        "sync_threads",
        "cluster_wait",
        "tensor_subscript_store",
        "other_tensor_subscript_store",
        "fragment_subscript_store",
    ],
)
def test_a_packet_moves_above_its_lane_loop_only_where_nothing_changes_it(
    before: tuple[str, ...], packets: int
) -> None:
    # The packet loads before the lane loop: impure inline asm or a store
    # through a subscript of a tensor argument there may write what it reads
    # (any argument may alias it; a register fragment's element is no
    # argument's), a barrier or a cluster wait there orders other threads'
    # stores before the load, which writes nothing itself, and its pointer
    # may read only names bound before the loop, its lane offset and mask
    # unchanged since they were recorded.
    assert _packets(*before, _LOAD) == packets


@pytest.mark.parametrize(
    ("before", "packets"),
    [((), 1), (("rows[0] = cutlass.Int32(2)",), 0)],
    ids=["unwritten", "written_in_the_loop"],
)
def test_a_packet_pointer_reads_no_element_its_lane_loop_writes(
    before: tuple[str, ...], packets: int
) -> None:
    # The row index is an element of ``rows``: written in the lane loop, it
    # holds another value at the load than above the loop, where the packet
    # would load.
    load = _LOAD.replace("cutlass.Int32(indices_0)", "cutlass.Int32(rows[0])")
    assert _packets(*before, load) == packets


def test_a_shared_memory_reduce_keeps_the_loop_whole() -> None:
    # Two warps per row reduce through shared memory: a second copy of the
    # loop would allocate its own buffers.
    logits = padded_rows(4, 1003, "cpu")
    knobs = {"num_threads": [1, 64]}
    assert _code(logits, cute_tail_peel=True, **knobs) == _code(logits, **knobs)


@helion.kernel(backend="cute", static_shapes=True)
def _row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(x.size(0)):
        out[tile_r] = x[tile_r, :].to(torch.float32).sum(-1)
    return out


def _spec(kernel: Kernel[Any], logits: torch.Tensor) -> ConfigSpec:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        return cpu_bind(kernel, (logits,)).config_spec


def test_the_knob_is_searched_where_a_device_loop_loads_a_float_tile() -> None:
    # Structure alone opens it: the same search space at a width the tiles
    # divide and at one they do not.
    specs = [
        _spec(online_softmax_stats, padded_rows(4, vocab, "cpu"))
        for vocab in (1003, 1024)
    ]
    assert all("cute_tail_peel" in spec._flat_fields() for spec in specs)
    assert specs[0].structural_fingerprint() == specs[1].structural_fingerprint()
    # A row reduction without a device loop does not offer it and rejects an
    # explicit True; an explicit False, as a config tuned where the knob
    # applies carries it, is dropped.
    spec = _spec(_row_sum, torch.randn(4, 1003, dtype=torch.bfloat16))
    assert "cute_tail_peel" not in spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="cute_tail_peel requires"):
        spec.normalized_config(helion.Config(cute_tail_peel=True))
    off = spec.normalized_config(
        helion.Config(**spec.default_config().config, cute_tail_peel=False)
    )
    assert off == spec.default_config()


def test_one_warp_tiles_with_a_partial_last_tile_carry_the_witnesses() -> None:
    (group,) = [
        group
        for group in _spec(
            online_softmax_stats, padded_rows(4, 1003, "cpu")
        ).compiler_coverage_groups
        if group.key == "cute_tail_peel"
    ]
    assert group.witnesses
    for witness in group.witnesses:
        assert witness.value is True
        config = witness.carrier.config
        # 256-column tiles leave a partial last tile of 1003 columns, on one
        # warp, with eight lanes to load in packets.
        assert config["block_sizes"][1] == 256
        assert config["num_threads"][1] == 32
        assert config["cute_vector_widths"][1] == 8
    # Where the tiles divide the row there is no partial tile to split off.
    assert not any(
        group.key == "cute_tail_peel"
        for group in _spec(
            online_softmax_stats, padded_rows(4, 1024, "cpu")
        ).compiler_coverage_groups
    )


@pytest.mark.parametrize(
    "before",
    [
        "if c:\n    lane_base_1 = tile_offset_1 + 1",
        "for _ in range(2):\n    lane_base_1 = tile_offset_1 + 1",
        "lane_base_1 += 1",
        "lane_base_1, other = tile_offset_1 + 1, 0",
        "tile_offset_1 = tile_offset_1 + 1",
    ],
    ids=["in_a_branch", "in_a_loop", "augmented", "tuple", "its_input"],
)
def test_a_lane_base_bound_again_takes_no_packet(before: str) -> None:
    # The lane offset's base is aligned only as long as nothing binds it, or
    # the loop variable it is defined from, again.
    assert _packets(_LOAD) == 1
    assert _packets(_LOAD, before=tuple(before.splitlines())) == 0


def test_another_dims_index_that_reads_the_lane_takes_no_packet() -> None:
    # A row index that reads the lane index differs per lane, where a
    # packet loads every lane's element at one row.
    load = _LOAD.replace("cutlass.Int32(indices_0)", "cutlass.Int32(indices_1 + 0)")
    assert _packets(load) == 0


def test_a_lane_base_bound_again_later_in_its_loop_takes_no_packet() -> None:
    # Bound again after the lane loop, in a loop around it, the tile offset
    # is off a packet boundary from that loop's second iteration on.
    source = textwrap.dedent(
        f"""
        for j in range(2):
            lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
            for vec_lane_1 in cutlass.range_constexpr(8):
                indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                mask_1 = True
                {_LOAD}
        """
    )
    names = iter(range(100))
    counts = []
    for after in ("", "    tile_offset_1 = tile_offset_1 + 1\n"):
        packetizer = _Packetizer(
            _FACTS, {"tile_offset_1": 256}, {}, lambda name: f"{name}_{next(names)}"
        )
        packetizer.block(ast.parse(source + after).body)
        counts.append(packetizer.count)
    assert counts == [1, 0]


def test_a_definition_a_persistent_loop_binds_again_proves_no_mask() -> None:
    # ``shift`` is 0 for the first virtual program id only: the complete
    # tiles' mask cannot be proven true from it.
    def split(*, rebound: bool) -> str:
        source = textwrap.dedent(
            f"""
            shift = 0
            for virtual_pid in range(4):
                for tile_offset_1 in range(0, 1003, 256):
                    lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
                    for vec_lane_1 in cutlass.range_constexpr(8):
                        indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                        mask_1 = indices_1 + shift < 1003
                        {_LOAD}
            """
        ) + ("    shift = 300\n" if rebound else "")
        names = iter(range(100))
        body = peel_vector_tails(
            ast.parse(source).body,
            {},
            thread_block_dims=(32, 1, 1),
            tensor_facts=_FACTS,
            new_var=lambda name: f"{name}_{next(names)}",
            persistent_loop="virtual_pid",
        )
        return ast.unparse(body)

    assert _PACKET in split(rebound=False)
    assert _PACKET not in split(rebound=True)


@pytest.mark.parametrize("branch_writes_a_rebound_name", [False, True])
def test_a_statement_writing_a_rebound_name_leaves_its_other_names_unknown(
    branch_writes_a_rebound_name: bool,
) -> None:
    # The branch may set ``shift`` to 300 whether or not it also writes the
    # ``scratch`` the persistent loop binds again: either way ``shift = 0``
    # no longer holds at the loop, and the complete tiles' mask is unproven.
    scratch = "    scratch = 1\n" if branch_writes_a_rebound_name else ""
    source = (
        "shift = 0\n"
        "if c:\n"
        "    shift = 300\n"
        + scratch
        + textwrap.dedent(
            f"""
            for virtual_pid in range(4):
                scratch = 2
                for tile_offset_1 in range(0, 1003, 256):
                    lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
                    for vec_lane_1 in cutlass.range_constexpr(8):
                        indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                        mask_1 = indices_1 + shift < 1003
                        {_LOAD}
            """
        )
    )
    names = iter(range(100))
    body = peel_vector_tails(
        ast.parse(source).body,
        {},
        thread_block_dims=(32, 1, 1),
        tensor_facts=_FACTS,
        new_var=lambda name: f"{name}_{next(names)}",
        persistent_loop="virtual_pid",
    )
    assert _PACKET not in ast.unparse(body)
