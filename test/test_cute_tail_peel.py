"""CPU tests: ``cute_tail_peel`` runs a device loop's complete tiles apart
from its partial last tile and loads their lanes in 8- or 16-byte packets;
the autotuner offers it where such a loop exists, with witnesses on one-warp
seeds.  A nest ``cute_vloop_sink`` interchanged loads predicated packets
already: its complete tiles load them unpredicated and drop the selects of
the masks the bounds proof discharges from static facts."""

from __future__ import annotations

import ast
import re
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune.heuristics import (
    register_defer_tile_reduce_coverage,
)
from helion._compiler.cute.autotune.heuristics import register_tail_peel_coverage
from helion._compiler.cute.peel_vector_tails import TensorFacts
from helion._compiler.cute.peel_vector_tails import _Packetizer
from helion._compiler.cute.peel_vector_tails import peel_vector_tails
from helion._compiler.program_id import ForEachProgramID
from helion._compiler.program_id import PersistentPhase
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.split_reduction_kernels import SPLIT_CONFIG
from helion._testing.split_reduction_kernels import split_column_sums
from helion._testing.tail_peel_kernels import PHASE_CONFIG
from helion._testing.tail_peel_kernels import STATS_CONFIG
from helion._testing.tail_peel_kernels import online_softmax_stats
from helion._testing.tail_peel_kernels import padded_rows
from helion._testing.tail_peel_kernels import phased_tail_rows
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion.autotuner.compiler_coverage import CompilerCoverageGroup
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


@pytest.mark.parametrize("pid_type", ["persistent_interleaved", "persistent_blocked"])
def test_phase_dispatch_peels_the_complete_tiles_only(pid_type: str) -> None:
    with cpu_render_target():
        bound = cpu_bind(phased_tail_rows, (padded_rows(4, 1003, "cpu"),))
        configs = [
            helion.Config(**PHASE_CONFIG, pid_type=pid_type, cute_tail_peel=peel)
            for peel in (False, True)
        ]
        stock, peeled = [
            bound.to_code(bound.config_spec.normalized_config(cfg)) for cfg in configs
        ]
    assert "tail_packet =" not in stock
    assert "tail_packet =" in peeled
    bounds = _vocab_loops(peeled)
    assert any("cutlass.Int32(0), cutlass.Int32(768)" in b for b in bounds)
    assert any("cutlass.Int32(768), cutlass.Int32(1003)" in b for b in bounds)


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


def _packets(
    *body: str,
    before: tuple[str, ...] = (),
    load_targets: bool = False,
    renamed: frozenset[str] = frozenset(),
    width: str = "8",
    constexpr: dict[str, int] | None = None,
) -> int:
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
        _FACTS,
        {"tile_offset_1": 256},
        constexpr or {},
        lambda name: f"{name}_{next(names)}",
        renamed,
    )
    parsed = ast.parse(
        source.replace("range_constexpr(8)", f"range_constexpr({width})")
    )
    if load_targets:
        # A generated target need not carry its Python parser context.
        for statement in ast.walk(parsed):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, (ast.Name, ast.Subscript)):
                        target.ctx = ast.Load()
    packetizer.block(parsed.body)
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
    # A private register fragment is known from its producing allocation,
    # not from its name. No other reference to the new allocation escaped.
    pre = ("_helion_vmerge_cache_0 = cute.make_rmem_tensor(8, cutlass.Float32)",)
    assert _packets(*before, _LOAD, before=pre) == packets
    assert _packets(*before, _LOAD, before=pre, load_targets=True) == packets


def test_packet_store_requires_an_unaliased_register_destination() -> None:
    store = "scratch[vec_lane_1] = cutlass.BFloat16(1)"
    fresh = "scratch = cute.make_rmem_tensor(8, cutlass.Float32)"
    assert _packets(store, _LOAD, before=(fresh,)) == 1
    for setup in (
        (),
        ("scratch = x",),
        ("a = y", "scratch = a"),
        (fresh, "alias = scratch"),
    ):
        assert _packets(store, _LOAD, before=setup, load_targets=True) == 0
    assert _packets("scratch = y", store, _LOAD, before=(fresh,)) == 0


def test_packet_address_uses_the_list_after_an_append() -> None:
    load = _LOAD.replace("cutlass.Int32(indices_0)", "cutlass.Int32(rows[-1])")
    assert _packets(load, before=("rows = [0]",)) == 1
    for ref in ("rows", "alias"):
        assert (
            _packets(f"{ref}.append(1)", load, before=("rows = [0]", "alias = rows"))
            == 0
        )


def test_packet_load_stays_after_an_augmented_update_of_an_aliased_input() -> None:
    load = _LOAD.replace("cutlass.Int32(indices_0)", "cutlass.Int32(rows[-1])")
    before = ("rows = [0]", "alias = rows")
    assert _packets("alias += [1]", load, before=before) == 0
    assert (
        _packets("alias *= 0", load.replace("rows[-1]", "len(rows)"), before=before)
        == 0
    )
    assert (
        _packets(
            "alias |= {0: 1}",
            load.replace("rows[-1]", "rows[0]"),
            before=("rows = {0: 0}", "alias = rows"),
        )
        == 0
    )
    # Both operands and every reaching binding of the accumulator are scalar.
    assert (
        _packets(
            "acc += cutlass.Float32(1)",
            load,
            before=(*before, "acc = cutlass.Float32(0)"),
        )
        == 1
    )
    assert _packets("n ^= 1", load, before=(*before, "n = 0")) == 1
    # A later iteration can see a mutable binding even when the first one
    # enters with an integer. An unproved RHS may also overload __radd__.
    assert _packets("acc += 1", load, "acc = rows", before=(*before, "acc = 0")) == 0
    assert _packets("acc += unknown", load, before=(*before, "acc = 0")) == 0


def test_lane_and_address_keep_their_actual_bindings() -> None:
    rows = _LOAD.replace("indices_0", "len(rows)")
    assert (
        _packets(
            "match rows:\n        case acc: acc *= 0",
            rows,
            before=("rows = [0]", "acc = 0"),
        )
        == 0
    )
    assert (
        _packets(
            "acc *= 0",
            rows,
            before=("rows = [0]", "acc = 0", "match rows:\n    case acc: pass"),
        )
        == 0
    )
    assert _packets("vec_lane_1 = 7 - vec_lane_1", _LOAD) == 0
    assert (
        _packets(
            "acc += cutlass.Int32([0 for alias[0] in (1,)][0])",
            rows.replace("len(rows)", "rows[0]"),
            before=("rows=[0]", "alias=rows", "acc=0"),
        )
        == 0
    )
    assert _packets(_LOAD, width="V", constexpr={"V": 8}) == 1
    assert (
        _packets(
            _LOAD,
            before=("V_next=16",),
            width="V",
            constexpr={"V": 8},
            renamed=frozenset({"V", "V_next"}),
        )
        == 0
    )
    spellings = frozenset({"row", "row_next"})
    addr = _LOAD.replace("indices_0", "row_next")
    assert _packets("row ^= 1", addr, before=("row = 0",), renamed=spellings) == 0
    assert _packets("row = 1", addr, renamed=spellings) == 0
    assert _packets("other += -1", _LOAD, before=("other = 1",), renamed=spellings) == 1


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
    def split(*, rebound: bool, shadow: bool = False, phi: bool = False) -> str:
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
        ) + (f"    {'shift_next' if phi else 'shift'} = 300\n" if rebound else "")
        names = iter(range(100))
        body = peel_vector_tails(
            ast.parse(("x = y\n" if shadow else "") + source).body,
            {},
            thread_block_dims=(32, 1, 1),
            tensor_facts=_FACTS,
            new_var=lambda name: f"{name}_{next(names)}",
            persistent_loop="virtual_pid",
            rename_groups={"shift_next": "shift"} if phi else {},
        )
        return ast.unparse(body)

    assert _PACKET in split(rebound=False)
    assert _PACKET not in split(rebound=True)
    assert _PACKET not in split(rebound=True, phi=True)
    assert _PACKET not in split(rebound=False, shadow=True)


def test_shared_phase_dispatch_does_not_share_a_mask_proof() -> None:
    # The phase emitter copies the list of roots, not the dispatch tree
    # nested inside it. These two phases intentionally share the same If.
    dispatch = ast.parse("if shared_pid < 3: pass").body[0]
    assert isinstance(dispatch, ast.If)
    work = ast.parse(
        textwrap.dedent(f"""
        for tile_offset_1 in range(0, 1003, 256):
            lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
            for vec_lane_1 in cutlass.range_constexpr(8):
                indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                mask_1 = indices_1 + shift < 1003
                {_LOAD}
        """)
    ).body
    owner = ForEachProgramID("shared_pid")
    owner.case_bodies = [list(work), [ast.Pass()]]
    owner.case_branches = [dispatch]
    owner.codegen_dispatch()

    def phase(start: int) -> ast.For:
        loop = ast.parse(
            f"for pid in range(4):\n    if {start} <= pid < {start + 1}: pass"
        ).body[0]
        assert isinstance(loop, ast.For)
        guard = loop.body[0]
        assert isinstance(guard, ast.If)
        init = ast.parse("shared_pid = pid").body[0]
        guard.body = [init, dispatch]
        PersistentPhase.mark(loop, str(start), str(start + 1), [], init)
        return loop

    names = iter(range(100))
    first, second = phase(0), phase(1)
    proof = PersistentPhase.match(first)
    assert proof is not None and id(dispatch) in proof.phase.uniform_guards(proof.guard)
    proof.guard.body.insert(1, ast.parse("shared_pid = lane").body[0])
    assert id(dispatch) not in proof.phase.uniform_guards(proof.guard)
    proof.guard.body.pop(1)
    body = peel_vector_tails(
        [*ast.parse("shift=0").body, first, *ast.parse("shift=300").body, second],
        {},
        thread_block_dims=(32, 1, 1),
        tensor_facts=_FACTS,
        new_var=lambda name: f"{name}_{next(names)}",
        persistent_loop=None,
    )
    phases = [stmt for stmt in body if isinstance(stmt, ast.For)]
    assert len(phases) == 2
    assert _PACKET in ast.unparse(phases[0])
    assert _PACKET not in ast.unparse(phases[1])


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


_MASKED_PACKET = "_cute_load_v16b_if("
# Sixteen warps (128 row threads beside 4 column threads): the row threads
# combine through shared memory.
_SIXTEEN_WARPS = {
    "block_sizes": [1, 32, 512],
    "num_threads": [1, 4, 128],
    "cute_lane_unroll": 4,
}


def _split_code(rows: int, rows_per_split: int, **knobs: object) -> str:
    x = torch.empty(rows, 64, dtype=torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(split_column_sums, (x, x, rows_per_split))
        config = helion.Config.from_dict({**SPLIT_CONFIG, **knobs})
        return bound.to_code(bound.config_spec.normalized_config(config))


def _row_loops(code: str) -> list[tuple[tuple[int, int], str]]:
    """``((start, end), body text)`` of the generated kernel's row loops."""
    loops: list[tuple[tuple[int, int], str]] = []
    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "tile_offset_2"
        ):
            assert isinstance(node.iter, ast.Call)
            start, end = (
                int(ast.unparse(bound).removeprefix("cutlass.Int32(").removesuffix(")"))
                for bound in node.iter.args[:2]
            )
            body = "\n".join(ast.unparse(statement) for statement in node.body)
            loops.append(((start, end), body))
    return loops


def _selects(body: str) -> int:
    """The selects under the tile mask or the kernel's own row mask."""
    return len(re.findall(r" if (?:mask_2|v_\d+)_u\d+ else ", body))


def _packet_loop_ast(body: str) -> str:
    """Ignore only the association of conjunctions guarding a packet load."""

    def terms(value: ast.expr) -> list[ast.expr]:
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
            return [term for child in value.values for term in terms(child)]
        return [value]

    tree = ast.parse(body)
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_cute_load_v16b_if"
            and isinstance(guard := call.args[1], ast.BoolOp)
            and isinstance(guard.op, ast.And)
        ):
            guard.values = terms(guard)
    return ast.dump(tree)


def test_a_sunk_nest_runs_its_complete_tiles_unmasked() -> None:
    # Eight splits of 600 rows in 128-row tiles: four complete tiles, then
    # 88 rows.  Each thread loads two tensors' rows, eight per tile.
    off = _split_code(4800, 600)
    ((bounds, whole),) = _row_loops(off)
    assert bounds == (0, 600)
    assert whole.count(_MASKED_PACKET) == 16 and _selects(whole) == 16
    on = _split_code(4800, 600, cute_tail_peel=True)
    (complete_bounds, complete), (partial_bounds, partial) = _row_loops(on)
    assert (complete_bounds, partial_bounds) == ((0, 512), (512, 600))
    # The complete tiles load every packet unpredicated and select nothing;
    # the last tile is the loop as it was.
    assert _MASKED_PACKET not in complete
    assert complete.count("cute.arch.load(") == 16
    assert _selects(complete) == 0
    assert _packet_loop_ast(partial) == _packet_loop_ast(whole)
    assert "_cute_proven_loop_bounds_0_abi_version = 1" in on


def test_a_split_index_bounded_by_the_grid_alone_takes_the_static_grid() -> None:
    # With the columns the fast axis, a block's split is ``block_idx // 4``,
    # which only the launch's 32 blocks bound: ``row < rows`` holds for the
    # static grid of the launch, not for a block index the proof knows
    # nothing of.
    order = {"loop_orders": [[1, 0]]}
    on = _split_code(4800, 600, cute_tail_peel=True, **order)
    assert "pid_1 = cutlass.Int32(cute.arch.block_idx()[0]) // num_blocks_0" in on
    (_bounds, complete), _partial = _row_loops(on)
    assert _MASKED_PACKET not in complete and _selects(complete) == 0


def test_an_xyz_launch_bounds_the_split_index_by_its_grid() -> None:
    # Under XYZ program ids a block's split is the block index of an axis of
    # its own, which again only the launch's grid bounds.
    on = _split_code(4800, 600, cute_tail_peel=True, pid_type="xyz")
    assert "pid_1 = cutlass.Int32(cute.arch.block_idx()[1])" in on
    (_bounds, complete), _partial = _row_loops(on)
    assert _MASKED_PACKET not in complete and _selects(complete) == 0


def test_tiles_that_divide_the_rows_are_unmasked_in_place() -> None:
    # Splits of 640 rows: five complete tiles, nothing to split off.
    ((bounds, body),) = _row_loops(_split_code(5120, 640, cute_tail_peel=True))
    assert bounds == (0, 640)
    assert _MASKED_PACKET not in body and _selects(body) == 0
    assert body.count("cute.arch.load(") == 16


@pytest.mark.parametrize(
    ("rows", "masked"),
    [(4300, 16), (4700, 2)],
    ids=["no_row_proven", "last_lane_unproven"],
)
def test_the_kernels_own_row_mask_is_discharged_only_from_static_facts(
    rows: int, masked: int
) -> None:
    # The last split is partial, so ``row < rows`` does not hold for every
    # row of a complete tile.  4300 rows leave the last split 100: no lane
    # is proven and the loop stays whole.  4700 leave it 500: the proof holds
    # for the seven lanes that stay below row 496 of a split and fails for
    # the eighth, which keeps its predicate and its select.
    on = _split_code(rows, 600, cute_tail_peel=True)
    if masked == 16:
        assert on == _split_code(rows, 600)
        return
    (_bounds, complete), (_partial_bounds, partial) = _row_loops(on)
    assert complete.count(_MASKED_PACKET) == masked
    assert complete.count("cute.arch.load(") == 16 - masked
    assert _selects(complete) == masked // 2
    assert partial.count(_MASKED_PACKET) == 16


def test_the_unmasked_packets_keep_their_cache_hints() -> None:
    hints = {"load_eviction_policies": ["first", "l2_last"]}
    on = _split_code(4800, 600, cute_tail_peel=True, **hints)
    (_bounds, complete), (_partial_bounds, partial) = _row_loops(on)
    assert partial.count(", 'l2_last')") == 8
    assert complete.count("_cute_load_l2_evict_last(") == 8
    streaming = {"load_eviction_policies": ["first", "streaming"]}
    on = _split_code(4800, 600, cute_tail_peel=True, **streaming)
    (_bounds, complete), (_partial_bounds, partial) = _row_loops(on)
    assert partial.count(", 'cs')") == 8
    assert complete.count("cop='cs')") == 8


def test_a_shared_memory_reduce_in_the_sunk_loop_keeps_it_whole() -> None:
    # The 128 row threads reduce each tile through shared memory, which a
    # second copy of the loop would allocate again; with the combine
    # deferred past the loop (``cute_defer_tile_reduce``) nothing is left in
    # it to keep it whole.
    assert _split_code(4800, 600, cute_tail_peel=True, **_SIXTEEN_WARPS) == _split_code(
        4800, 600, **_SIXTEEN_WARPS
    )
    deferred = {**_SIXTEEN_WARPS, "cute_defer_tile_reduce": True}
    (bounds, complete), _partial = _row_loops(
        _split_code(4800, 600, cute_tail_peel=True, **deferred)
    )
    assert bounds == (0, 512)
    assert _MASKED_PACKET not in complete


_SUNK_NEST = """
shift = 0
for lane_1 in range(2):
    base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8
    for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(600), cutlass.Int32(128)):
        row = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[1]) * 8 + shift
        valid = operator.lt(row, 600)
        keep = valid & valid
        vec = _cute_load_v16b_if(x.iterator + cutlass.Int32(row) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(base) * cutlass.Int32(x.layout.stride[1]), keep and row < 600, ir.VectorType.get([8], cutlass.Uint16.mlir_type))
        for vec_lane_1 in cutlass.range_constexpr(8):
            acc[vec_lane_1] = acc[vec_lane_1] + (cutlass.Float32(vec[vec_lane_1]) if keep else cutlass.Float32(0))
    tail = 0
"""


def _peeled(
    source: str,
    *,
    grid: tuple[int, int, int] | None = None,
    rename_groups: dict[str, str] | None = None,
) -> str:
    names = iter(range(100))
    return ast.unparse(
        ast.Module(
            body=peel_vector_tails(
                ast.parse(textwrap.dedent(source)).body,
                {},
                thread_block_dims=(2, 16, 1),
                tensor_facts=_FACTS,
                new_var=lambda name: f"{name}_{next(names)}",
                persistent_loop=None,
                block_grid_dims=grid,
                rename_groups=rename_groups,
            ),
            type_ignores=[],
        )
    )


def test_sunk_packet_does_not_use_a_pre_rename_mask_value() -> None:
    source = _SUNK_NEST.replace("shift = 0", "shift = 0\nshift_next = 600")
    code = _peeled(source, rename_groups={"shift_next": "shift"})
    assert "cute.arch.load(x.iterator" not in code
    assert _MASKED_PACKET in code


def test_unrelated_scalar_warp_callback_preserves_sunk_packet_prefix() -> None:
    late = "\nvalue = cute.arch.warp_reduction(arg, lambda a, b: a * b)\n"
    assert "vec = cute.arch.load(x.iterator" in _peeled(
        _SUNK_NEST + late + "assert True\n"
    )
    assert _packets("assert ready", _LOAD) == 0
    assert "vec = cute.arch.load(x.iterator" not in _peeled(
        _SUNK_NEST + late.replace("a * b", "rows.pop()")
    )


def test_runtime_phases_keep_separate_sunk_packet_predicates() -> None:
    # A runtime phase has no trip certificate. The nested packet unmasker
    # still visits its dispatch, which the scheduler shares between phases.
    work = ast.parse(textwrap.dedent(_SUNK_NEST)).body[1:]
    dispatch = ast.parse("if pid < 3: pass").body[0]
    assert isinstance(dispatch, ast.If)
    dispatch.body = work

    def phase() -> ast.For:
        loop = ast.parse("for pid in range(trips()):\n    if pid < end: pass").body[0]
        assert isinstance(loop, ast.For)
        guard = loop.body[0]
        assert isinstance(guard, ast.If)
        guard.body = [dispatch]
        PersistentPhase.mark(loop, "0", "end", [])
        return loop

    names = iter(range(100))
    body = peel_vector_tails(
        [*ast.parse("shift=0").body, phase(), *ast.parse("shift=300").body, phase()],
        {},
        thread_block_dims=(2, 16, 1),
        tensor_facts=_FACTS,
        new_var=lambda name: f"{name}_{next(names)}",
        persistent_loop=None,
    )
    first, second = [ast.unparse(n) for n in body if isinstance(n, ast.For)]
    assert "cute.arch.load(x.iterator" in first
    assert "cute.arch.load(x.iterator" not in second
    assert _MASKED_PACKET in second


def test_conditional_generated_target_invalidates_a_packet_bound() -> None:
    source = textwrap.dedent(_SUNK_NEST).replace(
        "shift = 0", "shift = 0\nif flag:\n    shift = 300"
    )
    body = ast.parse(source).body
    branch = body[1]
    assert isinstance(branch, ast.If)
    assignment = branch.body[0]
    assert isinstance(assignment, ast.Assign) and isinstance(
        assignment.targets[0], ast.Name
    )
    assignment.targets[0].ctx = ast.Load()
    names = iter(range(100))
    result = peel_vector_tails(
        body,
        {},
        thread_block_dims=(2, 16, 1),
        tensor_facts=_FACTS,
        new_var=lambda name: f"{name}_{next(names)}",
        persistent_loop=None,
    )
    source = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "cute.arch.load(x.iterator" not in source
    assert _MASKED_PACKET in source


def test_the_pass_unmasks_a_nested_loops_complete_tiles() -> None:
    peeled = _peeled(_SUNK_NEST)
    complete, partial = peeled.split("for tile_offset_2 in range(")[1:]
    assert complete.startswith("cutlass.Int32(0), cutlass.Int32(512)")
    assert partial.startswith("cutlass.Int32(512), cutlass.Int32(600)")
    assert "vec = cute.arch.load(x.iterator" in complete
    assert "] + cutlass.Float32(vec[vec_lane_1])\n" in complete
    assert " if keep else " not in complete
    assert "_cute_load_v16b_if(" in partial and " if keep else " in partial


_BASE = "    base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n"
# A branch the proof resolves for every thread of the launch.
_RESOLVED_BRANCH = (
    "    if cutlass.Int32(cute.arch.thread_idx()[0]) < 2:\n"
    "        first = 1\n"
    "        second = 2\n"
)
# A lane loop with a packet of its own, under a predicate the proof discharges.
_PROVEN_LANE_LOOP = (
    "    for j in cutlass.range_constexpr(2):\n"
    "        other = _cute_load_v16b_if(y.iterator + cutlass.Int32(j) * "
    "cutlass.Int32(y.layout.stride[1]), j < 2, "
    "ir.VectorType.get([8], cutlass.Uint16.mlir_type))\n"
)


@pytest.mark.parametrize(
    "ahead",
    [_RESOLVED_BRANCH + _BASE, _RESOLVED_BRANCH + _BASE + _PROVEN_LANE_LOOP],
    ids=["an_assignment_in_the_loops_place", "another_loop_in_the_loops_place"],
)
def test_a_branch_the_proof_resolves_ahead_of_the_loop_moves_nothing(
    ahead: str,
) -> None:
    # The proof replaces a branch it resolves by the branch's body, in its
    # clones of the enclosing loop's statements: two statements for one, so
    # every statement after it stands one place further.  The complete tiles
    # are still the row loop's, and the statements around it stay as they
    # were: the nest is split exactly as without them.
    assert textwrap.dedent(_SUNK_NEST).count(_BASE) == 1
    without = _peeled(_SUNK_NEST)
    assert without.count(_BASE) == 1 and "cute.arch.load(x.iterator" in without
    assert _peeled(_SUNK_NEST.replace(_BASE, ahead)) == without.replace(_BASE, ahead)


def test_a_loop_split_for_its_lane_loads_keeps_their_packets_unmasked() -> None:
    # A top-level loop that loads ``x`` lane by lane under a mask against
    # its end and ``y`` in one predicated packet.  The split gives the
    # complete tiles ``x``'s packet, and they then load ``y``'s unpredicated
    # in place; the partial last tile keeps both loads as they were.
    wide = (
        "wide = _cute_load_v16b_if(y.iterator + cutlass.Int32(lane_base_1) * "
        "cutlass.Int32(y.layout.stride[1]), lane_base_1 + 7 < 1003, "
        "ir.VectorType.get([8], cutlass.Uint16.mlir_type))"
    )
    source = f"""
    for tile_offset_1 in range(0, 1003, 256):
        lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
        {wide}
        for vec_lane_1 in cutlass.range_constexpr(8):
            indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
            mask_1 = indices_1 < 1003
            {_LOAD}
    """
    complete, partial = _peeled(source).split("for tile_offset_1 in range(")[1:]
    assert complete.startswith("0, 768, 256)")
    assert partial.startswith("768, 1003, 256)")
    assert "tail_packet_0 = cute.arch.load(x.iterator" in complete
    assert "load = cutlass.Uint16(tail_packet_0[vec_lane_1])" in complete
    assert "wide = cute.arch.load(y.iterator" in complete
    assert "_cute_load_v16b_if(" not in complete and ".load() if " not in complete
    assert wide in partial and _LOAD in partial


# One mutation of the nest per fact the proof may not use: each leaves the
# row unproven where the loop reads it, so the loop stays as it is.
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("    tail = 0", "    shift = 100"),
        ("shift = 0\n", "shift = unknown()\n"),
        ("cute.arch.thread_idx()[1]) * 8 + shift", "cute.arch.thread_idx()[1]) * 15"),
        ("operator.lt(row, 600)", "operator.lt(row, rows)"),
        ("keep = valid & valid", "keep = valid & other"),
        (
            "        for vec_lane_1",
            "        barrier = cute.arch.mbarrier_init(1)\n        for vec_lane_1",
        ),
    ],
    ids=[
        "bound_again_later_in_the_enclosing_loop",
        "unknown_offset",
        "rows_past_the_tile",
        "a_runtime_row_count",
        "an_unproven_mask",
        "a_resource_the_copy_would_take_again",
    ],
)
def test_the_pass_keeps_a_nested_loop_it_cannot_prove(old: str, new: str) -> None:
    assert old in _SUNK_NEST
    source = _SUNK_NEST.replace(old, new)
    assert _peeled(source) == ast.unparse(ast.parse(textwrap.dedent(source)))


_PACKET_CALL = (
    "_cute_load_v16b_if(x.iterator + cutlass.Int32(row) * "
    "cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(base) * "
    "cutlass.Int32(x.layout.stride[1]), keep and row < 600, "
    "ir.VectorType.get([8], cutlass.Uint16.mlir_type))"
)


@pytest.mark.parametrize(
    "packet",
    [
        # Sixteen bf16: a 32-byte packet.
        _PACKET_CALL.replace("v16b_if", "v32b_if").replace("[8]", "[16]"),
        # Bytes packed in one word.
        _PACKET_CALL.replace("cutlass.Uint16", "cutlass.Uint8"),
        # No unpredicated 8-byte load reads noncoherently.
        _PACKET_CALL.replace("v16b_if", "v8b_if")
        .replace("[8]", "[4]")
        .replace("mlir_type))", "mlir_type), 'nc_l2_first')"),
    ],
    ids=["wide_packet", "byte_carrier", "hint_without_a_plain_load"],
)
def test_a_packet_no_unpredicated_load_carries_stays_predicated(packet: str) -> None:
    assert _PACKET_CALL in textwrap.dedent(_SUNK_NEST)
    source = _SUNK_NEST.replace(_PACKET_CALL, packet)
    assert _peeled(source) == ast.unparse(ast.parse(textwrap.dedent(source)))


def test_a_loop_without_a_complete_tile_stays_as_it_is() -> None:
    # 100 rows in one partial 128-row tile, the packet under a mask that
    # holds on every thread: there are no complete tiles to run apart.
    source = (
        _SUNK_NEST.replace(
            "cutlass.Int32(600), cutlass.Int32(128)",
            "cutlass.Int32(100), cutlass.Int32(128)",
        )
        .replace("operator.lt(row, 600)", "operator.lt(base, 16)")
        .replace("keep and row < 600", "keep")
    )
    assert "cutlass.Int32(100)" in source and "operator.lt(base, 16)" in source
    assert _peeled(source) == ast.unparse(ast.parse(textwrap.dedent(source)))
    # The same mask on a loop of several tiles is proven.
    several = source.replace("cutlass.Int32(100)", "cutlass.Int32(600)")
    assert "vec = cute.arch.load(x.iterator" in _peeled(several)


@pytest.mark.parametrize(("rows", "row_loops"), [(600, 2), (640, 1)])
def test_a_loop_around_an_unmasked_loop_stays_as_it_is(
    rows: int, row_loops: int
) -> None:
    # The second packet's mask holds for the complete tiles of the OUTER
    # loop only.  The row loop takes the first packet's unmasking, split for
    # it (600 rows) or in place (640 rows, which its tiles divide), and the
    # outer loop is not split for the second on top: nesting never multiplies
    # the copies.
    source = f"""
    for tile_offset_0 in range(cutlass.Int32(0), cutlass.Int32(20), cutlass.Int32(8)):
        column = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[0]) * 4
        for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32({rows}), cutlass.Int32(128)):
            row = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[1]) * 8
            vec = _cute_load_v16b_if(x.iterator + cutlass.Int32(row) * cutlass.Int32(x.layout.stride[0]), row < {rows}, ir.VectorType.get([8], cutlass.Uint16.mlir_type))
            other = _cute_load_v16b_if(x.iterator + cutlass.Int32(column) * cutlass.Int32(x.layout.stride[1]), column + 3 < 20, ir.VectorType.get([8], cutlass.Uint16.mlir_type))
    """
    peeled = _peeled(source)
    assert peeled.count("for tile_offset_0 in range(") == 1
    assert peeled.count("for tile_offset_2 in range(") == row_loops
    assert peeled.count("vec = cute.arch.load(") == 1
    assert peeled.count("vec = _cute_load_v16b_if(") == row_loops - 1
    assert peeled.count("other = _cute_load_v16b_if(") == row_loops


# A device loop over the rows that loads one packet under its tile mask.
_ROW_LOOP = (
    "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32({rows}), "
    "cutlass.Int32(128)):\n"
    "    row = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[1]) * 8\n"
    "    vec = _cute_load_v16b_if(x.iterator + cutlass.Int32(row) * "
    "cutlass.Int32(x.layout.stride[0]), row < {rows}, "
    "ir.VectorType.get([8], cutlass.Uint16.mlir_type))\n"
)
# A device loop over the columns, with a partial last tile, around ``NEST``.
_COLUMN_LOOP = (
    "for tile_offset_0 in range(cutlass.Int32(0), cutlass.Int32(20), "
    "cutlass.Int32(8)):\n"
    "    column = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[0]) * 4\n"
    "    NEST\n"
)


def _held(holder: str, nest: str) -> str:
    """``holder`` with the statements of ``nest`` in place of its ``NEST``."""
    line = next(line for line in holder.splitlines() if line.strip() == "NEST")
    indent = line.removesuffix("NEST")
    return holder.replace(f"{line}\n", textwrap.indent(f"{nest.rstrip()}\n", indent))


@pytest.mark.parametrize(
    "holder",
    [
        "if flag:\n    NEST\n",
        "if flag:\n    other = 1\nelse:\n    NEST\n",
        "with scope():\n    NEST\n",
        "for attempt in range(2):\n    other = attempt\nelse:\n    NEST\n",
        # A test the proof resolves for every thread of the launch: its
        # clone of the branch is replaced by the branch's body.
        "if cutlass.Int32(cute.arch.thread_idx()[0]) < 2:\n    NEST\n",
        "if flag:\n    with scope():\n        NEST\n",
    ],
    ids=[
        "a_branch",
        "an_else_branch",
        "a_with_block",
        "a_loops_else",
        "a_branch_the_proof_resolves",
        "a_with_block_under_a_branch",
    ],
)
def test_a_loop_under_a_branch_or_in_a_with_block_is_settled_where_it_stands(
    holder: str,
) -> None:
    # The pass makes of a row loop under a branch, in a ``with`` block or in
    # a loop's ``else`` what it makes of the row loop alone, at the top level
    # and in an enclosing device loop: split where the last row tile is
    # partial (600 rows), unmasked in place where the tiles divide the rows
    # (640).  The statements around the row loop stay as they were; the
    # column loop in particular is never split for the row loop's packet,
    # which would copy its whole body and leave the packet predicated in the
    # copy of its partial last tile.
    for rows, row_loops in ((600, 2), (640, 1)):
        row_loop = _ROW_LOOP.format(rows=rows)
        marker, _, alone = _peeled(row_loop).partition("\n")
        assert marker.startswith("_cute_proven_loop_bounds_")
        assert alone.count("for tile_offset_2 in range(") == row_loops
        assert alone.count("vec = cute.arch.load(") == 1
        assert alone.count("vec = _cute_load_v16b_if(") == row_loops - 1
        for around in ("NEST\n", _COLUMN_LOOP):

            def held(nest: str, around: str = around) -> str:
                return ast.unparse(ast.parse(_held(around, _held(holder, nest))))

            assert _peeled(held(row_loop)) == f"{marker}\n{held(alone)}"


@pytest.mark.parametrize(
    "holder",
    ["while flag:\n    NEST\n", "try:\n    NEST\nfinally:\n    other = 1\n"],
    ids=["a_while_loop", "a_try_block"],
)
def test_a_loop_in_a_statement_the_proof_does_not_walk_stays_as_it_is(
    holder: str,
) -> None:
    # The bounds proof walks loops, branches and ``with`` blocks only: a row
    # loop in any other compound statement keeps its predicated packet, and
    # so does the nest around that statement.
    for rows in (600, 640):
        for around in ("NEST\n", _COLUMN_LOOP):
            source = _held(around, _held(holder, _ROW_LOOP.format(rows=rows)))
            assert _peeled(source) == ast.unparse(ast.parse(source))


@pytest.mark.parametrize("under_a_branch", [False, True])
@pytest.mark.parametrize(("rows", "unmasked"), [(600, 0), (640, 2)])
def test_a_loop_nested_in_a_loop_split_for_its_lane_loads_is_not_split_again(
    rows: int, unmasked: int, under_a_branch: bool
) -> None:
    # The outer loop is split for its masked lane loads, so its body runs in
    # two copies.  A row loop in it with a partial last tile (600 rows) keeps
    # its predicated packet in both: a second split would run the row loop's
    # body four times over.  Row tiles that divide the rows (640) are
    # unmasked in place, in each copy.  The same holds for a row loop under
    # a branch of the outer loop's body.
    row_loop = _ROW_LOOP.format(rows=rows)
    if under_a_branch:
        row_loop = "if flag:\n" + textwrap.indent(row_loop, "    ")
    source = f"""
    for tile_offset_1 in range(0, 1003, 256):
        lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 8
        for vec_lane_1 in cutlass.range_constexpr(8):
            indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
            mask_1 = indices_1 < 1003
            {_LOAD}
    """
    source = textwrap.dedent(source) + textwrap.indent(row_loop, "    ")
    peeled = _peeled(source)
    assert peeled.count("for tile_offset_1 in range(") == 2
    assert peeled.count("tail_packet_0 = cute.arch.load(x.iterator") == 1
    assert peeled.count("for tile_offset_2 in range(") == 2
    assert peeled.count("vec = cute.arch.load(") == unmasked
    assert peeled.count("vec = _cute_load_v16b_if(") == 2 - unmasked


def test_a_row_past_the_launch_grid_is_proven_only_with_the_static_grid() -> None:
    # Each block reads its own 600-row split of 1200 rows: ``row < 1200``
    # holds for the two blocks of the launch, which only its static grid says.
    source = _SUNK_NEST.replace(
        "+ shift\n", "+ shift + cutlass.Int32(cute.arch.block_idx()[0]) * 600\n"
    ).replace("operator.lt(row, 600)", "operator.lt(row, 1200)")
    source = source.replace("keep and row < 600", "keep")
    assert source.count("600") == 2 and "1200" in source
    assert _peeled(source) == ast.unparse(ast.parse(textwrap.dedent(source)))
    assert "vec = cute.arch.load(x.iterator" in _peeled(source, grid=(2, 1, 1))
    assert _peeled(source, grid=(3, 1, 1)) == _peeled(source)


def test_sunk_witnesses_ride_on_the_deferred_combines() -> None:
    x = torch.empty(4800, 64, dtype=torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        spec = cpu_bind(split_column_sums, (x, x, 600)).config_spec
    (group,) = [g for g in spec.compiler_coverage_groups if g.key == "cute_tail_peel"]
    (dependency,) = group.dependencies
    assert (dependency.key, dependency.value) == ("cute_defer_tile_reduce", True)
    carriers = [witness.carrier.config for witness in group.witnesses]
    assert [(c["block_sizes"], c["num_threads"]) for c in carriers] == [
        ([1, 16, 128], [1, 2, 16]),
        ([1, 32, 128], [1, 4, 8]),
    ]
    for carrier in carriers:
        assert carrier["cute_defer_tile_reduce"] is True
        assert carrier["cute_vloop_sink"] is True
        # Each witness is code its carrier does not render.
        (_bounds, complete), _partial = _row_loops(
            _split_code(4800, 600, **{**carrier, "cute_tail_peel": True})
        )
        assert _MASKED_PACKET not in complete
        assert _MASKED_PACKET in _row_loops(_split_code(4800, 600, **carrier))[0][1]
    # A kernel whose loops carry no tile sum keeps the witnesses of its
    # one-warp seeds, without a dependency.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        stats = cpu_bind(online_softmax_stats, (padded_rows(4, 1003, "cpu"),))
    (plain,) = [
        g
        for g in stats.config_spec.compiler_coverage_groups
        if g.key == "cute_tail_peel"
    ]
    assert not plain.dependencies and plain.witnesses


def test_one_warp_seeds_carry_the_witnesses_no_deferred_witness_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A split column reduction with one more seed, which the one-warp rule
    # admits: the rows as eight vector lanes of 16 threads, 128-row tiles
    # that leave the last tile of a split's 600 rows partial.
    x = torch.empty(4700, 64, dtype=torch.bfloat16)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(split_column_sums, (x, x, 600))
        env, spec = bound.env, bound.config_spec
        spec.compiler_seed_configs.append(
            helion.Config.from_dict(
                {
                    **SPLIT_CONFIG,
                    "num_threads": [1, 16, 16],
                    "cute_vector_widths": [1, 1, 8],
                }
            )
        )

        def tail_peel_group() -> CompilerCoverageGroup:
            # The fields and the seeds stay; register the two groups again.
            spec._compiler_coverage_groups = ()
            spec._compiler_coverage_fields = ()
            register_defer_tile_reduce_coverage(env)
            register_tail_peel_coverage(env)
            (group,) = [
                group
                for group in spec.compiler_coverage_groups
                if group.key == "cute_tail_peel"
            ]
            return group

        # With both kinds of carriers the deferred witnesses take the group
        # (a field has one, and its carriers share the dependency).
        group = tail_peel_group()
        assert [entry.key for entry in group.dependencies] == ["cute_defer_tile_reduce"]
        assert [w.carrier.config["cute_vector_widths"] for w in group.witnesses] == [
            [1, 8, 1],
            [1, 8, 1],
        ]

        # Where the strict transfer takes the knob on none of them, the
        # one-warp seed carries the witness, as before the deferred combine.
        strict = ConfigGeneration.strict_config_pair

        def no_peel_on_a_deferred_combine(
            self: ConfigGeneration, config: helion.Config
        ) -> object:
            if config.config.get("cute_defer_tile_reduce") is True and (
                config.config.get("cute_tail_peel") is True
            ):
                raise exc.InvalidConfig("cute_tail_peel with a deferred combine")
            return strict(self, config)

        monkeypatch.setattr(
            ConfigGeneration, "strict_config_pair", no_peel_on_a_deferred_combine
        )
        group = tail_peel_group()
        assert not group.dependencies
        (witness,) = group.witnesses
        assert witness.value is True
        assert witness.carrier.config["cute_vector_widths"] == [1, 1, 8]
        assert witness.carrier.config["cute_defer_tile_reduce"] is False
