"""The in-place snapshot proof of ``fuse_two_pass_loads`` and the packet prefetch
of ``prefetch_reduction_packets`` compare names through the final rename's
groups: the rename merges a loop's output name into the carried name, so
``idx_next = idx + 1`` writes ``idx``."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from helion._compiler.cute import fuse_two_pass_loads as fuser_module
from helion._compiler.cute import prefetch_reduction_packets as prefetch_module
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.passes import pipeline
from helion._compiler.cute.prefetch_reduction_packets import prefetch_reduction_packets

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only


def _fuses(source: str, rename_groups: dict[str, str]) -> bool:
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
        rename_groups=rename_groups,
    )
    return "_fuse_cache_" in ast.unparse(ast.Module(body=result, type_ignores=[]))


def _snapshot_kernel(step: str) -> str:
    # The consume sweep reads x[lane] and stores x[idx] in place: the cache
    # of the first sweep serves the read only while each iteration stores
    # to its own element.
    return f"""
for synthetic_lane_7 in range(9):
    a = (x.iterator + synthetic_lane_7).load()
    acc = acc + a
for synthetic_lane_7 in range(9):
    idx = synthetic_lane_7
    b = (x.iterator + synthetic_lane_7).load()
    {step} = idx + 1
    (x.iterator + idx).store(b + 1)
"""


@pytest.mark.parametrize(
    ("step", "rename_groups", "fuses"),
    [
        # After the rename, idx_next is idx: iteration i stores x[i + 1],
        # which iteration i + 1 reads.
        ("idx_next", {"idx_next": "idx"}, False),
        ("idx_next", {}, True),
        ("idx", {}, False),
    ],
    ids=["merged", "unmerged", "rebound"],
)
def test_in_place_snapshot_proof_compares_names_through_rename_groups(
    step: str, rename_groups: dict[str, str], fuses: bool
) -> None:
    assert _fuses(_snapshot_kernel(step), rename_groups) is fuses


def test_in_place_snapshot_serves_only_the_element_the_first_sweep_cached() -> None:
    # After the rename, the consume sweep loads and stores x[lane + 1]: as
    # written, its load is the first sweep's x[lane], and its store another
    # element.
    source = """
for synthetic_lane_7 in range(9):
    a = (x.iterator + synthetic_lane_7).load()
    acc = acc + a
for synthetic_lane_7 in range(9):
    idx = synthetic_lane_7
    idx_next = synthetic_lane_7 + 1
    b = (x.iterator + idx).load()
    (x.iterator + idx_next).store(b + 1)
"""
    assert not _fuses(source, {"idx_next": "idx"})


def _prefetch_kernel(
    between: str,
    consumer: str,
    *,
    setup: str = "start = cutlass.Int32(0)",
    sweep: str = "range(start, start + 64, 64)",
    packet: str = "pw",
    producer: str = "",
) -> str:
    return f"""
cache = cute.make_rmem_tensor(4, cutlass.Uint16)
cute.arch.griddepcontrol_wait()
total = cutlass.Float32(0)
{setup}
for roffset_1 in {sweep}:
    for lane in range(1):
        before = roffset_1 + cute.arch.thread_idx()[0] * 4 + lane * 32
        {producer}
        px = cute.arch.load(x.iterator + before, ir.VectorType.get([4], cutlass.Uint16.mlir_type))
        for v in cutlass.range_constexpr(4):
            cache[lane * 4 + v] = px[v]
            total = total + cutlass.Float32(px[v])
total = cute.arch.warp_reduction_sum(total)
cute.arch.sync_threads()
{between}
for roffset_1 in {sweep}:
    for lane in range(1):
{consumer}
        valid = after < extent
        {packet} = _cute_load_l2_evict_last(weight.iterator + after if valid else weight.iterator, ir.VectorType.get([4], cutlass.Uint16.mlir_type))
        values = []
        for v in cutlass.range_constexpr(4):
            result = cutlass.Float32(cache[lane * 4 + v]) * total * cutlass.Float32(pw[v]) if valid else cutlass.Float32(0)
            values.append(cutlass.Uint16(result))
        if valid:
            _cute_store_u16_vec(out.iterator + after, values)
"""


def _prefetches(
    source: str,
    rename_groups: dict[str, str],
    constexpr_values: dict[str, int] | None = None,
) -> bool:
    result = prefetch_reduction_packets(
        ast.parse(source).body,
        constexpr_values=constexpr_values or {},
        thread_block_dims=(8, 1, 1),
        tensor_dtypes={
            "x": "cutlass.BFloat16",
            "weight": "cutlass.BFloat16",
            "out": "cutlass.BFloat16",
        },
        reload_modes={1: "register_prefetch"},
        argument_names={"x", "weight", "out", "extent"},
        proven_disjoint_tensor_pairs={frozenset(("weight", "out"))},
        rename_groups=rename_groups,
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    return "_consume_prefetch_" in code


_AFTER = "        after = roffset_1 + cute.arch.thread_idx()[0] * 4 + lane * 32"
_THROUGH_BASE = (
    "        base = roffset_1 + cute.arch.thread_idx()[0] * 4\n"
    "        base_next = base + 32\n"
    "        after = base + lane * 32"
)


@pytest.mark.parametrize(
    ("between", "consumer", "rename_groups", "prefetches"),
    [
        # After the rename, start_next is start: the later sweep runs over
        # the next slice, so a packet prefetched in the first one is not its.
        ("start_next = start + 64", _AFTER, {"start_next": "start"}, False),
        ("start_next = start + 64", _AFTER, {}, True),
        ("start = start + 64", _AFTER, {}, False),
        # After the rename, base_next is base: the address reads the base
        # the consumer advanced, which the moved definitions would not.
        ("", _THROUGH_BASE, {"base_next": "base"}, False),
        ("", _THROUGH_BASE, {}, True),
        # After the rename, cache_next is cache: the private buffer becomes
        # the weight tensor, which the store then writes ahead of the packet.
        ("cache_next = weight\ncache[0] = 123", _AFTER, {"cache_next": "cache"}, False),
        ("cache_next = weight\ncache[0] = 123", _AFTER, {}, True),
    ],
    ids=[
        "range-merged",
        "range-unmerged",
        "range-rebound",
        "address-merged",
        "address-unmerged",
        "private-merged",
        "private-unmerged",
    ],
)
def test_packet_prefetch_compares_names_through_rename_groups(
    between: str, consumer: str, rename_groups: dict[str, str], prefetches: bool
) -> None:
    assert _prefetches(_prefetch_kernel(between, consumer), rename_groups) is prefetches


@pytest.mark.parametrize(
    ("setup", "sweep", "rename_groups", "prefetches"),
    [
        # After the rename, stop_next is stop: the sweeps run twice, and both
        # trips would fill the one cache slot.
        ("stop = 64\nstop_next = stop + 64", "0, stop", {"stop_next": "stop"}, False),
        ("stop = 64\nstop_next = stop + 64", "0, stop", {}, True),
        ("stop = 64\nstop = stop + 64", "0, stop", {}, False),
        # The copy reads stop after the rename advanced it.
        (
            "stop = 64\nstop_next = stop + 64\nlimit = stop",
            "0, limit",
            {"stop_next": "stop"},
            False,
        ),
    ],
    ids=["merged", "unmerged", "rebound", "copy-merged"],
)
def test_single_trip_proof_compares_names_through_rename_groups(
    setup: str, sweep: str, rename_groups: dict[str, str], prefetches: bool
) -> None:
    source = _prefetch_kernel("", _AFTER, setup=setup, sweep=f"range({sweep}, 64)")
    assert _prefetches(source, rename_groups) is prefetches


@pytest.mark.parametrize(
    ("setup", "sweep", "rename_groups", "prefetches"),
    [
        # The constexpr stop gives the single-trip proof its value, which the
        # store of stop_next changes after the rename: the sweeps run twice.
        ("stop_next = stop + 64", "0, stop", {"stop_next": "stop"}, False),
        ("stop_next = stop + 64", "0, stop", {}, True),
        (
            "stop_next = stop + 64\nlimit = stop",
            "0, limit",
            {"stop_next": "stop"},
            False,
        ),
    ],
    ids=["merged", "unmerged", "copy-merged"],
)
def test_single_trip_proof_compares_constexpr_stores_through_rename_groups(
    setup: str, sweep: str, rename_groups: dict[str, str], prefetches: bool
) -> None:
    source = _prefetch_kernel("", _AFTER, setup=setup, sweep=f"range({sweep}, 64)")
    assert _prefetches(source, rename_groups, {"stop": 64}) is prefetches


@pytest.mark.parametrize(
    ("rename_groups", "prefetches"),
    [
        # After the rename, the consumer reads the packet it loads as pw, a
        # name the move would leave as it was.
        ({"pw_next": "pw"}, False),
        # Without it, pw is another tensor, and the packet a dead load.
        ({}, True),
    ],
    ids=["merged", "unmerged"],
)
def test_packet_prefetch_compares_the_packet_name_through_rename_groups(
    rename_groups: dict[str, str], prefetches: bool
) -> None:
    source = _prefetch_kernel(
        "",
        _AFTER,
        setup="start = cutlass.Int32(0)\npw = cute.make_rmem_tensor(4, cutlass.Uint16)",
        packet="pw_next",
    )
    assert _prefetches(source, rename_groups) is prefetches


@pytest.mark.parametrize(
    ("rename_groups", "prefetches"),
    [
        # After the rename, lane_next = lane + 1 advances the producer's lane
        # index, so its cache slot is not the one the consumer reads.
        ({"lane_next": "lane"}, False),
        ({}, True),
    ],
    ids=["merged", "unmerged"],
)
def test_packet_prefetch_compares_the_induction_variables_through_rename_groups(
    rename_groups: dict[str, str], prefetches: bool
) -> None:
    source = _prefetch_kernel(
        "",
        "        after = roffset_1 + cute.arch.thread_idx()[0] * 4",
        producer="lane_next = lane + 1",
    )
    assert _prefetches(source, rename_groups) is prefetches


@pytest.mark.parametrize(
    ("rename_groups", "fuses"),
    [
        # After the rename, idx_next = lane + 1 rebinds idx: the consume loop
        # loads and stores x[lane + 1], each spelling its own element, while
        # its load was matched to the first sweep's x[lane].
        ({"idx_next": "idx"}, False),
        ({}, True),
    ],
    ids=["merged", "unmerged"],
)
def test_in_place_snapshot_declines_a_loop_holding_two_spellings_of_a_group(
    rename_groups: dict[str, str], fuses: bool
) -> None:
    source = """
for synthetic_lane_7 in range(9):
    a = (x.iterator + synthetic_lane_7).load()
    acc = acc + a
for synthetic_lane_7 in range(9):
    idx = synthetic_lane_7
    idx_next = synthetic_lane_7 + 1
    b = (x.iterator + idx).load()
    (x.iterator + idx).store(b + 1)
"""
    assert _fuses(source, rename_groups) is fuses


_VECTOR = "ir.VectorType.get([4], cutlass.Float32.mlir_type)"


@pytest.mark.parametrize(
    ("rename_groups", "read", "fuses"),
    [
        # After the rename, pw[0] reads the packet loaded into pw_next, a read
        # the cache rewrite of pw_next would leave behind.
        ({"pw_next": "pw"}, "pw", False),
        ({}, "pw_next", True),
    ],
    ids=["merged", "unmerged"],
)
def test_a_vector_load_under_a_merged_name_is_not_fused(
    rename_groups: dict[str, str], read: str, fuses: bool
) -> None:
    source = f"""
for synthetic_lane_7 in range(8):
    pv = cute.arch.load(x.iterator + synthetic_lane_7 * 4, {_VECTOR})
    acc = acc + cutlass.Float32(pv[0])
for synthetic_lane_7 in range(8):
    pw_next = cute.arch.load(x.iterator + synthetic_lane_7 * 4, {_VECTOR})
    total = total + cutlass.Float32({read}[0])
"""
    assert _fuses(source, rename_groups) is fuses


def test_the_pipeline_passes_the_rename_groups_to_both_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def stub(name: str) -> Callable[..., list[ast.stmt]]:
        def run(
            body: list[ast.stmt], *args: object, **kwargs: object
        ) -> list[ast.stmt]:
            seen[name] = kwargs.get("rename_groups")
            return body

        return run

    monkeypatch.setattr(fuser_module, "fuse_two_pass_loads", stub("fuser"))
    monkeypatch.setattr(prefetch_module, "prefetch_reduction_packets", stub("prefetch"))
    state = MagicMock()
    pipeline._fuse_two_pass_loads(state)
    pipeline._prefetch_reduction_packets(state)
    assert seen == {
        "fuser": state.ctx.rename_groups,
        "prefetch": state.ctx.rename_groups,
    }
