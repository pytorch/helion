"""Tile transport of the fragment lowering for the anchored chunk kernels.

``cute_fragment_vector_io`` (16-byte row packets for proved host windows),
``cute_fragment_batched_loads`` (one barrier per run of independent host
loads), ``cute_fragment_split_k`` (idle warps share K in the tf32 mma phase),
the in-place evaluation of coordinate-only mma operands and the direct
coordinate recovery of Inductor's affine indices.  The CPU tests pin the
search surface and the generated code; the GPU tests check the numerics
against torch.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_fp32_dot import _chunk_cumsum
from test.test_cute_fragment_tile_staging import _anchored_gram
from test.test_cute_fragment_tile_staging import _anchored_gram_reference
from test.test_cute_fragment_tile_staging import _gram_inputs

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import BooleanFragment
import helion.language as hl

VEC = "cute_fragment_vector_io"
BATCH = "cute_fragment_batched_loads"
SPLIT = "cute_fragment_split_k"
SLAB = "cute_fragment_slab_loads"
DOT = "cute_fp32_dot"
PAD = "cute_fragment_padded_rows"
MMA = "_helion_fragment_mma.mma_tf32("
SYNC = "cute.arch.sync_threads()"
PACKET = "num_bits_per_copy=128"

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _odd_width_gram(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A gram whose rows are 100 bf16 wide: no 16-byte packet divides them."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    A = torch.zeros([BHN, C, C], dtype=q.dtype, device=q.device)
    for tile_b in hl.tile(BHN, block_size=1):
        rows = hl.arange(16)
        cols = hl.arange(16)
        a = hl.dot(
            q[tile_b, rows, :].float(), k[tile_b, cols, :].float().transpose(-2, -1)
        )
        A[tile_b, rows, cols] = a.to(A.dtype)
    return A


@pytest.fixture
def _cpu_only():
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        yield


def _config(bound: object, **overrides: object) -> helion.Config:
    spec = bound.config_spec  # type: ignore[attr-defined]
    return helion.Config.from_dict(dict(spec.default_config()) | overrides)


def test_transport_knobs_join_the_search_surface(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    spec = bound.config_spec
    assert spec.cute_fragment_vector_io_root_ids
    assert spec.cute_fragment_batched_loads_root_ids
    assert spec.cute_fragment_split_k_root_ids
    fields = spec._flat_fields()
    default = spec.default_config()
    for key in (VEC, BATCH, SPLIT):
        assert isinstance(fields[key], BooleanFragment)
        assert key not in default
        # Each knob is a deferred coverage witness of its own.
        assert any(
            group.key == key and any(w.value is True for w in group.witnesses)
            for group in spec.compiler_coverage_groups
        )
    for key in (VEC, BATCH):
        off = _config(bound, **{key: False})
        spec.normalize(off.config)
        assert off == default
        with pytest.raises(exc.InvalidConfig, match="complete fragment root"):
            spec.normalize(_config(bound, **{key: 1}).config)
    # The split rides the tf32 mma phase: alone it is rejected, and the
    # fixing normalization drops it instead of pinning an inert knob.
    with pytest.raises(exc.InvalidConfig, match="tf32_mma"):
        spec.normalize(_config(bound, **{SPLIT: True}).config)
    generation = spec.create_config_generation()
    _, effective = generation.strict_config_pair(
        _config(bound, **{DOT: "tf32_mma", SPLIT: True})
    )
    assert effective[SPLIT] is True and effective[DOT] == "tf32_mma"
    # The split's witness carries the tf32 mode.
    (group,) = [g for g in spec.compiler_coverage_groups if g.key == SPLIT]
    assert all(w.carrier.config.get(DOT) == "tf32_mma" for w in group.witnesses)


def test_a_root_without_a_second_load_has_no_batching(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    spec = bound.config_spec
    assert not spec.cute_fragment_batched_loads_root_ids
    assert BATCH not in spec._flat_fields()
    assert spec.cute_fragment_vector_io_root_ids
    assert VEC in spec._flat_fields()


def test_odd_rows_decline_the_packets(_cpu_only: None) -> None:
    q = torch.ones(4, 32, 100, dtype=torch.bfloat16)
    bound = _cpu_bind(_odd_width_gram, (q, q))
    # The heuristic offers the knob (the 32-column output store qualifies);
    # the emitter proves 100 % 8 != 0 for the q / k windows and keeps their
    # scalar loads, while the iota-window store into the zero-filled output
    # (16 of 32 bf16 columns, 16-byte aligned rows) moves as packets.
    assert bound.config_spec.cute_fragment_vector_io_root_ids
    dense = bound.to_code(_config(bound))
    vec = bound.to_code(_config(bound, **{VEC: True}))
    assert dense.count(".load()") == vec.count(".load()") == 2
    assert dense.count(".store(") == 1 and ".store(" not in vec
    assert vec.count(PACKET) == 1
    assert dense.count(SYNC) == vec.count(SYNC)


def test_row_packets_replace_every_scalar_access(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    scalar = bound.to_code(_config(bound))
    vec = bound.to_code(_config(bound, **{VEC: True}))
    # Ten window loads and two iota-window stores: every one of them moves
    # through 16-byte packets (two copies per load: global -> registers ->
    # shared; one per store), with one address and one predicate per packet.
    assert scalar.count(".load()") == 10 and scalar.count(".store(") == 2
    assert ".load()" not in vec and ".store(" not in vec
    assert vec.count("cute.copy(") == 22
    assert vec.count(PACKET) == 22
    # bf16 packets are eight lanes, fp32 packets four; the shared-memory
    # side asserts the packet's alignment to the compiler.
    assert "cute.make_layout((8,)), cutlass.BFloat16" in vec
    assert "cute.make_layout((4,)), cutlass.Float32" in vec
    assert "cute.assume(" in vec
    assert re.search(r"fragment_packet_inner(_\d+)? = .* \* 8\n", vec)
    # Staging and barriers are untouched: same buffers, same phases.
    assert re.findall(r"allocate_tensor\(.*\)", scalar) == re.findall(
        r"allocate_tensor\(.*\)", vec
    )
    assert scalar.count(SYNC) == vec.count(SYNC) == 18


def test_batched_loads_share_one_barrier_per_run(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    serial = bound.to_code(_config(bound))
    batched = bound.to_code(_config(bound, **{BATCH: True}))
    # Each tier stages five windows (the anchor row, two row windows, two
    # column windows) before anything reads them: four barriers fewer per
    # tier, every load still issued, every consumer still fenced.
    assert serial.count(SYNC) == 18
    assert batched.count(SYNC) == 10
    assert serial.count(".load()") == batched.count(".load()") == 10
    # No two consecutive barriers, and the first barrier follows the last
    # load of the run rather than the first.
    assert "sync_threads()\n    cute.arch.sync_threads()" not in batched
    first_barrier = batched.index(SYNC)
    assert batched[:first_barrier].count(".load()") == 5


def test_coordinate_only_operands_are_read_in_place(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    simt = bound.to_code(_config(bound))
    tf32 = bound.to_code(_config(bound, **{DOT: "tf32_mma"}))
    # The lower-triangular mask is a pure function of its coordinates: the
    # mma lanes evaluate the compare in place.  No staging phase, barrier
    # or buffer for it; the chunk tile keeps its padded staged copy.  The
    # result's only reader is the host store, so the lanes write it directly
    # (test_cute_fragment_direct_store): one barrier and one buffer fewer.
    assert tf32.count(MMA) == 1
    assert simt.count(SYNC) - tf32.count(SYNC) == 2
    assert len(re.findall(r"allocate_tensor\(", tf32)) == 2
    loop = tf32.split("for fragment_mma_tile")[1].split(SYNC)[0]
    assert "operator.ge(" in loop
    # Inductor's affine index names each coordinate outright: no
    # ``// stride % extent`` recovery round trip inside the lane loop.
    assert not re.search(r"\(\(.*\) // 64\)", loop)


def test_direct_coordinates_keep_the_fallback_for_non_affine_indices(
    _cpu_only: None,
) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    code = bound.to_code(_config(bound, **{DOT: "tf32_mma", PAD: True}))
    # The staged q window feeds the operand through its layout's affine
    # index, read with its own coordinates (the 136-stride bf16 rows)...
    assert "* 136 +" in code
    # ...while the element walks still unflatten their thread index.
    assert re.search(r"fragment_index(_\d+)? // 128 % 16", code)


def test_split_k_shares_a_tile_between_idle_warps(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    whole = bound.to_code(_config(bound, **{DOT: "tf32_mma", PAD: True}))
    split = bound.to_code(_config(bound, **{DOT: "tf32_mma", PAD: True, SPLIT: True}))
    assert "fragment_mma_split" not in whole and "fragment_mma_unit" not in whole
    # Tier 0 is two 16x8 tiles over K = 128 (16 steps) for four warps: two
    # splits of eight steps each; tier 1 has four tiles for four warps and
    # keeps the whole walk.
    assert split.count("fragment_mma_split") > 0
    units = re.findall(
        r"for fragment_mma_unit(_\d+)? in range\(fragment_thread // 32, (\d+), 4\)",
        split,
    )
    assert [count for _, count in units] == ["4"]
    assert "cutlass.range_constexpr(8):" in split
    assert "cutlass.range_constexpr(16):" in split
    # The partials are summed at the read: two fp32 reads per element.
    assert re.search(
        r"\(fragment_buffer(_\d+)?\[0 \* \d+ \+ .*\] \+ fragment_buffer", split
    )
    assert whole.count(SYNC) == split.count(SYNC)


def test_slab_loads_stage_each_tensor_once(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    spec = bound.config_spec
    assert spec.cute_fragment_slab_loads_root_ids
    assert isinstance(spec._flat_fields()[SLAB], BooleanFragment)
    windows = bound.to_code(_config(bound))
    slabs = bound.to_code(_config(bound, **{SLAB: True}))
    # Ten window loads of three tensors (two tiers of q rows, k columns, gc
    # rows / columns / anchor rows) become three 32-row slabs staged at the
    # start of the root; every window is a view at its static offset.
    assert windows.count(".load()") == 10
    assert slabs.count(".load()") == 3
    assert slabs.count(SYNC) == 11 and windows.count(SYNC) == 18
    first_window_use = slabs.index("fragment_coordinate")
    assert slabs[:first_window_use].count(SYNC) == 3
    sizes = re.findall(
        r"allocate_tensor\((cutlass\.\w+), cute\.make_layout\(\((\d+),\)\)", slabs
    )
    assert sizes[:3] == [
        ("cutlass.Float32", "4096"),
        ("cutlass.BFloat16", "4096"),
        ("cutlass.BFloat16", "4096"),
    ]
    # Tier 1's row window and anchor row read rows 16.. of the gc slab.
    assert re.search(r"\(fragment_coordinate_\d+ \+ 16\) \* 128", slabs)
    assert re.search(r"fragment_buffer\[0 \* 4096 \+ 16 \* 128 \+", slabs)
    # With the transport knobs the three slabs are one batched packet phase.
    everything = bound.to_code(_config(bound, **{SLAB: True, VEC: True, BATCH: True}))
    assert everything.count(SYNC) == 9
    assert everything.count("cute.copy(") == 8  # 3 slabs x 2 + 2 stores
    assert ".load()" not in everything


def test_slab_plan_declines_a_single_window_root(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    assert not bound.config_spec.cute_fragment_slab_loads_root_ids
    assert SLAB not in bound.config_spec._flat_fields()


def test_a_cast_of_a_conflict_free_tile_is_read_in_place(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    # A 64-column bf16 chunk tile has 128-byte rows: padded rows stride
    # 72 elements, which spreads the B lanes' reads over the banks, so the
    # ``.to(float32)`` operand is read (and converted) in place: no padded
    # fp32 copy, one barrier fewer.  Dense 128-byte rows conflict four ways
    # and keep the copy.  (The result itself is stored from the lanes in
    # both renders: test_cute_fragment_direct_store.)
    dense = bound.to_code(_config(bound, **{DOT: "tf32_mma", "block_sizes": [64]}))
    padded = bound.to_code(
        _config(bound, **{DOT: "tf32_mma", PAD: True, "block_sizes": [64]})
    )
    assert dense.count(SYNC) == 3 and padded.count(SYNC) == 2
    assert len(re.findall(r"allocate_tensor\(", dense)) == 2
    assert len(re.findall(r"allocate_tensor\(", padded)) == 1
    loop = padded.split("for fragment_mma_tile")[1].split(SYNC)[0]
    assert "* 72 +" in loop  # the bf16 tile, read by the lanes
    assert "cutlass.Float32(" in loop


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("vec", "batch", "split"),
    [(False, False, False), (True, True, False), (True, True, True)],
)
def test_anchored_gram_slabs_match_reference(
    vec: bool, batch: bool, split: bool
) -> None:
    torch.manual_seed(0)
    q = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    gc = (-torch.rand(64, 32, 128, device=DEVICE) * 0.1).cumsum(1)
    scale = 0.125
    bound = _anchored_gram.bind((q, k, gc, scale))
    knobs = {
        DOT: "tf32_mma",
        PAD: True,
        "cute_fragment_unrolled_loops": True,
        "cute_fragment_threads": 256,
        SLAB: True,
        VEC: vec,
        BATCH: batch,
    }
    if split:
        knobs[SPLIT] = True
    config = _config(bound, **knobs)
    code = bound.to_code(config)
    assert code.count(".load()") + code.count("cute.copy(") // 2 <= 3 + 2
    out = bound.compile_config(config)(q, k, gc, scale)
    torch.cuda.synchronize()
    expected = _anchored_gram_reference(q, k, gc, scale)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    if not split:
        plain = bound.compile_config(
            _config(bound, **(knobs | {SLAB: False, VEC: False, BATCH: False}))
        )(q, k, gc, scale)
        torch.cuda.synchronize()
        assert torch.equal(out, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_in_place_cast_operand_matches_reference() -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = _chunk_cumsum.bind((g,))
    config = _config(
        bound, **{DOT: "tf32_mma", PAD: True, VEC: True, "block_sizes": [64]}
    )
    code = bound.to_code(config)
    # One barrier after the packet load, one after the lane loop (which
    # stores the result itself: test_cute_fragment_direct_store).
    assert code.count(SYNC) == 2
    out = bound.compile_config(config)(g)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, g.float().cumsum(1), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dot", "threads", "vec", "batch", "split"),
    [
        ("simt", 128, True, False, False),
        ("simt", 256, False, True, False),
        ("tf32_mma", 128, True, True, False),
        ("tf32_mma", 256, True, True, True),
        ("tf32_mma", 512, False, False, True),
    ],
)
def test_anchored_gram_transport_matches_reference(
    dot: str, threads: int, vec: bool, batch: bool, split: bool
) -> None:
    torch.manual_seed(0)
    q = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    gc = (-torch.rand(64, 32, 128, device=DEVICE) * 0.1).cumsum(1)
    scale = 0.125
    bound = _anchored_gram.bind((q, k, gc, scale))
    knobs = {
        DOT: dot,
        PAD: True,
        "cute_fragment_unrolled_loops": True,
        "cute_fragment_threads": threads,
        VEC: vec,
        BATCH: batch,
    }
    if split:
        knobs[SPLIT] = True
    config = _config(bound, **knobs)
    code = bound.to_code(config)
    assert (PACKET in code) == vec
    assert ("fragment_mma_split" in code) == split
    out = bound.compile_config(config)(q, k, gc, scale)
    torch.cuda.synchronize()
    expected = _anchored_gram_reference(q, k, gc, scale)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    # The transport knobs move the same scalars: the staged tiles, and so
    # the result, are bitwise those of the scalar walk.
    if not split:
        plain = bound.compile_config(
            _config(bound, **(knobs | {VEC: False, BATCH: False}))
        )(q, k, gc, scale)
        torch.cuda.synchronize()
        assert torch.equal(out, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("threads", [128, 256])
def test_cumsum_row_packets_match_reference(threads: int) -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = _chunk_cumsum.bind((g,))
    config = _config(
        bound, **{DOT: "tf32_mma", VEC: True, "cute_fragment_threads": threads}
    )
    code = bound.to_code(config)
    assert MMA in code and PACKET in code
    out = bound.compile_config(config)(g)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, g.float().cumsum(1), rtol=1e-5, atol=1e-5)
    plain = bound.compile_config(
        _config(bound, **{DOT: "tf32_mma", "cute_fragment_threads": threads})
    )(g)
    torch.cuda.synchronize()
    assert torch.equal(out, plain)
