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


# -- windows at a runtime row offset (the varlen anchored gram) ---------------


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _varlen_anchored_gram(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The two tiers of ``_anchored_gram`` over token-major q / k
    ``[T, H, D]``: chunk ``j`` of head ``h`` reads its rows at
    ``token_base[j]`` (the same static windows shifted by one runtime
    offset), each row live below ``valid_len[j]`` and zero past it; gc is
    chunk-major ``[H * NT, C, D]``."""
    NT = token_base.size(0)
    HNT = gc.size(0)
    C = hl.specialize(gc.size(1))
    D = hl.specialize(gc.size(2))
    A = torch.zeros([HNT, C, C], dtype=q.dtype, device=q.device)
    for tile_r in hl.tile(HNT, block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        dcols = hl.arange(D)
        rows0 = hl.arange(16)
        cols0 = hl.arange(16)
        m0 = rows0 < vlen
        q0 = torch.where(
            m0[:, None],
            hl.load(q, [base + rows0, h, dcols], extra_mask=m0[:, None]),
            0,
        ).float()
        k0 = torch.where(
            m0[:, None],
            hl.load(k, [base + rows0, h, dcols], extra_mask=m0[:, None]),
            0,
        ).float()
        n0 = gc[r, 0, :]
        e0 = torch.exp2(gc[r, rows0, :] - n0[None, :])
        kg0 = (k0 * torch.exp2(n0[None, :] - gc[r, rows0, :])).transpose(-2, -1)
        a0 = hl.dot(q0 * e0, kg0) * scale
        A[r, rows0, cols0] = a0 * (rows0[:, None] >= cols0[None, :]).to(a0.dtype)

        rows1 = 16 + hl.arange(16)
        cols1 = hl.arange(32)
        m1 = rows1 < vlen
        mc1 = cols1 < vlen
        q1 = torch.where(
            m1[:, None],
            hl.load(q, [base + rows1, h, dcols], extra_mask=m1[:, None]),
            0,
        ).float()
        kc1 = torch.where(
            mc1[:, None],
            hl.load(k, [base + cols1, h, dcols], extra_mask=mc1[:, None]),
            0,
        ).float()
        n1 = gc[r, 16, :]
        e1 = torch.exp2(gc[r, rows1, :] - n1[None, :])
        ec1 = torch.exp2(n1[None, :] - gc[r, cols1, :])
        a1 = hl.dot(q1 * e1, (kc1 * ec1).transpose(-2, -1)) * scale
        A[r, rows1, cols1] = a1 * (rows1[:, None] >= cols1[None, :]).to(a1.dtype)
    return A


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mismatched_varlen_windows(
    x: torch.Tensor, token_base: torch.Tensor, valid_len: torch.Tensor
) -> torch.Tensor:
    """Three windows of x at one runtime offset that must not share a slab:
    the second's mask is over another window than its rows (rows 16.. under
    ``rows 0.. < vlen``), the third has no mask (its rows past the limit
    are the next sequence's, not zero)."""
    NT = token_base.size(0)
    HNT = x.size(1) * NT
    out = torch.zeros([HNT, 48, x.size(2)], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(HNT, block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        rows0 = hl.arange(16)
        rows1 = 16 + hl.arange(16)
        rows2 = 32 + hl.arange(16)
        m0 = rows0 < vlen
        a = hl.load(x, [base + rows0, h, slice(None)], extra_mask=m0[:, None])
        b = hl.load(x, [base + rows1, h, slice(None)], extra_mask=m0[:, None])
        c = hl.load(x, [base + rows2, h, slice(None)])
        out[r, rows0, :] = a.float() * 2.0
        out[r, rows1, :] = b.float() * 3.0
        out[r, rows2, :] = c.float() * 5.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _contiguous_dim_windows(x: torch.Tensor) -> torch.Tensor:
    """Two static windows of ``x [B, D, N]`` varying along its contiguous
    dimension (N's rows 0.. and 16..): one 32-wide slab whose varying
    dimension is the contiguous one, walked element by element."""
    B = x.size(0)
    D = hl.specialize(x.size(1))
    out = torch.zeros([B, 16, 16], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(B, block_size=1):
        rows0 = hl.arange(16)
        rows1 = 16 + hl.arange(16)
        cols0 = hl.arange(16)
        dcols = hl.arange(D)
        a = x[tile_b, dcols, rows0]
        b = x[tile_b, dcols, rows1]
        out[tile_b, rows0, cols0] = hl.dot(
            (a.float() * 2.0).transpose(-2, -1), b.float()
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _varlen_contiguous_dim_windows(
    x: torch.Tensor, token_base: torch.Tensor, valid_len: torch.Tensor
) -> torch.Tensor:
    """``_contiguous_dim_windows`` at a runtime offset: ``x [H, D, T]``
    (tokens contiguous) read at ``token_base[j] + rows`` under
    ``(rows < valid_len[j])[None, :]``, the limit broadcast over the
    leading dim."""
    NT = token_base.size(0)
    H = x.size(0)
    D = hl.specialize(x.size(1))
    HNT = H * NT
    out = torch.zeros([HNT, 16, 16], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(HNT, block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        rows0 = hl.arange(16)
        rows1 = 16 + hl.arange(16)
        cols0 = hl.arange(16)
        dcols = hl.arange(D)
        m0 = rows0 < vlen
        m1 = rows1 < vlen
        a = torch.where(
            m0[None, :],
            hl.load(x, [h, dcols, base + rows0], extra_mask=m0[None, :]),
            0,
        )
        b = torch.where(
            m1[None, :],
            hl.load(x, [h, dcols, base + rows1], extra_mask=m1[None, :]),
            0,
        )
        out[r, rows0, cols0] = hl.dot((a.float() * 2.0).transpose(-2, -1), b.float())
    return out


def _varlen_tables(
    lens: tuple[int, ...], c: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(token_base [NT], valid_len [NT], NT)`` of sequences of ``lens``
    tokens cut into chunks of ``c`` rows (the last chunk of a sequence
    ragged)."""
    token_base: list[int] = []
    valid_len: list[int] = []
    start = 0
    for length in lens:
        for i in range(0, length, c):
            token_base.append(start + i)
            valid_len.append(min(length - i, c))
        start += length
    i64 = torch.int64
    return (
        torch.tensor(token_base, dtype=i64, device=device),
        torch.tensor(valid_len, dtype=i64, device=device),
        len(token_base),
    )


def _varlen_gram_inputs(
    lens: tuple[int, ...] = (37, 64, 27),
    h: int = 2,
    d: int = 128,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> tuple[torch.Tensor, ...]:
    token_base, valid_len, nt = _varlen_tables(lens, 32, device)
    total = sum(lens)
    if seed is None:
        q = torch.ones(total, h, d, dtype=torch.bfloat16, device=device)
        k = torch.ones(total, h, d, dtype=torch.bfloat16, device=device)
        gc = torch.zeros(h * nt, 32, d, dtype=torch.float32, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(total, h, d, device=device, generator=g).to(torch.bfloat16)
        k = torch.randn(total, h, d, device=device, generator=g).to(torch.bfloat16)
        gc = (-torch.rand(h * nt, 32, d, device=device, generator=g) * 0.1).cumsum(1)
    return q, k, gc, token_base, valid_len, 0.125


def _varlen_contiguous_inputs(
    lens: tuple[int, ...] = (37, 64, 27),
    h: int = 2,
    d: int = 32,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_base, valid_len, _nt = _varlen_tables(lens, 32, device)
    total = sum(lens)
    if seed is None:
        x = torch.ones(h, d, total, dtype=torch.float32, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(h, d, total, device=device, generator=g)
    return x, token_base, valid_len


def _varlen_contiguous_dim_windows_reference(
    x: torch.Tensor, token_base: torch.Tensor, valid_len: torch.Tensor
) -> torch.Tensor:
    """Every chunk's 32 token columns gathered (zero past the sequence's
    end), then ``(2 a)^T b`` over its two 16-column halves."""
    h, d, t = x.shape
    nt = token_base.numel()
    rows = torch.arange(32, device=x.device)
    index = (token_base[:, None] + rows[None, :]).clamp(max=t - 1)
    valid = (rows[None, :] < valid_len[:, None]).to(x.dtype)
    # [H, NT, D, 32], row h * NT + j
    chunks = x[:, :, index].permute(0, 2, 1, 3) * valid[None, :, None, :]
    chunks = chunks.reshape(h * nt, d, 32)
    a = chunks[:, :, :16] * 2.0
    b = chunks[:, :, 16:]
    return a.transpose(-2, -1) @ b


def _varlen_anchored_gram_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The dense reference over every chunk's rows gathered (zero past the
    sequence's end)."""
    nt = token_base.numel()
    h = q.shape[1]
    rows = torch.arange(32, device=q.device)
    index = (token_base[:, None] + rows[None, :]).clamp(max=q.shape[0] - 1)
    valid = (rows[None, :] < valid_len[:, None]).to(q.dtype)
    # [H * NT, 32, D], row h * NT + j
    qc = torch.cat([q[index, head] * valid[:, :, None] for head in range(h)])
    kc = torch.cat([k[index, head] * valid[:, :, None] for head in range(h)])
    assert qc.shape[0] == gc.shape[0] == h * nt
    return _anchored_gram_reference(qc, kc, gc, scale)


def test_varlen_windows_at_one_runtime_offset_share_a_slab(_cpu_only: None) -> None:
    from helion._compiler.cute.fragment_slab_loads import plan_slab_loads

    bound = _cpu_bind(_varlen_anchored_gram, _varlen_gram_inputs())
    spec = bound.config_spec
    # q (two row tiers) and k (rows, then columns) are read at the runtime
    # offset token_base[j] under the valid_len[j] limit; gc's windows are
    # static at the uniform chunk row tile_r.begin: three slabs.
    assert spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        plan = plan_slab_loads(bound.env, bound.host_function.device_ir.graphs)
    groups = {id(group): group for group in plan.values()}
    assert len(groups) == 3 and len(plan) == 10
    by_tensor = {group.tensor.args[0]: group for group in groups.values()}
    assert sorted(by_tensor) == ["gc", "k", "q"]
    for name in ("q", "k"):
        group = by_tensor[name]
        assert group.offset is not None and group.offset.tensor == "token_base"
        assert group.limit is not None and group.limit.tensor == "valid_len"
        assert (group.start, group.extent) == (0, 32)
    assert by_tensor["gc"].offset is None and by_tensor["gc"].limit is None
    assert by_tensor["gc"].descriptors[0][0] == "uniform"
    assert sorted(by_tensor["q"].members.values()) == [(0, 16), (16, 16)]
    assert sorted(by_tensor["k"].members.values()) == [(0, 16), (0, 32)]
    windows = bound.to_code(_config(bound))
    slabs = bound.to_code(_config(bound, **{SLAB: True}))
    # Ten window loads + the two 0-d table loads become the three slab loads
    # staged ahead of the body, behind the offset and the limit read once
    # per CTA (the body's own 0-d loads of them stay).
    assert windows.count(".load()") == 12
    assert slabs.count(".load()") == 7
    assert slabs.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    assert slabs.count("fragment_slab_scalar_1 = cutlass.Int32(0)") == 1
    assert "(token_base.iterator + " in slabs and "(valid_len.iterator + " in slabs
    # Every slab row is predicated by the limit (in the window's own
    # coordinates) and the tensor's end at the shifted row.
    assert len(re.findall(r"% 32 < fragment_slab_scalar_1\n", slabs)) == 2
    assert len(re.findall(r"fragment_slab_scalar \+ 0 \+ .+? % 32 < \d+\n", slabs)) == 2
    assert slabs.count(SYNC) < windows.count(SYNC)
    # The members read views at their static offsets: tier 1's q rows 16..
    # of the q slab; k's columns the whole k slab.
    assert re.search(r"\(fragment_coordinate_\d+ \+ 16\) \* 128", slabs)
    # With the packets the q / k slabs move as eight-lane bf16 packets under
    # one row predicate each; the gc slab as four-lane fp32 packets.
    everything = bound.to_code(_config(bound, **{SLAB: True, VEC: True, BATCH: True}))
    assert everything.count(".load()") == 4  # the two tables, twice
    assert everything.count("cute.copy(") == 3 * 2 + 2
    assert "cute.make_layout((8,)), cutlass.BFloat16" in everything
    assert "cute.make_layout((4,)), cutlass.Float32" in everything
    assert everything.count(SYNC) < slabs.count(SYNC)


def test_varlen_windows_under_another_mask_keep_their_walk(_cpu_only: None) -> None:
    from helion._compiler.cute.fragment_slab_loads import plan_slab_loads

    token_base, valid_len, nt = _varlen_tables((37, 64, 27), 48)
    x = torch.ones(128, 2, 64, dtype=torch.bfloat16)
    bound = _cpu_bind(_mismatched_varlen_windows, (x, token_base, valid_len))
    # A mask over another window than the rows it guards, or no mask at
    # all beside a masked member, is not the slab's row predicate: no group.
    assert not bound.config_spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        assert plan_slab_loads(bound.env, bound.host_function.device_ir.graphs) == {}


def test_a_slab_along_the_contiguous_dim_is_walked(_cpu_only: None) -> None:
    from helion._compiler.cute.fragment_slab_loads import plan_slab_loads

    bound = _cpu_bind(_contiguous_dim_windows, (torch.ones(4, 32, 128),))
    spec = bound.config_spec
    assert spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        plan = plan_slab_loads(bound.env, bound.host_function.device_ir.graphs)
    groups = {id(group): group for group in plan.values()}
    assert len(groups) == 1 and len(plan) == 2
    (group,) = groups.values()
    assert [kind for kind, _ in group.descriptors] == ["block", "window", "window"]
    assert group.window_dim == 2 and (group.start, group.extent) == (0, 32)
    assert group.offset is None and group.limit is None
    windows = bound.to_code(_config(bound))
    slabs = bound.to_code(_config(bound, **{SLAB: True}))
    # The two windows are one slab load; the second reads its columns 16...
    assert windows.count(".load()") == 2 and slabs.count(".load()") == 1
    assert slabs.count(SYNC) < windows.count(SYNC)
    assert re.search(r"\+ 16\) \* 1\]", slabs)
    # The plain windows move as packets along the contiguous dim (two loads
    # and the store); the slab varies along it and is walked: with the
    # packets knob only the store is a packet copy.
    vectors = bound.to_code(_config(bound, **{VEC: True}))
    everything = bound.to_code(_config(bound, **{SLAB: True, VEC: True}))
    assert vectors.count(PACKET) == 3 and everything.count(PACKET) == 1
    assert everything.count(".load()") == 1
    assert everything.count("cute.copy(") == 1 and "cute.copy(" not in slabs


def test_varlen_windows_along_the_contiguous_dim_share_a_slab(
    _cpu_only: None,
) -> None:
    from helion._compiler.cute.fragment_slab_loads import plan_slab_loads

    bound = _cpu_bind(_varlen_contiguous_dim_windows, _varlen_contiguous_inputs())
    spec = bound.config_spec
    assert spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        plan = plan_slab_loads(bound.env, bound.host_function.device_ir.graphs)
    groups = {id(group): group for group in plan.values()}
    assert len(groups) == 1 and len(plan) == 2
    (group,) = groups.values()
    # The head is the uniform scalar tile_r.begin // NT; the windows along
    # the contiguous token dim share token_base[j] under valid_len[j].
    assert [kind for kind, _ in group.descriptors] == ["uniform", "window", "window"]
    assert group.window_dim == 2 and (group.start, group.extent) == (0, 32)
    assert group.offset is not None and group.limit is not None
    assert group.offset.tensor == "token_base" and group.limit.tensor == "valid_len"
    windows = bound.to_code(_config(bound))
    slabs = bound.to_code(_config(bound, **{SLAB: True}))
    # Two window loads + two 0-d table loads -> one slab load + the offset
    # and limit read once ahead of it + the body's two table loads.
    assert windows.count(".load()") == 4 and slabs.count(".load()") == 5
    assert slabs.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    assert slabs.count("fragment_slab_scalar_1 = cutlass.Int32(0)") == 1
    assert "(token_base.iterator + " in slabs and "(valid_len.iterator + " in slabs
    # Each slab column is live below the limit (in the window's own
    # coordinates) and inside the tensor's 128 tokens; the offset shifts
    # the unit-stride token index.
    assert len(re.findall(r"% 32 < fragment_slab_scalar_1\n", slabs)) == 1
    assert len(re.findall(r"fragment_slab_scalar \+ 0 \+ .+? % 32 < 128\n", slabs)) == 1
    assert re.search(
        r"cutlass\.Int64\(fragment_slab_scalar \+ 0 \+ [^)]*% 32\) \* cutlass\.Int64\(1\)",
        slabs,
    )
    assert re.search(r"\+ 16\) \* 1\]", slabs)
    assert slabs.count(SYNC) < windows.count(SYNC)
    # Varying along the contiguous dim, the slab is walked under every
    # transport knob.
    everything = bound.to_code(_config(bound, **{SLAB: True, VEC: True, BATCH: True}))
    assert everything.count(".load()") == 5
    assert everything.count(PACKET) == 1 and everything.count("cute.copy(") == 1


def test_engine_varlen_gram_plans_three_slabs(_cpu_only: None) -> None:
    import importlib

    from helion._compiler.cute.fragment_slab_loads import plan_slab_loads

    engine = importlib.import_module("examples.linear.linear_attention_engine")
    fn = engine.chunk_fwd_A_diag_anchored_varlen_helion.fn
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    token_base, valid_len, nt = _varlen_tables((100, 64, 90), 64)
    h, d = 2, 64
    total = 254
    hnt = h * nt
    bf16, f32 = torch.bfloat16, torch.float32
    args = (
        torch.ones(total, h, d, dtype=bf16),
        torch.ones(total, h, d, dtype=bf16),
        torch.zeros(hnt, 64, d, dtype=f32),
        token_base,
        valid_len,
        torch.zeros(hnt, 64, 64, dtype=bf16),
        torch.zeros(hnt, 64, 64, dtype=bf16),
        0.125,
        True,
    )
    bound = _cpu_bind(kernel, args)
    # The engine's varlen anchored gram: four q tiers, seven k windows and
    # eleven gc windows (four of them under the chunk-size tiers' ``if``)
    # are three slabs; the tiers' captured ``base`` / ``vlen`` / ``dcols``
    # resolve to the root's loads and iota.
    assert bound.config_spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        plan = plan_slab_loads(bound.env, bound.host_function.device_ir.graphs)
    groups = {id(group): group for group in plan.values()}
    assert len(groups) == 3 and len(plan) == 22
    assert sorted(len(g.members) for g in groups.values()) == [4, 7, 11]
    code = bound.to_code(_config(bound, **{SLAB: True, VEC: True}))
    assert code.count(".load()") == 4
    assert code.count("fragment_slab_scalar") >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("vec", "batch", "threads"),
    [(False, False, 256), (True, True, 256), (True, True, 512)],
)
def test_varlen_anchored_gram_slabs_match_reference(
    vec: bool, batch: bool, threads: int
) -> None:
    # Sequences of 70, 32 and 45 tokens in chunks of 32: ragged last chunks
    # of 6 and 13 rows, and a last chunk whose rows past the end lie past
    # the tensor's end.
    args = _varlen_gram_inputs((70, 32, 45), device=DEVICE, seed=0)
    bound = _varlen_anchored_gram.bind(args)
    knobs = {
        DOT: "tf32_mma",
        PAD: True,
        "cute_fragment_unrolled_loops": True,
        "cute_fragment_threads": threads,
        SLAB: True,
        VEC: vec,
        BATCH: batch,
    }
    config = _config(bound, **knobs)
    code = bound.to_code(config)
    assert code.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    out = bound.compile_config(config)(*args)
    torch.cuda.synchronize()
    expected = _varlen_anchored_gram_reference(*args)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    # The slabs stage the same scalars the masked windows read: the result
    # is bitwise that of the plain walk.
    plain = bound.compile_config(
        _config(bound, **(knobs | {SLAB: False, VEC: False, BATCH: False}))
    )(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("vec", "batch"), [(False, False), (True, True)])
def test_varlen_contiguous_dim_slab_matches_reference(vec: bool, batch: bool) -> None:
    # Ragged last chunks of 6 and 13 tokens, and a last chunk whose tokens
    # past the sequence's end lie past the tensor's end.
    args = _varlen_contiguous_inputs((70, 32, 45), device=DEVICE, seed=0)
    bound = _varlen_contiguous_dim_windows.bind(args)
    knobs = {SLAB: True, VEC: vec, BATCH: batch}
    config = _config(bound, **knobs)
    code = bound.to_code(config)
    assert code.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    out = bound.compile_config(config)(*args)
    torch.cuda.synchronize()
    expected = _varlen_contiguous_dim_windows_reference(*args)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    # The slab stages the same scalars the masked windows read: bitwise
    # the plain walk.
    plain = bound.compile_config(
        _config(bound, **(knobs | {SLAB: False, VEC: False, BATCH: False}))
    )(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, plain)
