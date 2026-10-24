"""The tf32 fragment mma's direct lane stores and tile-row walk.

A dot whose only reader is a host store of the dot itself writes its lane
results straight into the host window (no staged result tile, no barrier
before the store walk and none after it) -- provided nothing between the dot
and the store in graph order touches host memory that may alias the window,
since the lanes write at the dot's position; ``cute_fragment_mma_rows`` makes
each warp walk whole 16-row tile rows with the row's A words hoisted across
its column tiles.  Both leave every output bitwise unchanged.  The CPU tests
pin the renders and the search surface; the GPU tests compare against the
staged twin and torch.
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
from test.test_cute_fp32_dot import _batched_dot
from test.test_cute_fp32_dot import _chunk_cumsum

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import BooleanFragment
import helion.language as hl

DOT = "cute_fp32_dot"
VEC = "cute_fragment_vector_io"
SPLIT = "cute_fragment_split_k"
ROWS = "cute_fragment_mma_rows"
THREADS = "cute_fragment_threads"
MMA = "_helion_fragment_mma.mma_tf32("
SYNC = "cute.arch.sync_threads()"
PACKET = "fragment_mma_packet"
PAIR = "num_bits_per_copy=64"

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_twice(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The cumsum's dot stored twice: a second reader keeps the staged tile."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    out2 = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        idx = hl.arange(chunk)
        ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)
        lower = ltri[None, :, :].broadcast_to([tile_b, chunk, chunk])  # pyrefly: ignore[no-matching-overload]
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(lower, values)
        out[tile_b, :, tile_d] = total
        out2[tile_b, :, tile_d] = total
    return out, out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_cast(g: torch.Tensor) -> torch.Tensor:
    """A cast between the dot and its store: not the dot itself."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.bfloat16, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        idx = hl.arange(chunk)
        ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)
        lower = ltri[None, :, :].broadcast_to([tile_b, chunk, chunk])  # pyrefly: ignore[no-matching-overload]
        values = g[tile_b, :, tile_d].to(torch.float32)
        out[tile_b, :, tile_d] = hl.dot(lower, values).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_masked(g: torch.Tensor) -> torch.Tensor:
    """A store with an extra mask keeps the staged walk."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.zeros([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        idx = hl.arange(chunk)
        ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)
        lower = ltri[None, :, :].broadcast_to([tile_b, chunk, chunk])  # pyrefly: ignore[no-matching-overload]
        values = g[tile_b, :, tile_d].to(torch.float32)
        keep = (idx < chunk - 8)[:, None]
        hl.store(
            out, [tile_b, slice(None), tile_d], hl.dot(lower, values), extra_mask=keep
        )
    return out


def _lower_triangle(tile_b: object, chunk: int) -> torch.Tensor:
    idx = hl.arange(chunk)
    ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)
    return ltri[None, :, :].broadcast_to([tile_b, chunk, chunk])  # pyrefly: ignore[no-matching-overload]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_store_between(g: torch.Tensor) -> torch.Tensor:
    """The window is written again between the dot and its store: the later
    store must win, so the lanes may not write ahead of it."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(_lower_triangle(tile_b, chunk), values)
        out[tile_b, :, tile_d] = values
        out[tile_b, :, tile_d] = total
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_load_between(g: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The window's old contents are read between the dot and its store."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out2 = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(_lower_triangle(tile_b, chunk), values)
        previous = out[tile_b, :, tile_d]
        out[tile_b, :, tile_d] = total
        out2[tile_b, :, tile_d] = previous
    return out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_view_store_between(g: torch.Tensor) -> torch.Tensor:
    """A store through another view of the window's storage in between."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    alias = out.view(batches, chunk, width)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(_lower_triangle(tile_b, chunk), values)
        alias[tile_b, :, tile_d] = values
        out[tile_b, :, tile_d] = total
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_loop_between(
    g: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A nested loop between the dot and its store is opaque: staged."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    out2 = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(_lower_triangle(tile_b, chunk), values)
        for tile_k in hl.tile(chunk):
            out2[tile_b, tile_k, tile_d] = other[tile_b, tile_k, tile_d] * 2.0
        out[tile_b, :, tile_d] = total
    return out, out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_fresh_store_between(
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A store into a fresh allocation between the dot and its store cannot
    alias the window: the lanes still write directly."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    out2 = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(_lower_triangle(tile_b, chunk), values)
        out2[tile_b, :, tile_d] = values * 3.0
        out[tile_b, :, tile_d] = total
    return out, out2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum_input_load_between(
    g: torch.Tensor, other: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """A load of one input between the dot and its store into another input:
    direct only under the bound key's proof that the two are disjoint."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out2 = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        values = g[tile_b, :, tile_d].to(torch.float32)
        total = hl.dot(_lower_triangle(tile_b, chunk), values)
        extra = other[tile_b, :, tile_d]
        out[tile_b, :, tile_d] = total
        out2[tile_b, :, tile_d] = extra * 2.0
    return out2


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


def _lane_loop(code: str) -> str:
    return code.split("for fragment_mma_tile")[1].split(SYNC)[0]


def test_the_stored_dot_writes_from_the_lanes(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    simt = bound.to_code(_config(bound))
    tf32 = bound.to_code(_config(bound, **{DOT: "tf32_mma"}))
    vec = bound.to_code(_config(bound, **{DOT: "tf32_mma", VEC: True}))
    for code in (tf32, vec):
        # No staged result tile: the two buffers are the bf16 chunk tile and
        # its padded fp32 operand copy.  The lane loop stores the four results
        # of a lane straight into ``out``; no element walk reads a buffer
        # back and no barrier follows the stores.
        assert code.count(MMA) == 1
        assert len(re.findall(r"allocate_tensor\(", code)) == 2
        assert not re.search(
            r"fragment_buffer\w*\[[^\n]*\] = [\w.]+\(fragment_mma_c", code
        )
        loop = _lane_loop(code)
        assert "out.iterator" in loop
        assert "fragment_mma_c)" in loop and "fragment_mma_c_3)" in loop
        assert "fragment_index" not in code.split("for fragment_mma_tile")[1]
        assert code.rstrip().endswith("return out")
    assert simt.count(SYNC) - tf32.count(SYNC) == 2
    # Without the packet proof each result is one guarded scalar store; with
    # ``cute_fragment_vector_io`` a lane's two adjacent columns leave as one
    # aligned 8-byte pair.
    assert _lane_loop(tf32).count(".store(") == 4 and PACKET not in tf32
    assert ".store(" not in vec
    assert _lane_loop(vec).count(PACKET) >= 2 and _lane_loop(vec).count(PAIR) == 2
    assert ".align(8)" in _lane_loop(vec)


def test_unproved_packets_store_scalars(_cpu_only: None) -> None:
    # Ten output columns: no 16-byte packet divides the output rows (the
    # fp32 operand loads still move as packets), so the pair proof fails and
    # each lane result leaves as a guarded scalar store -- still straight
    # from the lanes, still without a staged tile.
    x = torch.ones(2, 24, 20)
    y = torch.ones(2, 20, 10)
    z = torch.ones(2, 24, 10)
    bound = _cpu_bind(_batched_dot, (x, y, z))
    code = bound.to_code(_config(bound, **{DOT: "tf32_mma", VEC: True}))
    loop = _lane_loop(code)
    assert PACKET not in loop and PAIR not in loop
    assert loop.count(".store(") == 4
    assert "< 10" in loop and "< 24" in loop
    assert "num_bits_per_copy=128" in code
    assert not re.search(r"fragment_buffer\w*\[[^\n]*\] = [\w.]+\(fragment_mma_c", code)


@pytest.mark.parametrize(
    "kernel", [_chunk_cumsum_twice, _chunk_cumsum_cast, _chunk_cumsum_masked]
)
def test_other_readers_keep_the_staged_tile(_cpu_only: None, kernel: object) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(kernel, (g,))  # type: ignore[arg-type]
    code = bound.to_code(_config(bound, **{DOT: "tf32_mma", VEC: True}))
    assert code.count(MMA) == 1
    assert PACKET not in code and PAIR not in code
    # The result tile is staged and read back by the store walk.
    assert len(re.findall(r"allocate_tensor\(", code)) == 3
    assert re.search(r"fragment_buffer\w*\[[^\n]*\] = [\w.]+\(fragment_mma_c\)", code)


def _assert_staged(code: str) -> None:
    # One mma site under the tile walk, one per column tile under the rows.
    assert code.count(MMA) >= 1
    assert PACKET not in code and PAIR not in code
    assert len(re.findall(r"allocate_tensor\(", code)) == 3
    assert re.search(r"fragment_buffer\w*\[[^\n]*\] = [\w.]+\(fragment_mma_c\)", code)


def _assert_direct(code: str) -> None:
    assert code.count(MMA) == 1
    assert len(re.findall(r"allocate_tensor\(", code)) == 2
    assert not re.search(r"fragment_buffer\w*\[[^\n]*\] = [\w.]+\(fragment_mma_c", code)
    loop = _lane_loop(code)
    assert "out.iterator" in loop and loop.count(PAIR) == 2


@pytest.mark.parametrize(
    ("kernel", "extra_args"),
    [
        (_chunk_cumsum_store_between, ()),
        (_chunk_cumsum_load_between, (torch.ones(4, 64, 128),)),
        (_chunk_cumsum_view_store_between, ()),
        (_chunk_cumsum_loop_between, (torch.ones(4, 64, 128),)),
    ],
)
def test_accesses_between_the_dot_and_its_store_keep_the_staged_tile(
    _cpu_only: None, kernel: object, extra_args: tuple[torch.Tensor, ...]
) -> None:
    # The lanes would write the window at the dot's position, ahead of a
    # store or load of that window (or of a view of its storage, or of a
    # nested loop that may do either) the program places before the store.
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(kernel, (g, *extra_args))  # type: ignore[arg-type]
    for knobs in (
        {DOT: "tf32_mma", VEC: True},
        {DOT: "tf32_mma", VEC: True, ROWS: True},
    ):
        _assert_staged(bound.to_code(_config(bound, **knobs)))


def test_disjoint_accesses_between_the_dot_and_its_store_stay_direct(
    _cpu_only: None,
) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    knobs = {DOT: "tf32_mma", VEC: True}
    # A store into a fresh allocation cannot alias the window.
    bound = _cpu_bind(_chunk_cumsum_fresh_store_between, (g,))
    _assert_direct(bound.to_code(_config(bound, **knobs)))
    # Two inputs are disjoint only by the bound key's alias matrix: distinct
    # tensors prove it; the same tensor (or an overlapping view of it) in
    # both slots keeps the staged tile.
    other = torch.ones(4, 64, 128)
    out = torch.ones(4, 64, 128)
    bound = _cpu_bind(_chunk_cumsum_input_load_between, (g, other, out))
    _assert_direct(bound.to_code(_config(bound, **knobs)))
    bound = _cpu_bind(_chunk_cumsum_input_load_between, (g, out, out))
    _assert_staged(bound.to_code(_config(bound, **knobs)))
    storage = torch.ones(5, 64, 128)
    bound = _cpu_bind(_chunk_cumsum_input_load_between, (g, storage[1:], storage[:4]))
    _assert_staged(bound.to_code(_config(bound, **knobs)))


def test_split_k_keeps_the_staged_partials(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 16, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    code = bound.to_code(
        _config(bound, **{DOT: "tf32_mma", VEC: True, SPLIT: True, THREADS: 512})
    )
    assert "fragment_mma_split" in code
    assert PACKET not in code and PAIR not in code
    rows = bound.to_code(
        _config(
            bound, **{DOT: "tf32_mma", VEC: True, SPLIT: True, ROWS: True, THREADS: 512}
        )
    )
    # A K split keeps its (tile, split) unit walk: the row knob is inert.
    assert rows == code


def test_mma_rows_joins_the_search_surface(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    spec = bound.config_spec
    assert spec.cute_fragment_mma_rows_root_ids
    assert isinstance(spec._flat_fields()[ROWS], BooleanFragment)
    assert ROWS not in spec.default_config()
    with pytest.raises(exc.InvalidConfig, match="requires cute_fp32_dot"):
        spec.normalize(_config(bound, **{ROWS: True}).config)
    generation = spec.create_config_generation()
    _, effective = generation.strict_config_pair(
        _config(bound, **{DOT: "tf32_mma", ROWS: True})
    )
    assert effective[ROWS] is True
    # The witness pins the walk's intended transport: tf32 lanes over packet
    # windows, an unrolled load walk and padded rows (all exact knobs).
    (group,) = [g for g in spec.compiler_coverage_groups if g.key == ROWS]
    (witness,) = group.witnesses
    assert witness.value is True and group.deferred
    carrier = witness.carrier.config
    assert carrier[DOT] == "tf32_mma"
    assert carrier[VEC] is True
    assert carrier["cute_fragment_unrolled_loops"] is True
    assert carrier["cute_fragment_padded_rows"] is True
    # The supplemental witness widens the innermost tile (the chunk width,
    # 128 here) so the search sees a row with column tiles to reuse across.
    (wide,) = group.supplemental_witnesses
    assert wide.value is True
    assert wide.carrier.config["block_sizes"][-1] == 128 > carrier["block_sizes"][-1]
    assert {k: v for k, v in wide.carrier.config.items() if k != "block_sizes"} == {
        k: v for k, v in carrier.items() if k != "block_sizes"
    }
    assert any(
        group.key == ROWS and any(w.value is True for w in group.witnesses)
        for group in spec.compiler_coverage_groups
    )


def test_mma_rows_hoists_the_a_words(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    tiles = bound.to_code(_config(bound, **{DOT: "tf32_mma", VEC: True}))
    rows = bound.to_code(_config(bound, **{DOT: "tf32_mma", VEC: True, ROWS: True}))
    assert "fragment_mma_tile_row" not in tiles and "fragment_mma_step" in tiles
    assert "fragment_mma_tile_row" in rows and "fragment_mma_step" not in rows
    # The tile walk evaluates the four lower-triangular compares of a step
    # inside its unrolled K loop, for every tile; the row walk forms all
    # 8 x 4 words once per tile row, ahead of the column-tile loop, which
    # reads only the B words from the staged chunk tile.
    assert tiles.count("operator.ge(") == 4
    head, column_loop = re.split(r"for fragment_mma_col_tile in range\(\d+\):", rows)
    assert head.count("operator.ge(") == 32
    assert "operator.ge(" not in column_loop
    assert column_loop.split(SYNC)[0].count(MMA) == 8
    assert "fragment_buffer" in column_loop.split(SYNC)[0]
    # Same barriers, buffers and direct stores as the tile walk.
    assert rows.count(SYNC) == tiles.count(SYNC)
    assert re.findall(r"allocate_tensor\(.*\)", rows) == re.findall(
        r"allocate_tensor\(.*\)", tiles
    )
    assert column_loop.count(PAIR) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("threads", "vec"), [(32, True), (128, False), (256, True)])
def test_direct_stores_match_the_staged_tile(threads: int, vec: bool) -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    knobs = {DOT: "tf32_mma", VEC: vec, THREADS: threads}
    direct = _chunk_cumsum.bind((g,))
    code = direct.to_code(_config(direct, **knobs))
    assert (PAIR in code) == vec and "fragment_index" not in _lane_loop(code)
    out = direct.compile_config(_config(direct, **knobs))(g)
    staged = _chunk_cumsum_twice.bind((g,))
    assert PAIR not in staged.to_code(_config(staged, **knobs))
    first, second = staged.compile_config(_config(staged, **knobs))(g)
    torch.cuda.synchronize()
    assert torch.equal(out, first) and torch.equal(first, second)
    torch.testing.assert_close(out, g.float().cumsum(1), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [False, True])
def test_accesses_between_the_dot_and_its_store_keep_program_order(
    rows: bool,
) -> None:
    torch.manual_seed(0)
    g = torch.randn(4, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    knobs = {DOT: "tf32_mma", VEC: True, ROWS: rows}
    expected = g.float().cumsum(1)

    def run(kernel: object, *args: torch.Tensor) -> object:
        bound = kernel.bind(args)  # type: ignore[attr-defined]
        result = bound.compile_config(_config(bound, **knobs))(*args)
        torch.cuda.synchronize()
        return result

    # The later store of the dot wins over the store of ``values`` before it.
    out = run(_chunk_cumsum_store_between, g)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    # The read between the dot and its store sees the window's old contents.
    out = torch.full((4, 64, 128), 7.0, device=DEVICE)
    out2 = run(_chunk_cumsum_load_between, g, out)
    assert torch.equal(out2, torch.full_like(out2, 7.0))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    # The same through a second input aliasing the window ...
    out = torch.full((4, 64, 128), 7.0, device=DEVICE)
    out2 = run(_chunk_cumsum_input_load_between, g, out, out)
    assert torch.equal(out2, torch.full_like(out2, 14.0))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    # ... and with disjoint inputs the lanes write directly (bitwise the
    # staged result) while the other input is read untouched.
    other = torch.randn(4, 64, 128, device=DEVICE)
    out = torch.empty(4, 64, 128, device=DEVICE)
    bound = _chunk_cumsum_input_load_between.bind((g, other, out))
    assert PAIR in bound.to_code(_config(bound, **knobs))
    out2 = run(_chunk_cumsum_input_load_between, g, other, out)
    assert torch.equal(out2, other * 2.0)
    staged = run(_chunk_cumsum_store_between, g)
    assert torch.equal(out, staged)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scalar_direct_stores_match_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 24, 20, device=DEVICE)
    y = torch.randn(3, 20, 10, device=DEVICE)
    z = torch.randn(3, 24, 10, device=DEVICE)
    bound = _batched_dot.bind((x, y, z))
    knobs = {DOT: "tf32_mma", VEC: True}
    code = bound.to_code(_config(bound, **knobs))
    assert PAIR not in code and _lane_loop(code).count(".store(") == 4
    out = bound.compile_config(_config(bound, **knobs))(x, y, z)
    rows = bound.compile_config(_config(bound, **(knobs | {ROWS: True})))(x, y, z)
    plain = bound.compile_config(_config(bound, **{DOT: "tf32_mma"}))(x, y, z)
    torch.cuda.synchronize()
    assert torch.equal(out, rows) and torch.equal(out, plain)
    torch.testing.assert_close(out, x @ y + z, rtol=1e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("threads", [32, 128])
def test_row_walk_matches_the_tile_walk(threads: int) -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = _chunk_cumsum.bind((g,))
    knobs = {DOT: "tf32_mma", VEC: True, THREADS: threads}
    tiles = bound.compile_config(_config(bound, **knobs))(g)
    rows = bound.compile_config(_config(bound, **(knobs | {ROWS: True})))(g)
    torch.cuda.synchronize()
    assert torch.equal(tiles, rows)
    torch.testing.assert_close(rows, g.float().cumsum(1), rtol=1e-5, atol=1e-5)
    # Partial tiles on every axis, a seeded accumulator and the staged twin.
    x = torch.randn(3, 24, 20, device=DEVICE)
    y = torch.randn(3, 20, 12, device=DEVICE)
    z = torch.randn(3, 24, 12, device=DEVICE)
    bound = _batched_dot.bind((x, y, z))
    tiles = bound.compile_config(_config(bound, **{DOT: "tf32_mma", THREADS: threads}))(
        x, y, z
    )
    rows = bound.compile_config(
        _config(bound, **{DOT: "tf32_mma", THREADS: threads, ROWS: True})
    )(x, y, z)
    torch.cuda.synchronize()
    assert torch.equal(tiles, rows)
    torch.testing.assert_close(rows, x @ y + z, rtol=1e-2, atol=2e-2)
