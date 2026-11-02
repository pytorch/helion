"""Staging of fragment tiles for the anchored chunk grams.

``cute_fragment_padded_rows`` (bank-conflict-free row strides),
``cute_fragment_unrolled_loops`` (constexpr element walks), the in-place
read of a whole-tile tf32 ``mma`` operand from staged leaves, the readonly
proof for loads beside a freshly allocated output, and the fp32-dot seed by
total multiply-adds.  The CPU tests pin the search surface and the generated
code; the GPU tests check the numerics against torch.
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

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import BooleanFragment
import helion.language as hl

PAD = "cute_fragment_padded_rows"
UNROLL = "cute_fragment_unrolled_loops"
PACKETS = "cute_fragment_packet_loads"
DOT = "cute_fp32_dot"
MMA = "_helion_fragment_mma.mma_tf32("

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _anchored_gram(
    q: torch.Tensor, k: torch.Tensor, gc: torch.Tensor, scale: float
) -> torch.Tensor:
    """Two 16-row tiers of the anchored intra-chunk gram (chunk 32): fp32
    operands that are pointwise products of staged loads, a transposed
    column operand, iota-window stores into a zero-filled output."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    A = torch.zeros([BHN, C, C], dtype=q.dtype, device=q.device)
    for tile_b in hl.tile(BHN, block_size=1):
        rows0 = hl.arange(16)
        cols0 = hl.arange(16)
        n0 = gc[tile_b, 0, :]
        e0 = torch.exp2(gc[tile_b, rows0, :] - n0[:, None, :])
        kg0 = k[tile_b, cols0, :].float() * torch.exp2(
            n0[:, None, :] - gc[tile_b, cols0, :]
        )
        a0 = hl.dot(q[tile_b, rows0, :].float() * e0, kg0.transpose(-2, -1)) * scale
        causal0 = (rows0[:, None] >= cols0[None, :])[None, :, :]
        A[tile_b, rows0, cols0] = a0 * causal0.to(a0.dtype)

        rows1 = 16 + hl.arange(16)
        cols1 = hl.arange(32)
        n1 = gc[tile_b, 16, :]
        e1 = torch.exp2(gc[tile_b, rows1, :] - n1[:, None, :])
        kg1 = k[tile_b, cols1, :].float() * torch.exp2(
            n1[:, None, :] - gc[tile_b, cols1, :]
        )
        a1 = hl.dot(q[tile_b, rows1, :].float() * e1, kg1.transpose(-2, -1)) * scale
        causal1 = (rows1[:, None] >= cols1[None, :])[None, :, :]
        A[tile_b, rows1, cols1] = a1 * causal1.to(a1.dtype)
    return A


def _anchored_gram_reference(
    q: torch.Tensor, k: torch.Tensor, gc: torch.Tensor, scale: float
) -> torch.Tensor:
    A = torch.zeros(q.shape[0], 32, 32, dtype=torch.float32, device=q.device)
    for tier in range(2):
        rows = slice(16 * tier, 16 * tier + 16)
        cols = slice(0, 16 * tier + 16)
        n = gc[:, 16 * tier, :]
        e = torch.exp2(gc[:, rows, :] - n[:, None, :])
        kg = k[:, cols, :].float() * torch.exp2(n[:, None, :] - gc[:, cols, :])
        r = torch.arange(16 * tier, 16 * tier + 16, device=q.device)
        c = torch.arange(0, 16 * tier + 16, device=q.device)
        a = (q[:, rows, :].float() * e) @ kg.transpose(-2, -1) * scale
        A[:, rows, cols] = a * (r[:, None] >= c[None, :]).float()
    return A


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sliced_gram_adjoint(
    k: torch.Tensor, gc: torch.Tensor, dA: torch.Tensor
) -> torch.Tensor:
    """One block pair of the gram adjoint: 16-row sub-block windows over a
    tiled channel axis, two tiny fp32 dots (one over a transposed load) and
    read-modify-write stores into a zero-filled output."""
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    K = k.size(2)
    dk = torch.zeros([BHN, C, K], dtype=torch.float32, device=k.device)
    block_k = hl.register_block_size(K)
    for tile_b, tile_k in hl.tile([BHN, K], block_size=[1, block_k]):
        rows = 16 + hl.arange(16)
        cols = hl.arange(16)
        gn = gc[tile_b, 16, tile_k]
        k_r = k[tile_b, rows, tile_k].to(torch.float32)
        k_c = k[tile_b, cols, tile_k].to(torch.float32)
        erow = torch.exp2(gc[tile_b, rows, tile_k] - gn[:, None, :])
        ecol = torch.exp2(gn[:, None, :] - gc[tile_b, cols, tile_k])
        kgr = k_r * erow
        kg = k_c * ecol
        dA_b = dA[tile_b, rows, cols]
        d_kgr = hl.dot(dA_b, kg)
        d_kg = hl.dot(dA_b.transpose(-2, -1), kgr)
        dk[tile_b, rows, tile_k] += d_kgr * erow
        dk[tile_b, cols, tile_k] += d_kg * ecol
    return dk


def _sliced_gram_adjoint_reference(
    k: torch.Tensor, gc: torch.Tensor, dA: torch.Tensor
) -> torch.Tensor:
    gn = gc[:, 16, :]
    erow = torch.exp2(gc[:, 16:32, :] - gn[:, None, :])
    ecol = torch.exp2(gn[:, None, :] - gc[:, 0:16, :])
    kgr = k[:, 16:32, :].float() * erow
    kg = k[:, 0:16, :].float() * ecol
    dA_b = dA[:, 16:32, 0:16]
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dk[:, 16:32, :] += (dA_b @ kg) * erow
    dk[:, 0:16, :] += (dA_b.transpose(-2, -1) @ kgr) * ecol
    return dk


@pytest.fixture
def _cpu_only():
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        yield


def _gram_inputs(device: torch.device | str = "cpu", batches: int = 4):
    q = torch.ones(batches, 32, 128, dtype=torch.bfloat16, device=device)
    k = torch.ones(batches, 32, 128, dtype=torch.bfloat16, device=device)
    gc = torch.zeros(batches, 32, 128, dtype=torch.float32, device=device)
    return q, k, gc, 0.125


def _config(bound: object, **overrides: object) -> helion.Config:
    spec = bound.config_spec  # type: ignore[attr-defined]
    return helion.Config.from_dict(dict(spec.default_config()) | overrides)


def test_staging_knobs_join_the_search_surface(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    spec = bound.config_spec
    assert spec.cute_fragment_padded_rows_root_ids
    assert spec.cute_fragment_unrolled_loops_root_ids
    fields = spec._flat_fields()
    assert isinstance(fields[PAD], BooleanFragment)
    assert isinstance(fields[UNROLL], BooleanFragment)
    default = spec.default_config()
    assert PAD not in default and UNROLL not in default
    for key in (PAD, UNROLL):
        off = _config(bound, **{key: False})
        spec.normalize(off.config)
        assert off == default
        # Each knob is a coverage witness of its own.
        assert any(
            group.key == key and any(w.value is True for w in group.witnesses)
            for group in spec.compiler_coverage_groups
        )
        with pytest.raises(exc.InvalidConfig, match="complete fragment root"):
            spec.normalize(_config(bound, **{key: 1}).config)


def test_loads_beside_a_fresh_zero_filled_output_are_readonly(
    _cpu_only: None,
) -> None:
    # The gram's outputs are ``torch.zeros`` the host function allocates on
    # every call: a storage no input can alias, so the loads beside its
    # stores are readonly and the packet-load knob is offered.
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    assert bound.config_spec.cute_fragment_packet_load_root_ids
    assert PACKETS in bound.config_spec._flat_fields()
    code = bound.to_code(_config(bound, **{PACKETS: True}))
    assert "num_bits_per_copy=64" in code  # bf16 packets of q / k


def test_padded_rows_stride_every_staged_tile(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    dense = bound.to_code(_config(bound))
    padded = bound.to_code(_config(bound, **{PAD: True}))
    # [rows, 128] fp32 tiles stride 132 words, [rows, 128] bf16 tiles 136
    # elements (68 words): both put consecutive rows four banks apart.
    assert "* 132 +" not in dense and "* 136 +" not in dense
    assert "* 132 +" in padded and "* 136 +" in padded
    sizes = [int(m) for m in re.findall(r"cute\.make_layout\(\((\d+),\)\)", padded)]
    assert any(size % 132 == 0 for size in sizes)  # rows x 132 fp32
    assert any(size % 136 == 0 for size in sizes)  # rows x 136 bf16
    # The 128-wide anchor row is a single row: nothing to conflict with, so
    # it stays dense; the stores and barriers are unchanged.
    assert "[0 * 132 +" not in padded and "[0 * 128 +" in padded
    assert dense.count(".store(") == padded.count(".store(")
    assert dense.count("sync_threads") == padded.count("sync_threads")


def test_unrolled_loops_walk_slots_as_constexpr(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    rolled = bound.to_code(_config(bound))
    unrolled = bound.to_code(_config(bound, **{UNROLL: True}))
    assert "fragment_slot in cutlass.range_constexpr(" not in rolled
    assert re.search(
        r"for fragment_index(_\d+)? in range\(fragment_thread, 2048, 128\)", rolled
    )
    # 2048 elements over 128 threads: 16 slots, no remainder guard.
    assert "cutlass.range_constexpr(16):" in unrolled
    assert re.search(
        r"fragment_index(_\d+)? = fragment_thread \+ fragment_slot(_\d+)? \* 128",
        unrolled,
    )
    assert not re.search(r"if fragment_index(_\d+)? < 2048", unrolled)
    # The anchor row (128 elements) is one slot; the 32 x 128 column tile at
    # 128 threads is 32 slots, past the unroll bound, so it stays rolled.
    assert "cutlass.range_constexpr(1):" in unrolled
    assert "in range(fragment_thread, 4096, 128)" in unrolled
    assert rolled.count("sync_threads") == unrolled.count("sync_threads")


def test_whole_tile_transposed_operand_is_read_in_place(_cpu_only: None) -> None:
    bound = _cpu_bind(_anchored_gram, _gram_inputs())
    simt = bound.to_code(_config(bound))
    tf32 = bound.to_code(_config(bound, **{DOT: "tf32_mma", PAD: True}))
    assert tf32.count(MMA) == 2
    # Each dot's [128, cols] column operand has one 16-row tile along M, so
    # its lanes evaluate the recipe (the exp2 decay) in place: the operand
    # copy and its barrier are gone, one per dot.
    assert simt.count("sync_threads") - tf32.count("sync_threads") == 2
    mma_loops = tf32.split("for fragment_mma_tile")[1:]
    assert len(mma_loops) == 2
    for body in mma_loops:
        loop = body.split("cute.arch.sync_threads()")[0]
        assert "cute.math.exp2(" in loop
        # The transposed read of the padded k tile: row stride 136 bf16.
        assert "* 136 +" in loop
    # The 16-row operand is re-read per 8-column tile: still staged (padded).
    assert "cute.make_layout((2112,))" in tf32


def test_a_partial_tile_keeps_the_staged_operands(_cpu_only: None) -> None:
    x = torch.ones(2, 24, 20)
    y = torch.ones(2, 20, 12)
    z = torch.ones(2, 24, 12)
    from test.test_cute_fp32_dot import _batched_dot

    bound = _cpu_bind(_batched_dot, (x, y, z))
    simt = bound.to_code(_config(bound))
    tf32 = bound.to_code(_config(bound, **{DOT: "tf32_mma"}))
    # Both operands keep their staged (padded) copies, one barrier each
    # beyond the simt path's direct reads of the loaded tiles: the lanes
    # never evaluate a recipe past the end of a partial tile.  (The result
    # itself is stored from the lanes -- test_cute_fragment_direct_store --
    # which takes one barrier back.)
    assert MMA in tf32
    assert "fragment_mma_c)" in tf32.split("for fragment_mma_tile")[1]
    assert tf32.count("sync_threads") - simt.count("sync_threads") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dot", "threads", "pad", "unroll"),
    [
        ("simt", 128, True, True),
        ("tf32_mma", 128, True, True),
        ("tf32_mma", 512, True, True),
        ("tf32_mma", 256, True, False),
        ("tf32_mma", 256, False, True),
    ],
)
def test_anchored_gram_matches_reference(
    dot: str, threads: int, pad: bool, unroll: bool
) -> None:
    torch.manual_seed(0)
    q = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(64, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    gc = (-torch.rand(64, 32, 128, device=DEVICE) * 0.1).cumsum(1)
    scale = 0.125
    bound = _anchored_gram.bind((q, k, gc, scale))
    config = _config(
        bound,
        **{DOT: dot, PAD: pad, UNROLL: unroll, "cute_fragment_threads": threads},
    )
    code = bound.to_code(config)
    assert (MMA in code) == (dot == "tf32_mma")
    # The bf16 k tiles stride 136 only with padded rows (the tf32 path pads
    # its staged 16-row operand to 132 words on its own).
    assert ("* 136 +" in code) == pad
    assert ("fragment_slot in cutlass.range_constexpr(" in code) == unroll
    out = bound.compile_config(config)(q, k, gc, scale)
    torch.cuda.synchronize()
    expected = _anchored_gram_reference(q, k, gc, scale)
    # bf16 output rounding dominates; tf32 adds ~1e-3 relative per product.
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dot", "block", "threads"),
    [("simt", 32, 128), ("tf32_mma", 32, 128), ("tf32_mma", 64, 256)],
)
def test_sliced_gram_adjoint_matches_reference(
    dot: str, block: int, threads: int
) -> None:
    torch.manual_seed(0)
    k = torch.randn(16, 32, 128, device=DEVICE, dtype=torch.bfloat16)
    gc = (-torch.rand(16, 32, 128, device=DEVICE) * 0.1).cumsum(1)
    dA = torch.randn(16, 32, 32, device=DEVICE) * 0.1
    bound = _sliced_gram_adjoint.bind((k, gc, dA))
    config = _config(
        bound,
        **{
            DOT: dot,
            PAD: True,
            UNROLL: True,
            "block_sizes": [block],
            "cute_fragment_threads": threads,
        },
    )
    code = bound.to_code(config)
    assert (MMA in code) == (dot == "tf32_mma")
    assert "fragment_slot in cutlass.range_constexpr(" in code
    out = bound.compile_config(config)(k, gc, dA)
    torch.cuda.synchronize()
    expected = _sliced_gram_adjoint_reference(k, gc, dA)
    tolerance = 1e-4 if dot == "simt" else 2e-2
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance)
