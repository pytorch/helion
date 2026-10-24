"""One 16-byte packet per thread on a persistent stride-1 slice (rope).

A pointwise kernel whose stride-1 dim is a bounded slice (``q[tile_b, :,
tile_t, :half]`` / ``half:``) addresses it with a persistent reduction block,
not a tile, so the tile vector hoist never saw it: every access was a 2-byte
scalar.  With ``num_threads = extent / V`` and ``cute_vector_widths = V`` on
that block the persistent lane is one vector per thread; the bounded slice
is recognised as the lane axis at a V-aligned offset, ``cute_vloop_sink``
interchanges the constexpr V-loop into the lane loops of the other
full-slice axes (the heads), the branch-local persistent vectorizer widens
every load and store inside each sunk V-loop, and two sibling full-slice
axes with the same explicit thread count share one thread axis instead of
repeating each other's work.

CPU code generation plus one GPU correctness check.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True)
def rope_like(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``examples/rope.py``'s forward: rank-4 [B, H, T, D] with the head axis a
    full slice and the last axis split in two static halves."""
    batch, _, seq_len, head_dim = q.size()
    half_dim = head_dim // 2
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    cos = cos.expand(batch, seq_len, head_dim).unsqueeze(1)
    sin = sin.expand(batch, seq_len, head_dim).unsqueeze(1)
    for tile_b, tile_t in hl.tile([batch, seq_len]):
        cos_first = cos[tile_b, :, tile_t, :half_dim].float()
        cos_second = cos[tile_b, :, tile_t, half_dim:].float()
        sin_first = sin[tile_b, :, tile_t, :half_dim].float()
        sin_second = sin[tile_b, :, tile_t, half_dim:].float()
        q_first = q[tile_b, :, tile_t, :half_dim].float()
        q_second = q[tile_b, :, tile_t, half_dim:].float()
        q_out[tile_b, :, tile_t, :half_dim] = (
            (q_first * cos_first).to(q.dtype) - (q_second * sin_first).to(q.dtype)
        ).to(q.dtype)
        q_out[tile_b, :, tile_t, half_dim:] = (
            (q_second * cos_second).to(q.dtype) + (q_first * sin_second).to(q.dtype)
        ).to(q.dtype)
        k_first = k[tile_b, :, tile_t, :half_dim].float()
        k_second = k[tile_b, :, tile_t, half_dim:].float()
        k_out[tile_b, :, tile_t, :half_dim] = (
            (k_first * cos_first).to(k.dtype) - (k_second * sin_first).to(k.dtype)
        ).to(k.dtype)
        k_out[tile_b, :, tile_t, half_dim:] = (
            (k_second * cos_second).to(k.dtype) + (k_first * sin_second).to(k.dtype)
        ).to(k.dtype)
    return q_out, k_out


@helion.kernel(backend="cute", static_shapes=True)
def scale_rows(x: torch.Tensor) -> torch.Tensor:
    """Two full-slice axes of ONE value: they need the cross product of their
    coordinates and may not share a thread axis."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :, :] = x[tile, :, :] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def batched_row_softmax(x: torch.Tensor) -> torch.Tensor:
    """``examples/batch_softmax.py``: two tile axes index the rows."""
    b, m, n = x.size()
    out = torch.empty_like(x)
    for tile_b, tile_m in hl.tile([b, m]):
        values = x[tile_b, tile_m, :]
        x_max = torch.amax(values, dim=2)
        exp_vals = torch.exp(values - x_max[:, :, None])
        sum_exp = torch.sum(exp_vals, dim=2)
        out[tile_b, tile_m, :] = exp_vals / sum_exp[:, :, None]
    return out


def _rope_args(
    batch: int = 2, q_heads: int = 8, k_heads: int = 4, seq: int = 64, dim: int = 128
) -> tuple[torch.Tensor, ...]:
    q = torch.empty(batch, q_heads, seq, dim, dtype=torch.bfloat16)
    k = torch.empty(batch, k_heads, seq, dim, dtype=torch.bfloat16)
    cos = torch.empty(batch, seq, dim, dtype=torch.bfloat16)
    sin = torch.empty(batch, seq, dim, dtype=torch.bfloat16)
    return q, k, cos, sin


def _rope_reference(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# Block ids: 0 batch tile, 1 seq tile, 2 half dim (64), 3 q heads, 4 k heads.
def _config(**overrides: object) -> helion.Config:
    config: dict[str, object] = {
        "block_sizes": [1, 16],
        "num_threads": [0, 0, 8, 4, 4],
        "cute_vector_widths": [8, 1, 1, 1, 1],
        "reduction_loops": [None],
        "cute_vloop_sink": True,
    }
    config.update(overrides)
    return helion.Config(**config)


def _cpu_code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **cfg: object
) -> str:
    bound = _cpu_bind(kernel, args)
    return bound.to_code(bound.config_spec.normalized_config(_config(**cfg)))


def _loops(code: str, target: str) -> list[ast.For]:
    return [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == target
    ]


def test_half_slices_vectorize_through_the_sunk_persistent_lane(
    cpu_only: None,
) -> None:
    code = _cpu_code(rope_like, _rope_args())
    # Every load is one 16-byte packet (4 cos/sin, 2 q, 2 k) and every store
    # one 16-byte flush; nothing is left scalar.
    assert code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 8
    assert code.count("_cute_store_u16_vec(") == 4
    assert ").load()" not in code
    assert ").store(" not in code
    # The second-half packets sit 64 elements past the lane base.
    assert re.search(
        r"_persistent_branch_vec_5 = cute\.arch\.load\(q\.iterator \+ .*64 \+ ", code
    )
    assert re.search(r"_cute_store_u16_vec\(q_out\.iterator \+ .*64 \+ ", code)
    # The V-loop was interchanged into the head lane loop: one instance for
    # the cos/sin prologue, one inside the q head loop, one for the k heads;
    # the cos/sin values cross into them as 8-element register fragments.
    assert len(_loops(code, "synthetic_lane_2")) == 3
    assert code.count("cute.make_rmem_tensor(8, cutlass.Float32)") == 4
    (head_loop,) = _loops(code, "synthetic_lane_3")
    assert ast.unparse(head_loop.iter) == "range(2)"
    assert len(_loops(ast.unparse(head_loop), "synthetic_lane_2")) == 1
    # q heads (8) and k heads (4) share thread axis 1 with 4 threads each: two
    # q heads per thread, one k head, and no repeated k traffic.
    assert "indices_3 = cutlass.Int32(cute.arch.thread_idx()[1]) + " in code
    assert "indices_4 = cutlass.Int32(cute.arch.thread_idx()[1])\n" in code
    assert "block=(8, 4, 16)" in code


def test_lane_unroll_issues_every_head_packet_first(cpu_only: None) -> None:
    code = _cpu_code(rope_like, _rope_args(), cute_lane_unroll=4)
    # The sinking pass left the head loop rolled (its loads were persistent
    # markers); once they are packets the loads-first unroll takes it: a
    # constexpr load phase collects both q packets of every head before the
    # compute phase consumes them.
    loops = _loops(code, "synthetic_lane_3")
    assert len(loops) == 2
    assert all(ast.unparse(loop.iter) == "cutlass.range_constexpr(2)" for loop in loops)
    assert code.count("_lane_unroll_loads_0.append(") == 1
    assert code.count("_lane_unroll_loads_1.append(") == 1
    assert "_persistent_branch_vec_4 = _lane_unroll_loads_0[synthetic_lane_3]" in code
    assert code.count("_cute_store_u16_vec(") == 4
    assert ").load()" not in code


def test_lane_unroll_saturates_at_sixteen_live_packets(cpu_only: None) -> None:
    # One thread per q head pair (num_threads 1 on both head axes: 8 q heads
    # per thread, two packets each); asking for 16 copies holds 16 packets at
    # 8 copies, so the unroll saturates there instead of spilling.
    code = _cpu_code(
        rope_like, _rope_args(), num_threads=[0, 0, 8, 1, 1], cute_lane_unroll=16
    )
    loops = _loops(code, "synthetic_lane_3")
    assert len(loops) == 2
    assert all(ast.unparse(loop.iter) == "cutlass.range_constexpr(8)" for loop in loops)
    assert code.count("_lane_unroll_loads_0.append(") == 1
    assert code.count("_cute_store_u16_vec(") == 4


def test_unequal_slice_thread_counts_keep_separate_lanes(cpu_only: None) -> None:
    code = _cpu_code(rope_like, _rope_args(), num_threads=[0, 0, 8, 4, 1])
    # The k heads walk their whole extent per thread (no shared axis), still
    # as packets.
    assert "indices_4 = cutlass.Int32(0) + cutlass.Int32(synthetic_lane_4)" in code
    assert code.count("_cute_store_u16_vec(") == 4
    assert ").store(" not in code
    assert "block=(8, 4, 16)" in code


def test_without_the_sink_the_nested_slice_loads_stay_scalar(cpu_only: None) -> None:
    code = _cpu_code(rope_like, _rope_args(), cute_vloop_sink=False)
    # The V-loop wraps the head loop; the cos/sin loads and the k accesses
    # (one head per thread, no lane loop) sit directly in its body and become
    # packets, the q accesses inside the q head loop stay scalar.
    assert len(_loops(code, "synthetic_lane_2")) == 1
    assert code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 6
    assert code.count("_cute_store_u16_vec(") == 2
    assert ").load()" in code
    assert ").store(" in code


def test_values_holding_both_axes_do_not_share_a_thread_axis(cpu_only: None) -> None:
    x = torch.empty(32, 8, 64, dtype=torch.bfloat16)
    bound = _cpu_bind(scale_rows, (x,))
    spec = bound.config_spec
    # Block 1 is the 8-wide axis, block 2 the 64-wide (stride-1) axis.
    num_threads = [0] * len(spec.num_threads)
    num_threads[spec.num_threads.block_id_to_index(1)] = 8
    num_threads[spec.num_threads.block_id_to_index(2)] = 8
    widths = [1] * len(spec.cute_vector_widths)
    widths[spec.cute_vector_widths.block_id_to_index(2)] = 8
    config = spec.normalized_config(
        helion.Config(
            block_sizes=[1],
            num_threads=num_threads,
            cute_vector_widths=widths,
            reduction_loops=[None],
        )
    )
    code = bound.to_code(config)
    # Equal explicit thread counts, but the two axes meet in ``x[tile, :, :]``:
    # the 8-wide axis keeps thread axis 0 and the vector lane takes axis 1.
    assert "indices_1 = cutlass.Int32(cute.arch.thread_idx()[0])" in code
    assert (
        "reduction_lane_base_2 = cutlass.Int32(cute.arch.thread_idx()[1]) * 8" in code
    )
    assert "block=(8, 8, 1)" in code


def test_pointwise_seeds_cover_the_persistent_slice(cpu_only: None) -> None:
    bound = _cpu_bind(rope_like, _rope_args())
    spec = bound.env.config_spec
    # The live-ins of the compiler's rolled lane-loop graphs are not loop
    # carried accumulators: the kernel stays in the pointwise family.
    assert not spec.accumulator_facts
    assert spec.pointwise_facts
    assert "cute_pointwise_vec" in spec.autotuner_heuristics
    seeds = spec.compiler_seed_configs
    assert seeds
    vec_slot = spec.cute_vector_widths.block_id_to_index(2)
    nt_slot = spec.num_threads.block_id_to_index(2)
    for seed in seeds:
        assert seed.config["cute_vloop_sink"] is True
        assert seed.config["cute_vector_widths"][vec_slot] == 8
        assert seed.config["num_threads"][nt_slot] == 8
        assert seed.config["reduction_loops"] == [None]
    # Several per-thread work splits: the q heads over 1, 2, 4 and 8 threads,
    # the k heads sharing the count whenever it divides their extent.
    q_slot = spec.num_threads.block_id_to_index(3)
    k_slot = spec.num_threads.block_id_to_index(4)
    splits = {seed.config["num_threads"][q_slot] for seed in seeds}
    assert splits == {1, 2, 4, 8}
    for seed in seeds:
        threads = seed.config["num_threads"]
        assert threads[k_slot] == (threads[q_slot] if threads[q_slot] <= 4 else 1)


def test_one_vector_lane_stays_persistent_beside_wide_tiles(cpu_only: None) -> None:
    bound = _cpu_bind(rope_like, _rope_args())
    spec = bound.config_spec
    # 32 seq threads leave 32 of the 1024-thread budget for the 64-wide slice;
    # the one-vector lane only needs 8 of them and stays persistent.
    config = spec.normalized_config(
        _config(block_sizes=[1, 32], num_threads=[0, 0, 8, 2, 2])
    )
    assert config.config["reduction_loops"] == [None]
    # A scalar request that cannot cover the extent is still rolled.
    rolled = spec.normalized_config(
        _config(
            block_sizes=[1, 32],
            num_threads=[0, 0, 8, 2, 2],
            cute_vector_widths=[1, 1, 1, 1, 1],
        )
    )
    assert rolled.config["reduction_loops"] != [None]


@skipIfNotCUDA()
def test_rope_like_matches_the_reference_on_the_gpu() -> None:
    torch.manual_seed(0)
    args = tuple(
        torch.randn(t.shape, dtype=t.dtype, device=DEVICE) for t in _rope_args()
    )
    _code, (q_out, k_out) = code_and_output(rope_like, args, **_config().config)
    expected_q, expected_k = _rope_reference(*args)
    torch.testing.assert_close(q_out, expected_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, expected_k, rtol=1e-2, atol=1e-2)


def test_row_reduction_over_two_tile_axes_gets_the_reduction_tile_seeds(
    cpu_only: None,
) -> None:
    bound = _cpu_bind(
        batched_row_softmax, (torch.empty(8, 256, 1024, dtype=torch.float16),)
    )
    spec = bound.env.config_spec
    assert "cute_reduction_tile" in spec.autotuner_heuristics
    seeds = spec.compiler_seed_configs
    row_slot = spec.num_threads.block_id_to_index(2)
    vec_slot = spec.cute_vector_widths.block_id_to_index(2)
    reload_slot = spec.cute_reduction_reloads.block_id_to_index(2)
    # One row per CTA in every seed; the persistent one-vector rows with 16-
    # and 32-byte packets (128 x V8, 64 x V16) and the two-packet rolled row
    # (64 x V8 over the 1024-wide chunk) beside the narrow-chunk seed, the
    # rows re-reading registers.
    assert all(seed.config["block_sizes"] == [1, 1] for seed in seeds)
    rows = {
        (
            seed.config["num_threads"][row_slot],
            seed.config["cute_vector_widths"][vec_slot],
            seed.config["reduction_loops"][0],
            (seed.config.get("cute_reduction_reloads") or ["auto"] * 3)[reload_slot],
        )
        for seed in seeds
    }
    assert {
        (128, 8, None, "register_prefetch"),
        (64, 16, None, "register_prefetch"),
        (64, 8, 1024, "register_prefetch"),
    } <= rows
