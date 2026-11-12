"""Pointwise seeds keep the inputs that fit in L2 resident and walk the grid in memory order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import _FALLBACK_L2_BYTES
from helion._compiler.cute.autotune.heuristics.cute import (
    _resident_input_eviction_policies,
)
from helion._compiler.cute.autotune.heuristics.cute import _reversed_root_loop_orders
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.sink_softmax_kernels import sink_normalize
import helion.language as hl

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_render_target():
        yield


def _rope_like(
    grad_q_out: torch.Tensor, grad_k_out: torch.Tensor, cos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A large stream, a small stream and a broadcast table (rope backward)."""
    batch, _, seq_len, head_dim = grad_q_out.size()
    half = head_dim // 2
    grad_q = torch.empty_like(grad_q_out)
    grad_k = torch.empty_like(grad_k_out)
    cos = cos.expand(batch, seq_len, head_dim).unsqueeze(1)
    for tile_b, tile_t in hl.tile([batch, seq_len]):
        c = cos[tile_b, :, tile_t, :half].float()
        q = grad_q_out[tile_b, :, tile_t, :half].float()
        k = grad_k_out[tile_b, :, tile_t, :half].float()
        grad_q[tile_b, :, tile_t, :half] = (q * c).to(grad_q.dtype)
        grad_k[tile_b, :, tile_t, :half] = (k * c).to(grad_k.dtype)
    return grad_q, grad_k


def _two_streams(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _bind(fn: Callable[..., Any], *args: torch.Tensor) -> BoundKernel:
    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    return cpu_bind(kernel, args)


def _rope_args(k_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 67 MB query gradient, k_heads * 2.1 MB key gradient, 2 MB table.
    return (
        torch.empty((4, 32, 2048, 128), dtype=torch.bfloat16),
        torch.empty((4, k_heads, 2048, 128), dtype=torch.bfloat16),
        torch.empty((1, 2048, 128), dtype=torch.bfloat16).expand(4, 2048, 128),
    )


def _slots(bound: BoundKernel, tensor: str) -> set[int]:
    return {
        fact.eviction_index
        for fact in bound.config_spec.memory_op_facts
        if fact.kind == "load" and fact.tensor_name == tensor
        if fact.eviction_index is not None
    }


def test_small_inputs_get_l2_last_and_the_large_stream_stays_default() -> None:
    bound = _bind(_rope_like, *_rope_args(k_heads=4))
    with bound.env:
        policies = _resident_input_eviction_policies(bound.env)
    assert policies is not None
    spec = bound.config_spec
    assert len(policies) == spec.load_eviction_policies.length
    small = _slots(bound, "grad_k_out") | _slots(bound, "cos")
    large = _slots(bound, "grad_q_out")
    assert small and large
    assert {policies[slot] for slot in small} == {"l2_last"}
    assert {policies[slot] for slot in large} == {""}


def test_inputs_beyond_a_quarter_of_l2_are_not_kept() -> None:
    # 32 key heads: the "small" stream is as large as the query gradient and
    # exceeds a quarter of the fallback L2; only the table stays resident.
    bound = _bind(_rope_like, *_rope_args(k_heads=32))
    with bound.env:
        policies = _resident_input_eviction_policies(bound.env)
    assert policies is not None
    assert {policies[slot] for slot in _slots(bound, "cos")} == {"l2_last"}
    assert {policies[slot] for slot in _slots(bound, "grad_k_out")} == {""}
    assert _FALLBACK_L2_BYTES // 4 < (4 * 32 * 2048 * 128 * 2)


def test_equal_streams_have_no_resident_variant() -> None:
    bound = _bind(_two_streams, torch.empty((4096, 4096)), torch.empty((4096, 4096)))
    with bound.env:
        assert _resident_input_eviction_policies(bound.env) is None


def test_reversed_root_loop_order_lists_the_last_axis_first() -> None:
    bound = _bind(_rope_like, *_rope_args(k_heads=4))
    with bound.env:
        assert _reversed_root_loop_orders(bound.env) == [[1, 0]]
    one_axis = _bind(_two_streams, torch.empty((65536,)), torch.empty((65536,)))
    with one_axis.env:
        assert _reversed_root_loop_orders(one_axis.env) is None


def test_persistent_pointwise_seeds_carry_the_resident_and_reversed_siblings() -> None:
    bound = _bind(_rope_like, *_rope_args(k_heads=4))
    seeds = bound.config_spec.compiler_seed_configs
    assert seeds
    with bound.env:
        resident = _resident_input_eviction_policies(bound.env)
    plain = [s for s in seeds if not s.config.get("load_eviction_policies")]
    kept = [s for s in seeds if s.config.get("load_eviction_policies") == resident]
    reversed_order = [s for s in kept if s.config.get("loop_orders") == [[1, 0]]]
    assert plain and kept and reversed_order
    # Every geometry gets both siblings and keeps the persistent vector lane.
    assert len(kept) == 2 * len(plain)
    for seed in seeds:
        assert seed.config["cute_vloop_sink"] is True
    # The sibling only changes the eviction list (and the order): same geometry.
    assert {_geometry(s) for s in kept} == {_geometry(s) for s in plain}


def _geometry(seed: helion.Config) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(seed.config["block_sizes"]), tuple(seed.config["num_threads"])  # pyrefly: ignore[bad-argument-type]


def _gather_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """embedding: rows gathered from a table, a two-axis pointwise tile."""
    out = torch.empty(
        [x.size(0), weight.size(1)], dtype=weight.dtype, device=weight.device
    )
    for tile_b, tile_e in hl.tile([x.size(0), weight.size(1)]):
        out[tile_b, tile_e] = weight[x[tile_b], tile_e]
    return out


def test_two_axis_pointwise_seeds_carry_wide_row_packets() -> None:
    bound = _bind(
        _gather_rows,
        torch.randint(32768, (8192,), dtype=torch.int32),
        torch.empty((32768, 512)),
    )
    seeds = bound.config_spec.compiler_seed_configs
    wide = [s for s in seeds if s.config.get("cute_min_blocks_per_mp") == 1]
    # One seed with the register hint alone and its programmatic-dependent twin
    # (plus their proven-bounds counterparts).
    assert len(wide) >= 2
    for seed in wide:
        blocks = cast("list[int]", seed.config["block_sizes"])
        threads = cast("list[int]", seed.config["num_threads"])
        widths = cast("list[int]", seed.config["cute_vector_widths"])
        layouts = cast("list[str]", seed.config["cute_lane_layouts"])
        # 32-byte fp32 packets: eight lanes across the contiguous axis cover a
        # 256-byte row segment per load, four loads per thread, 32 rows per CTA.
        assert widths[-1] == 8
        assert threads[-1] == 8 and threads[0] == 32
        assert blocks == [32, 256]
        assert layouts[-1] == "strided"
        assert seed.config["loop_orders"] == [[1, 0]]
    assert {bool(s.config.get("cute_pdl")) for s in wide} == {False, True}
    # A one-axis kernel has no row axis to batch: no such seed.
    flat = _bind(_two_streams, torch.empty((65536,)), torch.empty((65536,)))
    assert not [
        s
        for s in flat.config_spec.compiler_seed_configs
        if s.config.get("cute_min_blocks_per_mp") == 1
    ]


@pytest.mark.parametrize(
    ("width", "block_sizes", "num_threads"),
    [
        # 24 packets take eight lanes either way; the tile rounds from 384.
        pytest.param(384, [32, 256], [32, 8], id="384"),
        # 6 packets: the lanes round from six (42 rows) to four (64 rows)
        # and the tile from 96.
        pytest.param(96, [64, 64], [64, 4], id="96"),
    ],
)
def test_wide_row_packets_round_a_row_down_to_a_power_of_two_tile(
    width: int, block_sizes: list[int], num_threads: list[int]
) -> None:
    """A bf16 row of 16-element packets that is not a power of two wide
    rounds the tile and the lanes down to powers of two: the config spec
    rejects other block sizes and thread counts, which would drop the seed
    before the search sees it."""
    bound = _bind(
        _gather_rows,
        torch.randint(32768, (8192,), dtype=torch.int32),
        torch.empty((32768, width), dtype=torch.bfloat16),
    )
    spec = bound.config_spec
    wide = [
        s
        for s in spec.compiler_seed_configs
        if s.config.get("cute_min_blocks_per_mp") == 1
    ]
    assert wide
    for seed in wide:
        assert seed.config["block_sizes"] == block_sizes
        assert seed.config["num_threads"] == num_threads
        assert cast("list[int]", seed.config["cute_vector_widths"])[-1] == 16
        spec.normalized_config(seed)


@pytest.mark.parametrize(
    ("width", "seeded"),
    [(1000, True), (1024, False), (600, False)],
    ids=["ragged", "power of two", "under 7/8 of its tile"],
)
def test_narrow_rows_seed_a_whole_row_tile(width: int, seeded: bool) -> None:
    """Where the pointwise seed flattens a row too narrow for a CTA of
    packets, a row whose length is no power of two also gets a tile of its
    own: a packet per thread across the row, a row per thread row and the
    lanes' math paired.  Its kernel splits no flat offset by the row length
    and loads the broadcast mask as packets.  A power-of-two row, whose flat
    split is shifts and masks, and a row that would leave more than an
    eighth of its tile masked keep the flattened seeds alone."""
    bound = _bind(
        sink_normalize,
        torch.empty((8, 64, width), dtype=torch.bfloat16),
        torch.empty((64, width), dtype=torch.bfloat16),
        torch.empty((8, 64)),
    )
    seeds = bound.config_spec.compiler_seed_configs
    assert seeds[0].config["flatten_loops"] == [True]
    rows = [s for s in seeds if s.config.get("cute_pair_lane_math")]
    if not seeded:
        assert rows == []
        return
    # The seed and its proven-bounds counterpart.
    assert sorted(bool(s.config.get("cute_proven_bounds")) for s in rows) == [
        False,
        True,
    ]
    for seed in rows:
        assert _geometry(seed) == ((1, 1, 1024), (1, 1, 128))
        assert seed.config["cute_vector_widths"] == [1, 1, 8]
        assert not any(cast("list[bool]", seed.config.get("flatten_loops", [])))
    code = bound.to_code(rows[0])
    assert f"% {width}" not in code and f"// {width}" not in code
    assert code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 2
    assert "_cute_exp_f32x2_parts(" in code


@pytest.mark.parametrize(
    ("queries", "geometry"),
    [(64, ((1, 2, 512), (1, 2, 64))), (1, ((2, 1, 512), (2, 1, 64)))],
    ids=["queries", "heads"],
)
def test_a_row_tile_takes_its_rows_from_an_outer_axis_long_enough(
    queries: int, geometry: tuple[tuple[int, ...], tuple[int, ...]]
) -> None:
    """A 504-wide row spans 64 lanes of eight, so a 128-thread CTA takes two
    rows: from the queries' axis, or from the heads when it holds one."""
    bound = _bind(
        sink_normalize,
        torch.empty((8, queries, 504), dtype=torch.bfloat16),
        torch.empty((queries, 504), dtype=torch.bfloat16),
        torch.empty((8, queries)),
    )
    rows = [
        s
        for s in bound.config_spec.compiler_seed_configs
        if s.config.get("cute_pair_lane_math")
    ]
    assert rows
    for seed in rows:
        assert _geometry(seed) == geometry
