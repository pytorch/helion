"""Pointwise seeds keep the inputs that fit in L2 resident and walk the grid in memory order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import _FALLBACK_L2_BYTES
from helion._compiler.autotuner_heuristics.cute import _resident_input_eviction_policies
from helion._compiler.autotuner_heuristics.cute import _reversed_root_loop_orders
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends("cute")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable():
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
    return _cpu_bind(kernel, args)


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
