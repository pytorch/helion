"""``cute_store_policy``: which packet flushes take a cache-policy helper."""

from __future__ import annotations

import ast
import functools
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.autotune import search_surface
from helion._compiler.cute.autotune.heuristics.cute import CutePointwiseVecHeuristic
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.packet_store_policy import lower_packet_store_policy
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.stored_row_kernels import centered_slices
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends("cute")]


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with mock_cuda_unavailable():
        yield


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.shape):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


def _code(width: int, policy: str, *, static: bool) -> str:
    kernel = helion.kernel(_add, backend="cute", static_shapes=static)
    bound = cpu_bind(
        kernel,
        (
            torch.empty(512, 4096, dtype=torch.bfloat16),
            torch.empty(512, 4096, dtype=torch.bfloat16),
        ),
    )
    return bound.to_code(
        helion.Config(
            block_sizes=[1, 4096],
            num_threads=[0, 256],
            cute_vector_widths=[1, width],
            cute_lane_layouts=["blocked", "strided"],
            cute_store_policy=policy,
        )
    )


@pytest.mark.parametrize("static", [True, False])
def test_streaming_policy_renames_every_full_packet_flush(static: bool) -> None:
    code = _code(8, "streaming", static=static)
    assert code.count("_cute_store_u16x8_streaming(") == 1
    assert "_cute_store_u16_vec(" not in code
    # Sixteen bf16 lanes flush as two 16-byte packets (sliced by the wide-lane
    # emitter under dynamic shapes, split from one 32-byte collector of the
    # hoisted-vector emitter under static shapes); both take the policy.
    code = _code(16, "streaming", static=static)
    assert code.count("_cute_store_u16x8_streaming(") == 2
    assert "_cute_store_u16_vec(" not in code
    assert "[0:8])" in code and "[8:16])" in code


def test_default_policy_keeps_the_plain_flush() -> None:
    code = _code(8, "default", static=False)
    assert "_cute_store_u16_vec(" in code
    assert "streaming" not in code


def test_l2_evict_last_needs_the_validated_target() -> None:
    # The CPU binding targets SM100, where the opaque descriptor is not
    # admitted: the knob normalizes to the default flush.
    code = _code(8, "l2_evict_last", static=False)
    assert "_cute_store_u16_vec(" in code
    assert "l2_evict_last" not in code


def _lower(source: str, policy: str = "streaming") -> str:
    body = ast.parse(source).body
    lowered = lower_packet_store_policy(
        body,
        policy=policy,
        global_tensor_names=frozenset({"out"}),
        target_device_capability=(10, 0),
        cuda_version="13.0",
    )
    return ast.unparse(ast.Module(body=lowered, type_ignores=[]))


def test_sliced_collectors_rename_only_full_packets() -> None:
    source = """
vals = []
for lane in cutlass.range_constexpr(16):
    vals.append(cutlass.BFloat16(lane).bitcast(cutlass.Uint16))
_cute_store_u16_vec(out.iterator + 0, vals[0:8])
_cute_store_u16_vec(out.iterator + 8, vals[8:16])
_cute_store_u16_vec(out.iterator + 16, vals[4:8])
_cute_store_u16_vec(out.iterator + 24, vals[8:24])
_cute_store_u16_vec(other.iterator + 0, vals[0:8])
"""
    lowered = _lower(source)
    assert lowered.count("_cute_store_u16x8_streaming(") == 2
    assert "_cute_store_u16x8_streaming(out.iterator + 0, vals[0:8])" in lowered
    assert "_cute_store_u16x8_streaming(out.iterator + 8, vals[8:16])" in lowered
    # A half packet, a slice past the collector and a non-global target keep
    # the default flush.
    assert lowered.count("_cute_store_u16_vec(") == 3
    assert _lower(source, "default") == ast.unparse(ast.parse(source))


def test_whole_32_byte_collector_splits_into_two_policy_flushes() -> None:
    source = """
vals = []
for lane in cutlass.range_constexpr(16):
    vals.append(cutlass.BFloat16(lane).bitcast(cutlass.Uint16))
_cute_store_u16_vec(out.iterator + base * 16, vals)
words = []
for lane in cutlass.range_constexpr(8):
    words.append(cutlass.Float32(lane).bitcast(cutlass.Uint32))
_cute_store_u32_vec(out.iterator + base * 8, words)
"""
    lowered = _lower(source)
    assert "_cute_store_u16x8_streaming(out.iterator + base * 16, vals[0:8])" in lowered
    assert (
        "_cute_store_u16x8_streaming(out.iterator + base * 16 + 8, vals[8:16])"
        in lowered
    )
    assert "_cute_store_u32x4_streaming(out.iterator + base * 8, words[0:4])" in lowered
    assert (
        "_cute_store_u32x4_streaming(out.iterator + base * 8 + 4, words[4:8])"
        in lowered
    )
    assert (
        "_cute_store_u16_vec(" not in lowered and "_cute_store_u32_vec(" not in lowered
    )


def test_pointwise_seeds_offer_a_streaming_store_alternate() -> None:
    kernel = helion.kernel(_add, backend="cute", static_shapes=True)
    bound = cpu_bind(
        kernel,
        (
            torch.empty(4096, 4096, dtype=torch.bfloat16),
            torch.empty(4096, 4096, dtype=torch.bfloat16),
        ),
    )
    with bound.env:
        seeds = CutePointwiseVecHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    policies = [seed.config.get("cute_store_policy", "default") for seed in seeds]
    assert "streaming" in policies and "default" in policies
    streaming = next(
        seed for seed in seeds if seed.config.get("cute_store_policy") == "streaming"
    )
    primary = seeds[0]
    # The alternate differs from the primary seed only in the store policy.
    assert {k: v for k, v in streaming.config.items() if k != "cute_store_policy"} == {
        k: v for k, v in primary.config.items() if k != "cute_store_policy"
    }
    bound.config_spec.normalize(streaming)
    assert streaming.config["cute_store_policy"] == "streaming"
    assert "_cute_store_u16x8_streaming(" in bound.to_code(streaming)


def _to_channels_last(x: torch.Tensor) -> torch.Tensor:
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


def _scale_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] * 2
    return out


def _row_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = torch.softmax(x[tile_m, :].to(torch.float32), dim=-1)
    return out


def _row_layout(bound: BoundKernel) -> dict[str, object]:
    """Four threads along a 64-element row (16 lanes each), strided."""
    (row,) = [block.block_id for block in bound.env.block_sizes if block.reduction]
    spec = bound.config_spec
    return {
        "block_sizes": [1],
        "num_threads": [
            4 if block == row else 1 for block in spec.num_threads.valid_block_ids()
        ],
        "cute_lane_layouts": [
            "strided" if block == row else "blocked"
            for block in spec.cute_lane_layouts.valid_block_ids()
        ],
    }


# One kernel per way a store gets its lanes: the tiles of a grid, in tensor
# order and transposed, the flattened tiles of a grid, and the full slice of
# a row with and without a reduction over it.  Each with its operands, its
# layout and the vector-width slot of the block on the stored tensor's
# contiguous dim: the tile there, or the full slice's own block (the first
# slot).
_LANE_KERNELS = {
    "tiles": (_add, 2, {"block_sizes": [1, 64], "num_threads": [0, 4]}, 1),
    "transposed": (
        _to_channels_last,
        1,
        {"block_sizes": [1, 64, 8], "num_threads": [1, 4, 8]},
        1,
    ),
    "flattened": (
        _add,
        2,
        {"block_sizes": [1, 64], "num_threads": [0, 4], "flatten_loops": [True]},
        1,
    ),
    "row": (_scale_rows, 1, _row_layout, 0),
    "reduced row": (_row_softmax, 1, _row_layout, 0),
}


@functools.cache
def _lane_bound(name: str) -> BoundKernel:
    function, operands, _, _ = _LANE_KERNELS[name]
    shape = (4, 64, 8) if name == "transposed" else (8, 64)
    kernel = helion.kernel(function, backend="cute", static_shapes=True)
    return cpu_bind(
        kernel, tuple(torch.empty(shape, dtype=torch.bfloat16) for _ in range(operands))
    )


@pytest.mark.parametrize("name", list(_LANE_KERNELS))
def test_a_policy_no_packet_takes_normalizes_to_default(name: str) -> None:
    """A 16-byte packet of a bf16 store is eight lanes of the block on the
    stored tensor's contiguous dim.  Narrower lanes there, and lanes of any
    width on another block, store nothing the policy applies to: the config
    is the default-policy config, with the same code.  (Behind a full slice
    the lanes are the slice's own block's.)  The reset must never change the
    code, so each config is rendered again with the reset switched off."""
    bound = _lane_bound(name)
    spec = bound.config_spec
    _, _, layout, lane = _LANE_KERNELS[name]
    if callable(layout):
        layout = layout(bound)
    slots = len(spec.cute_vector_widths)
    kept = []
    for block in range(slots):
        for width in (4, 8, 16):
            widths = [width if slot == block else 1 for slot in range(slots)]
            requested = helion.Config(
                **layout, cute_vector_widths=widths, cute_store_policy="streaming"
            )
            normalized = spec.normalized_config(requested)
            policy = normalized.config.get("cute_store_policy", "default")
            reaches = lane == block and width >= 8
            assert policy == ("streaming" if reaches else "default"), (widths, policy)
            code = bound.to_code(normalized)
            with patch.object(
                search_surface, "store_policy_reaches_a_packet", return_value=True
            ):
                unreset = spec.normalized_config(requested)
                assert unreset.config["cute_store_policy"] == "streaming"
                assert bound.to_code(unreset) == code, widths
            if "_streaming(" in code:
                kept.append(widths)
            else:
                # Normalization is idempotent, and the default-policy config
                # is the same config.
                assert spec.normalized_config(normalized) == normalized
    # The policy does apply where the reset keeps it: the test would pass
    # trivially on a kernel that stores no packet at all.
    assert kept, name


def _pairs(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = -value
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_component_packet_keeps_its_policy(dtype: torch.dtype) -> None:
    """A component packet has its lanes on the dim ahead of the stride-1
    one: ``out[tile, c]`` on ``[elements, 2]`` flushes both components of
    its vector lanes in one store, and no tile indexes the contiguous dim.
    The policy stays where that packet is one the policy applies to (16 or
    32 bytes) and is reset on narrower lanes and on lanes too wide for a
    packet; the reset changes no code either way."""
    kernel = helion.kernel(_pairs, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (torch.empty(4096, dtype=dtype),))
    spec = bound.config_spec
    kept = []
    for width in (2, 4, 8, 16):
        requested = helion.Config(
            block_sizes=[64 * width],
            num_threads=[64],
            cute_vector_widths=[width],
            cute_store_policy="streaming",
        )
        normalized = spec.normalized_config(requested)
        packet_bytes = width * 2 * dtype.itemsize
        assert normalized.config.get("cute_store_policy", "default") == (
            "streaming" if 16 <= packet_bytes <= 32 else "default"
        ), width
        code = bound.to_code(normalized)
        with patch.object(
            search_surface, "store_policy_reaches_a_packet", return_value=True
        ):
            unreset = spec.normalized_config(requested)
            assert unreset.config["cute_store_policy"] == "streaming"
            assert bound.to_code(unreset) == code, width
        if "_streaming(" in code:
            kept.append(width)
    assert kept == [8 // dtype.itemsize, 16 // dtype.itemsize]


def _reset_keeps_the_code(bound: BoundKernel, requested: helion.Config) -> str:
    """The normalized policy of ``requested`` and its code, which must be
    the code of the config with the reset switched off."""
    spec = bound.config_spec
    normalized = spec.normalized_config(requested)
    code = bound.to_code(normalized)
    with patch.object(
        search_surface, "store_policy_reaches_a_packet", return_value=True
    ):
        unreset = spec.normalized_config(requested)
        assert unreset.config["cute_store_policy"] == "streaming"
        assert bound.to_code(unreset) == code, requested
    assert ("_streaming(" in code) <= (
        normalized.config.get("cute_store_policy") == "streaming"
    )
    return normalized.config.get("cute_store_policy", "default")


def test_a_row_loop_nests_rows_take_the_lanes_of_their_own_block() -> None:
    """The rows of a nest (a device loop over rows under a tile, each row
    stored through a full slice beside a carried column partial, as a
    LayerNorm backward does) are stored in the lanes of the row's own block.
    Eight bf16 lanes there keep the policy and its flushes; lanes of any
    width on the device loop's tile or on the outer tile store no packet
    and read the policy as the default, with the same code."""
    bound = cpu_bind(centered_slices, (torch.empty(4, 6, 128, dtype=torch.bfloat16),))
    spec = bound.config_spec
    (row,) = [block.block_id for block in bound.env.block_sizes if block.reduction]
    slots = spec.cute_vector_widths.valid_block_ids()
    layout = {
        "block_sizes": [1, 1],
        "num_threads": [
            16 if block == row else 1 for block in spec.num_threads.valid_block_ids()
        ],
        "cute_lane_layouts": [
            "strided" if block == row else "blocked"
            for block in spec.cute_lane_layouts.valid_block_ids()
        ],
    }
    for block in slots:
        for width in (8, 16):
            if block == row and width == 16:
                # Sixteen lanes on each of sixteen threads are more than the
                # row holds: kept, the pass forms no such lanes.
                continue
            policy = _reset_keeps_the_code(
                bound,
                helion.Config(
                    **layout,
                    cute_vector_widths=[
                        width if slot == block else 1 for slot in slots
                    ],
                    cute_store_policy="streaming",
                ),
            )
            assert policy == ("streaming" if block == row else "default"), (
                block,
                width,
            )


def _rotated_halves(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    half = x.size(1) // 2
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        lo = x[tile_m, :half].to(torch.float32) * scale
        hi = x[tile_m, half:].to(torch.float32) * scale
        out[tile_m, :half] = (-hi).to(x.dtype)
        out[tile_m, half:] = lo.to(x.dtype)
    return out


def test_a_partial_slices_rows_take_the_lanes_of_the_slices_block() -> None:
    """``out[tile, :half]`` beside a reduction over the whole row (the keys
    of a rotary embedding under an RMSNorm): the stored dim has the block of
    the whole row, but the halves run on a block of their own, and their
    lanes store the packets.  Eight lanes there keep the policy and its
    flushes, whatever the whole row's width is; lanes on the tile keep
    nothing."""
    kernel = helion.kernel(_rotated_halves, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (torch.empty(8, 128, dtype=torch.bfloat16),))
    spec = bound.config_spec
    full, half = [block.block_id for block in bound.env.block_sizes if block.reduction]
    stores = [fact for fact in spec.memory_op_facts if fact.kind == "store"]
    assert stores and all(fact.indexed_block_ids[-1] == full for fact in stores)
    slots = spec.cute_vector_widths.valid_block_ids()
    layout = {
        "block_sizes": [1],
        "num_threads": [
            8 if block == half else 0 if block == full else 1
            for block in spec.num_threads.valid_block_ids()
        ],
        "reduction_loops": [64],
    }

    def policy_and_code(block: int, width: int) -> tuple[str, str]:
        requested = helion.Config(
            **layout,
            cute_vector_widths=[width if slot == block else 1 for slot in slots],
            cute_store_policy="streaming",
        )
        policy = _reset_keeps_the_code(bound, requested)
        return policy, bound.to_code(spec.normalized_config(requested))

    policy, code = policy_and_code(half, 8)
    assert policy == "streaming"
    assert code.count("_streaming(") == 2
    for block in slots:
        if block not in (full, half):
            assert policy_and_code(block, 8)[0] == "default"
            assert policy_and_code(block, 16)[0] == "default"


def _add_into(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile(x.shape):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


def _pairs_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        value = x[tile]
        out[tile, 0] = value
        out[tile, 1] = -value
    return out


@pytest.mark.parametrize("offset", [0, 3, 4])
def test_an_output_view_off_the_packet_boundary_takes_no_policy(offset: int) -> None:
    """An output whose base is on no 16-byte boundary (a view three or four
    bf16 elements into its storage) is stored in no 16-byte packet: its
    policy reads as the default at every vector width, and the reset changes
    no code.  The same kernel on an aligned view keeps the policy."""
    x = torch.empty(8, 64, dtype=torch.bfloat16)
    out = torch.empty(8, 72, dtype=torch.bfloat16)[:, offset : offset + 64]
    kernel = helion.kernel(_add_into, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (x, x, out))
    assert bound.config_spec.cute_unaligned_store_tensors == (
        frozenset({"out"}) if offset else frozenset()
    )
    for width in (4, 8, 16):
        policy = _reset_keeps_the_code(
            bound,
            helion.Config(
                block_sizes=[1, 64],
                num_threads=[0, 4],
                cute_vector_widths=[1, width],
                cute_store_policy="streaming",
            ),
        )
        assert policy == ("default" if offset or width < 8 else "streaming"), width


def test_a_component_packet_off_the_boundary_takes_no_policy() -> None:
    """Both components of a lane's elements are one store of 16 bytes or
    more only into a tensor whose base is on that boundary: pairs of fp32
    eight bytes into their storage read the policy as the default."""
    x = torch.empty(4096, dtype=torch.float32)
    out = torch.empty(2 * 4096 + 2, dtype=torch.float32)[2:].view(4096, 2)
    kernel = helion.kernel(_pairs_into, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (x, out))
    assert bound.config_spec.cute_unaligned_store_tensors == frozenset({"out"})
    for width in (2, 4, 8):
        policy = _reset_keeps_the_code(
            bound,
            helion.Config(
                block_sizes=[64 * width],
                num_threads=[64],
                cute_vector_widths=[width],
                cute_store_policy="streaming",
            ),
        )
        assert policy == "default", width


def _row_sums_and_doubled(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty(x.size(0), dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        sums[tile_m] = x[tile_m, :].sum(-1)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n.index * 1] = x[tile_m, tile_n] * 2
    return out, sums


def test_a_computed_index_keeps_the_lanes_of_its_tile() -> None:
    """``out[tile_m, tile_n.index * 1]`` beside a reduction over the rows:
    the stored dim has the reduction's block and the subscript no block of
    its own, as a slice would, but its lanes are the column tile's.  Four
    fp32 lanes there are a 16-byte packet, and the policy stays with its
    flush whatever the reduction's lanes are."""
    kernel = helion.kernel(_row_sums_and_doubled, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (torch.empty(8, 128),))
    spec = bound.config_spec
    (row,) = [block.block_id for block in bound.env.block_sizes if block.reduction]
    tiles = spec.block_sizes.valid_block_ids()
    column = tiles[-1]
    (store,) = [
        fact for fact in spec.memory_op_facts if fact.kind == "store" and fact.ndim == 2
    ]
    assert store.subscript_block_ids[-1] is None
    assert store.indexed_block_ids[-1] == row
    assert spec.cute_sliced_store_dims == frozenset()
    requested = helion.Config(
        block_sizes=[64 if block == column else 1 for block in tiles],
        num_threads=[
            16 if block == column else 32 if block == row else 1
            for block in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            4 if block == column else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_store_policy="streaming",
    )
    assert _reset_keeps_the_code(bound, requested) == "streaming"
    assert "_streaming(" in bound.to_code(spec.normalized_config(requested))


def _doubled_under_a_new_axis(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.shape):
        out[None, tile_m, tile_n] = x[tile_m, tile_n][None, :, :] * 2
    return out


def test_a_store_under_a_new_axis_keeps_the_policy() -> None:
    """``out[None, tile_m, tile_n]``: the ``None`` takes a position of the
    index and no dim of ``out``, and the store's fact counts its subscripts
    along the index, so it reads the row tile on the stride-1 dim.  Four
    fp32 lanes of the column tile are a 16-byte packet all the same: the
    store facts of such a kernel are not read, and the policy stays with
    its flush.  (Without a ``None``, an integer among the subscripts or
    not, the facts stay in use.)"""
    kernel = helion.kernel(
        _doubled_under_a_new_axis, backend="cute", static_shapes=True
    )
    bound = cpu_bind(kernel, (torch.empty(8, 64),))
    spec = bound.config_spec
    row, column = spec.block_sizes.valid_block_ids()
    (store,) = [fact for fact in spec.memory_op_facts if fact.kind == "store"]
    assert store.subscript_block_ids == (None, row)
    assert store.subscript_strides == (64, 1)
    assert spec.cute_new_axis_stores
    requested = helion.Config(
        block_sizes=[1, 64],
        num_threads=[1, 8],
        cute_vector_widths=[1, 4],
        cute_store_policy="streaming",
    )
    assert _reset_keeps_the_code(bound, requested) == "streaming"
    assert "_streaming(" in bound.to_code(spec.normalized_config(requested))
    for name in ("tiles", "row"):
        assert not _lane_bound(name).config_spec.cute_new_axis_stores
    pairs = helion.kernel(_pairs, backend="cute", static_shapes=True)
    spec = cpu_bind(pairs, (torch.empty(4096),)).config_spec
    assert not spec.cute_new_axis_stores


def _doubled_into_a_view(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    view = out[2:]
    for tile in hl.tile(x.numel()):
        view[tile] = x[tile] * 2
    return out


@pytest.mark.parametrize("offset", [2, 4])
def test_a_view_that_restores_the_alignment_keeps_the_policy(offset: int) -> None:
    """The kernel stores into a view two fp32 elements into its argument.
    An argument that starts eight bytes off a 16-byte boundary puts the
    view on it: the view takes the packets and the policy.  An aligned
    argument puts the view eight bytes off, and the policy reads as the
    default.  The reset changes no code either way."""
    x = torch.empty(128)
    out = torch.empty(128 + 2 + offset)[offset:]
    assert out.data_ptr() % 16 == (8 if offset == 2 else 0)
    kernel = helion.kernel(_doubled_into_a_view, backend="cute", static_shapes=True)
    bound = cpu_bind(kernel, (x, out))
    spec = bound.config_spec
    assert spec.cute_unaligned_store_tensors == (
        frozenset() if offset == 2 else frozenset({"view"})
    )
    requested = helion.Config(
        block_sizes=[64],
        num_threads=[16],
        cute_vector_widths=[4],
        cute_store_policy="streaming",
    )
    policy = _reset_keeps_the_code(bound, requested)
    assert policy == ("streaming" if offset == 2 else "default")
    code = bound.to_code(spec.normalized_config(requested))
    assert ("_streaming(" in code) is (offset == 2)


def test_an_inert_policy_of_a_saved_config_loads_as_default() -> None:
    # A config an earlier version tuned with the policy carried along (the
    # pick of a search of better-benchmark's pointwise_06377502b8f5 copy:
    # vector widths [8, 4, 8], four channels per thread, scalar stores).
    bound = _lane_bound("transposed")
    saved = {
        "block_sizes": [4, 16, 8],
        "num_threads": [4, 16, 8],
        "cute_vector_widths": [8, 4, 8],
        "cute_lane_layouts": ["strided", "strided", "strided"],
        "loop_orders": [[1, 2, 0]],
        "cute_store_policy": "streaming",
    }
    normalized = bound.config_spec.normalized_config(helion.Config(**saved))
    assert normalized.config["cute_store_policy"] == "default"
    default = bound.config_spec.normalized_config(
        helion.Config(**(saved | {"cute_store_policy": "default"}))
    )
    assert normalized == default


def test_a_spec_without_store_facts_keeps_the_policy() -> None:
    # Nothing is known about the stores of a config spec that no kernel
    # filled in: the policy is not judged.
    config: dict[str, object] = {"cute_store_policy": "streaming"}
    ConfigSpec(backend=CuteBackend()).normalize(config)
    assert config["cute_store_policy"] == "streaming"


def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def test_a_matmul_kernel_keeps_the_policy() -> None:
    # A matmul's epilogue stores packets of its own, which the vector
    # widths of the tiles do not describe: the policy is not judged, here
    # at the default config's scalar lanes.
    kernel = helion.kernel(_matmul, backend="cute", static_shapes=True)
    x = torch.empty(64, 64, dtype=torch.bfloat16)
    spec = cpu_bind(kernel, (x, x)).config_spec
    assert spec.matmul_facts
    default = spec.default_config().config
    assert set(default["cute_vector_widths"]) == {1}
    normalized = spec.normalized_config(
        helion.Config.from_dict(dict(default) | {"cute_store_policy": "streaming"})
    )
    assert normalized.config["cute_store_policy"] == "streaming"


def test_random_flat_configs_are_normalized_as_generated() -> None:
    """The search's flat configs carry the reset (``repair_flat_config``):
    a random config is its own normalized form, and none carries a policy
    its vector widths leave without a packet."""
    bound = _lane_bound("transposed")
    spec = bound.config_spec
    saved = random.getstate()
    try:
        random.seed(20261118)
        with bound.env:
            generation = ConfigGeneration(spec)
            policy = generation._key_to_flat_indices["cute_store_policy"][0][0]
            drawn = set()
            for _ in range(64):
                for flat in (generation.random_flat(), generation.biased_random_flat()):
                    config = generation.unflatten(flat)
                    assert generation.flatten(config) == flat
                    channel_width = config.config["cute_vector_widths"][1]
                    drawn.add((flat[policy], channel_width >= 8))
    finally:
        random.setstate(saved)
    # Streaming survives exactly beside eight or more channel lanes.
    assert drawn == {("default", False), ("default", True), ("streaming", True)}


def test_a_streaming_seed_of_scalar_stores_is_its_primary() -> None:
    """The pointwise seeds vectorize the innermost tile, the positions, which
    this store has strided: their streaming alternates store scalars and now
    normalize to their primaries, so the search compiles each code once."""
    bound = _lane_bound("transposed")
    spec = bound.config_spec
    streaming = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_store_policy") == "streaming"
    ]
    assert streaming
    with bound.env:
        transferred = [
            config for _, config in ConfigGeneration(spec).seed_flat_config_pairs()
        ]
    assert all(
        config.config.get("cute_store_policy", "default") == "default"
        for config in transferred
    )
    assert len(transferred) <= len(spec.compiler_seed_configs) - len(streaming)
