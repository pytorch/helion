"""``cute_store_policy``: which packet flushes take a cache-policy helper."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CutePointwiseVecHeuristic
from helion._compiler.cute.packet_store_policy import lower_packet_store_policy
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends("cute")


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable():
        yield


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.shape):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


def _code(width: int, policy: str, *, static: bool) -> str:
    kernel = helion.kernel(_add, backend="cute", static_shapes=static)
    bound = _cpu_bind(
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
    bound = _cpu_bind(
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
