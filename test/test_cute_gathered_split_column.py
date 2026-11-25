"""CPU tests: ``CuteGatheredSplitColumnHeuristic`` seeds a split column
reduction that reads its columns through a flat gather along the column tile
(helion.inductor's copy reindexed onto a split's first kernel): one split per
CTA, 8 column threads of 16-byte packets, 16 row threads, the gather sunk."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._compiler.cute import flat_gather_lane
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteGatheredSplitColumnHeuristic,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _merged_heads_sum(
    x: torch.Tensor, out: torch.Tensor, partials: torch.Tensor
) -> None:
    # The heads merge of a [16, 128, 64] attention output (4 images of 4
    # heads) beside its split sum over the 512 rows of 4 x 128 positions in
    # 16 chunks of 32: the copy reindexed onto the split's first kernel.
    for tile_z, tile_y, tile_x in hl.tile([16, 4, 64]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            # helion.inductor's flat subscript, the column tile's index last.
            z = tile_z.index.reshape([tile_z, 1, 1, 1])
            r = tile_r.index.reshape([tile_r, 1])
            row = (
                64 * ((r + 32 * z) % 128)
                + 8192 * tile_y.index.reshape([tile_y, 1, 1])
                + 32768 * (z // 4)
            )
            value = x[row + tile_x.index].to(torch.float32)
            out[tile_z, tile_y, tile_r, tile_x] = value.to(torch.bfloat16)
            acc = acc + torch.sum(value, -2)
        partials[tile_z, tile_y, tile_x] = acc


@helion.kernel(backend="cute", static_shapes=True)
def _scaled_lane_sum(
    x: torch.Tensor, out: torch.Tensor, partials: torch.Tensor
) -> None:
    # The same nest whose gather takes every other element along the columns:
    # no lane term of unit coefficient, no seed.
    for tile_z, tile_y, tile_x in hl.tile([16, 4, 64]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            z = tile_z.index.reshape([tile_z, 1, 1, 1])
            r = tile_r.index.reshape([tile_r, 1])
            row = (
                128 * ((r + 32 * z) % 128)
                + 16384 * tile_y.index.reshape([tile_y, 1, 1])
                + 65536 * (z // 4)
            )
            value = x[row + 2 * tile_x.index].to(torch.float32)
            out[tile_z, tile_y, tile_r, tile_x] = value.to(torch.bfloat16)
            acc = acc + torch.sum(value, -2)
        partials[tile_z, tile_y, tile_x] = acc


def _args(numel: int) -> tuple[torch.Tensor, ...]:
    out = torch.empty_strided((16, 4, 32, 64), (8192, 64, 256, 1), dtype=torch.bfloat16)
    return (torch.empty(numel, dtype=torch.bfloat16), out, torch.empty(16, 4, 64))


def test_a_gathered_split_column_sum_takes_the_packet_seed() -> None:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_merged_heads_sum, _args(131072))
        spec = bound.config_spec
        assert "cute_gathered_split_column" in spec.autotuner_heuristics
        (seed,) = CuteGatheredSplitColumnHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        # One split per CTA (below the search's floor of the split axis),
        # the 64 columns as 8 threads of 8-element bf16 packets, 16 row
        # threads over the split's 32 rows.
        seed = spec.normalized_config(seed)
        assert seed.config["block_sizes"] == [1, 1, 64, 32]
        assert seed.config["num_threads"] == [1, 1, 8, 16]
        assert seed.config["cute_vector_widths"] == [1, 1, 8, 1]
        assert seed.config["cute_vloop_sink"] is True
        code = bound.to_code(seed)
    assert re.search(r"block=\(8, 16, 1\)", code)
    # The gather loads its rows in packets.
    assert "_vsink_vec" in code


@helion.kernel(backend="cute", static_shapes=True)
def _unaligned_rows_sum(
    x: torch.Tensor, out: torch.Tensor, partials: torch.Tensor
) -> None:
    # The same nest whose gathered rows are 66 elements apart: no row of
    # whole 16-byte packets, which the sink would need, no seed.
    for tile_z, tile_y, tile_x in hl.tile([16, 4, 64]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            z = tile_z.index.reshape([tile_z, 1, 1, 1])
            r = tile_r.index.reshape([tile_r, 1])
            row = (
                66 * ((r + 32 * z) % 128)
                + 8448 * tile_y.index.reshape([tile_y, 1, 1])
                + 33792 * (z // 4)
            )
            value = x[row + tile_x.index].to(torch.float32)
            out[tile_z, tile_y, tile_r, tile_x] = value.to(torch.bfloat16)
            acc = acc + torch.sum(value, -2)
        partials[tile_z, tile_y, tile_x] = acc


def test_rows_the_sink_cannot_load_in_packets_take_no_seed() -> None:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        spec = cpu_bind(_unaligned_rows_sum, _args(135168)).config_spec
    assert "cute_gathered_split_column" not in spec.autotuner_heuristics


@helion.kernel(backend="cute", static_shapes=True)
def _mixed_gathers_sum(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, partials: torch.Tensor
) -> None:
    # The same nest gathering a bf16 and an fp32 tensor through rows 68
    # elements apart: the sink proves 4-element packets for both gathers (16
    # bytes of fp32), where the seed of the bf16 load would ask for 8: no
    # seed.
    for tile_z, tile_y, tile_x in hl.tile([16, 4, 64]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            z = tile_z.index.reshape([tile_z, 1, 1, 1])
            r = tile_r.index.reshape([tile_r, 1])
            row = (
                68 * ((r + 32 * z) % 128)
                + 8704 * tile_y.index.reshape([tile_y, 1, 1])
                + 34816 * (z // 4)
            )
            value = x[row + tile_x.index].to(torch.float32) * w[row + tile_x.index]
            out[tile_z, tile_y, tile_r, tile_x] = value.to(torch.bfloat16)
            acc = acc + torch.sum(value, -2)
        partials[tile_z, tile_y, tile_x] = acc


def test_a_seed_wider_than_the_sinks_packets_is_declined() -> None:
    x, out, partials = _args(139264)
    w = torch.empty(139264, dtype=torch.float32)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_mixed_gathers_sum, (x, w, out, partials))
        env, device_ir = bound.env, bound.host_function.device_ir
        match = CuteGatheredSplitColumnHeuristic._match(env, device_ir)
        assert match is not None
        column_block_id, reduced_block_id, source = match
        assert source.dtype == torch.bfloat16
        assert (
            flat_gather_lane.flat_gather_sink_vec_width(
                env, device_ir, {column_block_id}, {reduced_block_id}
            )
            == 4
        )
        assert CuteGatheredSplitColumnHeuristic.get_seed_configs(env, device_ir) == []
    assert "cute_gathered_split_column" not in bound.config_spec.autotuner_heuristics


def test_a_gather_without_a_unit_lane_takes_no_seed() -> None:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        spec = cpu_bind(_scaled_lane_sum, _args(262144)).config_spec
    assert "cute_gathered_split_column" not in spec.autotuner_heuristics
