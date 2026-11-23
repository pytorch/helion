"""Two row-reduction seed families appended after every other seed
(``postprocess_seed_configs``), and the normalization rule the second needs.

* ``heuristics/cute_row_stream_retention.py``: twins of the first row seeds
  that load packets, each with one input stream kept in L2
  (``l1_l2_last``) and the others streamed.
* ``heuristics/cute_lane_unrolled_warp_rows.py``: the lane-unrolled row at
  several one-warp rows per CTA, which stays persistent beside the wider
  tile axis because its config names ``cute_lane_unroll``
  (``_cute_config_forms_lane_unrolled_row``).

The GPU numerics of the warp rows are at the end of the module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune.heuristics import cute_row_stream_retention
from helion._compiler.cute.autotune.heuristics.cute_lane_unrolled_warp_rows import (
    CuteLaneUnrolledWarpRowsHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_lane_unrolled_warp_rows import (
    lane_unrolled_warp_rows_seed_configs,
)
from helion._compiler.cute.autotune.heuristics.cute_row_stream_retention import (
    CuteRowStreamRetentionHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_row_stream_retention import (
    row_stream_retention_seed_configs,
)
from helion._compiler.cute.autotune.heuristics.cute_row_stream_retention import (
    row_streams,
)
from helion._compiler.cute.target_facts import TargetFacts
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render(TargetFacts())]


def _two_stream_norm(
    residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    rows, hidden = x.shape
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        summed = residual[tile, :].to(torch.float32) + x[tile, :].to(torch.float32)
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-5) * weight[None, :].to(torch.float32)
        out[tile, :] = y.to(out.dtype)
    return out


def _one_stream_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, hidden = x.shape
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        row = x[tile, :].to(torch.float32)
        mean = torch.mean(row, dim=-1, keepdim=True)
        centered = row - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-5) * weight[None, :].to(torch.float32)
        out[tile, :] = y.to(out.dtype)
    return out


def _component_norm(
    hidden: torch.Tensor, pairs: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # ``pairs`` is [rows, channels, 2]: component 0 is a stride-2 load.
    rows, channels = hidden.shape
    out = torch.empty([rows, channels], dtype=hidden.dtype, device=hidden.device)
    for tile in hl.tile(rows):
        x = hidden[tile, :].to(torch.float32) + pairs[tile, :, 0].to(torch.float32)
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-12) * weight[None, :].to(torch.float32)
        out[tile, :] = y.to(out.dtype)
    return out


def _workspace_norm(
    residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # The summed row goes to a workspace before the reduction and comes back
    # after it, as better-benchmark's ``softmax_dropout_bwd`` reads its
    # product back.
    rows, hidden = x.shape
    workspace = torch.empty([rows, hidden], dtype=torch.float32, device=x.device)
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        summed = residual[tile, :].to(torch.float32) + x[tile, :].to(torch.float32)
        workspace[tile, :] = summed
        mean = torch.mean(summed, dim=-1, keepdim=True)
        y = (workspace[tile, :] - mean) * weight[None, :].to(torch.float32)
        out[tile, :] = y.to(out.dtype)
    return out


def _printed_norm(
    residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # ``_two_stream_norm`` with an op that has side effects and names no
    # memory: the walk for stores read back asks the host function about it.
    rows, hidden = x.shape
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        summed = residual[tile, :].to(torch.float32) + x[tile, :].to(torch.float32)
        mean = torch.mean(summed, dim=-1, keepdim=True)
        hl.device_print("weight", weight[0])
        y = (summed - mean) * weight[None, :].to(torch.float32)
        out[tile, :] = y.to(out.dtype)
    return out


def _kernel(fn: Any) -> helion.Kernel:
    return helion.kernel(fn, backend="cute", static_shapes=True, autotune_effort="none")


_BOUND: dict[object, tuple[Any, tuple[torch.Tensor, ...]]] = {}


def _bind(fn: Any, *shapes: tuple[int, ...], dtype: torch.dtype) -> Any:
    key = (fn, shapes, dtype)
    if key not in _BOUND:
        arguments = tuple(torch.empty(shape, dtype=dtype) for shape in shapes)
        _BOUND[key] = (cpu_bind(_kernel(fn), arguments), arguments)
    return _BOUND[key][0]


def _retention_seeds(bound: Any) -> list[helion.Config]:
    return [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("load_eviction_policies")
        and "l1_l2_last" in seed.config["load_eviction_policies"]
    ]


def _policies(seed: helion.Config) -> list[str]:
    return [policy for policy in seed.config["load_eviction_policies"] if policy]


# -- stream retention ----------------------------------------------------------


def test_two_stream_rows_get_a_twin_per_stream_on_two_carriers() -> None:
    # 1024 x 1024 bf16: each stream is 2 MiB.
    bound = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    spec = bound.config_spec
    with bound.env:
        twins = row_stream_retention_seed_configs(
            bound.env,
            bound.host_function.device_ir,
            [
                seed
                for seed in spec.compiler_seed_configs
                if seed not in _retention_seeds(bound)
            ],
        )
    assert len(twins) == 4
    assert CuteRowStreamRetentionHeuristic.name in spec.autotuner_heuristics
    # Two carriers, each twinned once per stream: either stream retained,
    # the other streamed, the broadcast weight left alone.
    assert [_policies(twin) for twin in twins] == [
        ["l1_l2_last", "nc_l2_first"],
        ["nc_l2_first", "l1_l2_last"],
    ] * 2
    for twin in twins:
        assert twin.config["cute_wide_packet_hints"] is True
        assert "cute_scalar_load_hints" not in twin.config
    carriers = [
        {
            key: value
            for key, value in twin.config.items()
            if key not in ("load_eviction_policies", "cute_wide_packet_hints")
        }
        for twin in twins
    ]
    assert carriers[0] == carriers[1] != carriers[2] == carriers[3]
    # The carriers are row seeds with threads and vector lanes of their own,
    # found among the seeds before the twins.
    seeds = [dict(seed.config) for seed in spec.compiler_seed_configs]
    for carrier in (carriers[0], carriers[2]):
        assert any(
            all(seed.get(key) == value for key, value in carrier.items())
            and not any(seed.get("load_eviction_policies") or [])
            for seed in seeds
        )
        assert any(width >= 2 for width in carrier["cute_vector_widths"])
        assert any(count > 0 for count in carrier["num_threads"])
    # The twins are the last seeds of the kernel, in that order.
    assert [_policies(seed) for seed in spec.compiler_seed_configs[-4:]] == [
        _policies(twin) for twin in twins
    ]


def test_a_retention_twin_renders_one_retained_and_one_streamed_packet() -> None:
    bound = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    twin = _retention_seeds(bound)[-1]
    assert _policies(twin) == ["nc_l2_first", "l1_l2_last"]
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        # The twin transfers through the search's flat layout unchanged.
        flat, effective = generation.strict_config_pair(twin)
        assert generation.strict_unflatten(flat) == effective
    code = bound.to_code(effective)
    body = code.split("def _helion_")[1].split("\ndef ")[0]
    assert body.count("_cute_load_l1_l2_evict_last") == 1
    assert "_cute_load_l1_l2_evict_last" in body.split("x.iterator")[0].splitlines()[-1]
    assert body.count("_cute_load_nc_l2_evict_first") == 1
    # Without the form knob the same policies lose the retained hint on a
    # 32-byte packet: the twins name the knob themselves.
    assert 16 in effective.config["cute_vector_widths"]
    legacy = helion.Config.from_dict(
        {
            key: value
            for key, value in effective.config.items()
            if key != "cute_wide_packet_hints"
        }
    )
    assert "_cute_load_l1_l2_evict_last" not in bound.to_code(legacy)


@pytest.mark.parametrize(
    "fn,shapes,dtype,reason",
    [
        # One stream: nothing to choose between.
        (_one_stream_norm, ((1024, 1024), (1024,)), torch.bfloat16, "one stream"),
        # Streams of 128 KiB cost a launch no bandwidth worth a seed.
        (
            _two_stream_norm,
            ((64, 1024), (64, 1024), (1024,)),
            torch.bfloat16,
            "small streams",
        ),
        # A 12 MiB stream beside a stride-2 component of a 24 MiB tensor: the
        # component loads element by element on every carrier, so one packet
        # stream is left.
        (
            _component_norm,
            ((4096, 768), (4096, 768, 2), (768,)),
            torch.float32,
            "one packet stream",
        ),
    ],
)
def test_kernels_without_two_sizable_streams_get_no_twin(
    fn: Any, shapes: tuple[tuple[int, ...], ...], dtype: torch.dtype, reason: str
) -> None:
    bound = _bind(fn, *shapes, dtype=dtype)
    assert not _retention_seeds(bound), reason
    assert (
        CuteRowStreamRetentionHeuristic.name
        not in bound.config_spec.autotuner_heuristics
    )


def test_a_stream_larger_than_l2_is_not_retained() -> None:
    bound = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    seeds = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed not in _retention_seeds(bound)
    ]
    with (
        bound.env,
        patch.object(
            cute_row_stream_retention, "_device_l2_bytes", return_value=1 << 20
        ),
    ):
        assert not row_stream_retention_seed_configs(
            bound.env, bound.host_function.device_ir, seeds
        )


def test_a_kernel_whose_seeds_name_policies_gets_no_twin() -> None:
    # Another family already chose the policies of this kernel's loads.
    bound = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    spec = bound.config_spec
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed not in _retention_seeds(bound)
    ]
    hinted = helion.Config.from_dict(
        seeds[0].config
        | {
            "load_eviction_policies": ["l1_l2_first"]
            * spec.load_eviction_policies.length
        }
    )
    with bound.env:
        device_ir = bound.host_function.device_ir
        assert len(row_stream_retention_seed_configs(bound.env, device_ir, seeds)) == 4
        assert not row_stream_retention_seed_configs(
            bound.env, device_ir, [*seeds, hinted]
        )


def test_an_invalid_seed_is_no_carrier() -> None:
    # A seed the search would repair or skip (a thread count that is no
    # power of two) is passed over; the next row seeds are twinned.
    bound = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    spec = bound.config_spec
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed not in _retention_seeds(bound)
    ]
    carrier = next(
        seed
        for seed in seeds
        if any(width >= 2 for width in seed.config.get("cute_vector_widths", []))
        and any(count > 0 for count in seed.config.get("num_threads", []))
    )
    invalid = helion.Config.from_dict(carrier.config | {"num_threads": [0, 150]})
    with bound.env:
        device_ir = bound.host_function.device_ir
        expected = row_stream_retention_seed_configs(bound.env, device_ir, seeds)
        assert (
            row_stream_retention_seed_configs(bound.env, device_ir, [invalid, *seeds])
            == expected
        )


def test_a_row_held_in_registers_is_no_carrier_where_stores_are_read_back() -> None:
    bound = _bind(
        _workspace_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    spec = bound.config_spec
    # The first row seed with threads and lanes of its own holds the row in
    # registers, and the code generator refuses it here: the barrier between
    # the workspace's store and its read has no place in the lane passes.
    held = next(
        seed
        for seed in spec.compiler_seed_configs
        if cute_row_stream_retention._is_carrier(seed)
    )
    assert held.config["reduction_loops"] == [None]
    with pytest.raises(exc.BackendUnsupported, match="lane-invariant statement"):
        bound.to_code(spec.normalized_config(held))
    # So it carries no twin: the twins are those of the two rolled rows
    # behind it, a twin per stream, and they compile.
    twins = _retention_seeds(bound)
    assert [_policies(twin)[:2] for twin in twins] == [
        ["l1_l2_last", "nc_l2_first"],
        ["nc_l2_first", "l1_l2_last"],
    ] * 2
    assert all(twin.config["reduction_loops"] == [1024] for twin in twins)
    code = bound.to_code(spec.normalized_config(twins[0]))
    assert "_cute_load_l1_l2_evict_last" in code
    # A kernel that reads back nothing keeps its twins on the rows held in
    # registers, an op with side effects or not (the seeds are read outside
    # the host function that op is classified in).
    for fn in (_two_stream_norm, _printed_norm):
        plain = _bind(fn, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16)
        assert _retention_seeds(plain)[0].config["reduction_loops"] == [None]


def test_streams_are_the_per_row_reduction_inputs() -> None:
    bound = _bind(
        _component_norm,
        (4096, 768),
        (4096, 768, 2),
        (768,),
        dtype=torch.float32,
    )
    spec = bound.config_spec
    with bound.env:
        facts = bound.host_function.device_ir.config_memory_op_facts(
            spec.default_config()
        )
    streams = row_streams(spec, facts)
    # The broadcast weight is no stream, and neither is the stride-2
    # operand: every row seed loads it element by element.
    assert list(streams) == ["hidden"]
    assert all(slots for slots in streams.values())
    square = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    with square.env:
        facts = square.host_function.device_ir.config_memory_op_facts(
            square.config_spec.default_config()
        )
    # Rows as long as the tile axis: the lane is told by its subscript, not
    # by its extent.
    assert list(row_streams(square.config_spec, facts)) == ["residual", "x"]


def test_seeds_set_only_existing_fields_at_every_shape() -> None:
    # Neither family opens a knob: the structural fingerprint of a kernel
    # the seeds fire for equals that of a neighbouring shape they skip.
    sizable = _bind(
        _two_stream_norm, (1024, 1024), (1024, 1024), (1024,), dtype=torch.bfloat16
    )
    small = _bind(
        _two_stream_norm, (64, 1024), (64, 1024), (1024,), dtype=torch.bfloat16
    )
    assert _retention_seeds(sizable) and not _retention_seeds(small)
    assert (
        sizable.config_spec.structural_fingerprint()
        == small.config_spec.structural_fingerprint()
    )


# -- lane-unrolled warp rows ---------------------------------------------------


def _warp_rows(bound: Any) -> list[helion.Config]:
    # The seeds before the family's own: it twins one of them.
    seeds = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("block_sizes") != [4]
        or seed.config.get("reduction_loops") != [None]
    ]
    with bound.env:
        return lane_unrolled_warp_rows_seed_configs(
            bound.env, bound.host_function.device_ir, seeds
        )


def _component_bound(rows: int = 4096, channels: int = 768) -> Any:
    return _bind(
        _component_norm,
        (rows, channels),
        (rows, channels, 2),
        (channels,),
        dtype=torch.float32,
    )


def test_a_masked_row_of_whole_warp_lanes_gets_four_warp_rows() -> None:
    bound = _component_bound()
    (seed,) = _warp_rows(bound)
    assert seed.config["block_sizes"] == [4]
    assert seed.config["num_threads"] == [4, 32]
    assert seed.config["reduction_loops"] == [None]
    assert seed.config["cute_reduction_reloads"] == ["register"]
    assert seed.config["cute_lane_unroll"] > 1
    spec = bound.config_spec
    assert seed in spec.compiler_seed_configs
    assert CuteLaneUnrolledWarpRowsHeuristic.name in spec.autotuner_heuristics
    normalized = spec.normalized_config(seed)
    # The row stays persistent beside four rows per CTA, and the normalized
    # config normalizes to itself.
    assert normalized.config["reduction_loops"] == [None]
    assert normalized.config["cute_lane_unroll"] == seed.config["cute_lane_unroll"]
    assert spec.normalized_config(normalized) == normalized
    code = bound.to_code(normalized)
    assert "block=(32, 4, 1)" in code
    assert "for roffset_" not in code
    assert "cute.arch.thread_idx()[1]" in code
    assert "cute.arch.warp_reduction_sum(" in code
    assert "_lane_unroll_loads_" in code


def test_without_the_unroll_the_same_layout_rolls_as_before() -> None:
    bound = _component_bound()
    spec = bound.config_spec
    (seed,) = _warp_rows(bound)
    plain = helion.Config.from_dict(seed.config | {"cute_lane_unroll": 1})
    normalized = spec.normalized_config(plain)
    assert normalized.config["reduction_loops"] == [256]
    assert "for roffset_" in bound.to_code(normalized)
    # Eight rows per CTA stay persistent the same way.
    eight = spec.normalized_config(
        helion.Config.from_dict(
            seed.config | {"block_sizes": [8], "num_threads": [8, 32]}
        )
    )
    assert eight.config["reduction_loops"] == [None]
    assert spec.normalized_config(eight) == eight
    assert "block=(32, 8, 1)" in bound.to_code(eight)


@pytest.mark.parametrize(
    "channels,reason",
    [
        # A power-of-two row is unmasked: its lanes are a vector's elements
        # and no lane loop takes the unroll.
        (1024, "power of two"),
        # 1000 = 31.25 lanes of a warp.
        (1000, "no whole lanes"),
        # 2080 = 65 lanes per thread, past what a thread keeps in registers.
        (2080, "too many lanes"),
        # A row a warp covers with one element per thread has no lane loop.
        (32, "one lane"),
    ],
)
def test_rows_one_warp_does_not_walk_in_whole_lanes_get_no_seed(
    channels: int, reason: str
) -> None:
    bound = _component_bound(rows=512, channels=channels)
    assert not _warp_rows(bound), reason
    assert (
        CuteLaneUnrolledWarpRowsHeuristic.name
        not in bound.config_spec.autotuner_heuristics
    )


@pytest.mark.parametrize("vector", [1, 2])
def test_the_admission_follows_the_lane_knob_normalization(vector: int) -> None:
    # A persistent row stays beside the wider tile axis only where it forms
    # the lane loop the unroll takes, which is where normalization keeps the
    # knob: a normalized config names both or neither, and normalizes to
    # itself.  A power-of-two row walks scalar lanes (kept) or, with a lane
    # vector, the vector's elements (no lane loop: rolled).
    bound = _component_bound(rows=512, channels=1024)
    spec = bound.config_spec
    config = helion.Config.from_dict(
        spec.default_config().config
        | {
            "block_sizes": [4],
            "num_threads": [4, 32],
            "reduction_loops": [None],
            "cute_lane_unroll": 16,
            "cute_vector_widths": [vector, 1],
        }
    )
    normalized = spec.normalized_config(config)
    assert spec.normalized_config(normalized) == normalized
    persistent = normalized.config["reduction_loops"] == [None]
    assert persistent == (vector == 1)
    if persistent:
        assert normalized.config["cute_lane_unroll"] == 16


def _row_layout(bound: Any, threads: int, rows: int = 4) -> helion.Config:
    spec = bound.config_spec
    return spec.normalized_config(
        helion.Config.from_dict(
            spec.default_config().config
            | {
                "block_sizes": [rows],
                "num_threads": [rows, threads],
                "reduction_loops": [None],
                "cute_lane_unroll": 4,
            }
        )
    )


def test_only_the_thread_count_the_row_runs_on_is_admitted() -> None:
    # A lane-loop row runs on one warp at most, so every count above it
    # would render the one-warp kernel: those roll as before, each its own
    # rolled kernel, and each admitted count is a kernel of its own.
    bound = _component_bound()
    codes = {}
    for threads in (8, 16, 32):
        config = _row_layout(bound, threads)
        assert config.config["reduction_loops"] == [None]
        assert config.config["num_threads"] == [4, threads]
        codes[threads] = bound.to_code(config)
        assert f"block=({threads}, 4, 1)" in codes[threads]
    for threads in (64, 96, 128, 256):
        config = _row_layout(bound, threads)
        assert config.config["reduction_loops"] != [None]
        codes[threads] = bound.to_code(config)
        assert "for roffset_" in codes[threads]
    assert len(set(codes.values())) == len(codes)


def test_a_symbolic_row_is_not_admitted() -> None:
    # Only a hint says a symbolic row is walked in whole lanes: it rolls as
    # it did before, where the same row with a static extent stays
    # persistent.
    arguments = (
        torch.empty((4096, 768)),
        torch.empty((4096, 768, 2)),
        torch.empty((768,)),
    )
    dynamic = cpu_bind(
        helion.kernel(
            _component_norm,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
        ),
        arguments,
    )
    assert _row_layout(dynamic, 32).config["reduction_loops"] != [None]
    assert _row_layout(_component_bound(), 32).config["reduction_loops"] == [None]
    assert not _warp_rows(dynamic)


def _twice_stored_norm(x: torch.Tensor) -> torch.Tensor:
    # The row total goes to column 0 before the scaled row overwrites it: two
    # stores of one tensor through different indices, whose threads race on
    # ``out[tile, 0]`` unless a barrier orders them.
    rows, hidden = x.shape
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        total = x[tile, :].sum(-1)
        out[tile, 0] = total
        out[tile, :] = x[tile, :] / total[:, None]
    return out


def _inplace_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # The row is read from and written to the same elements of ``x``.
    rows, hidden = x.shape
    for tile in hl.tile(rows):
        row = x[tile, :].to(torch.float32)
        mean = torch.mean(row, dim=-1, keepdim=True)
        centered = row - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-5) * weight[None, :].to(torch.float32)
        x[tile, :] = y.to(x.dtype)
    return x


def _aliased_center(x: torch.Tensor) -> torch.Tensor:
    # ``out`` is another tensor on the storage of ``x``: the row is read
    # through one and written through the other.
    out = x.view_as(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :]
        out[tile, :] = row - row.mean(-1, keepdim=True)
    return out


@pytest.mark.parametrize(
    "fn,shapes",
    [
        # The workspace row is read back after its store.
        (_workspace_norm, ((1024, 768), (1024, 768), (768,))),
        # Two stores of ``out`` meet on its first column.
        (_twice_stored_norm, ((1024, 768),)),
        # A load and a store of the same elements of ``x``.
        (_inplace_norm, ((1024, 768), (768,))),
        # A read and a write of one storage through two tensors, which the
        # lane split cannot order.
        (_aliased_center, ((1024, 768),)),
    ],
)
def test_a_kernel_whose_accesses_need_a_barrier_rolls_the_row_as_before(
    fn: Any, shapes: tuple[tuple[int, ...], ...]
) -> None:
    # Accesses that need an order across threads take a thread barrier, and
    # the code generator refuses a row held in registers on lanes there (the
    # barrier has no place in the lane passes).  Such a kernel keeps the
    # rolled row this layout was before the admission, which compiles, and
    # gets no warp-rows seed.
    bound = _bind(fn, *shapes, dtype=torch.float32)
    spec = bound.config_spec
    assert not spec.cute_register_lane_rows
    for rows in (2, 4):
        config = _row_layout(bound, 32, rows=rows)
        assert config.config["reduction_loops"] != [None]
        assert spec.normalized_config(config) == config
        assert "for roffset_" in bound.to_code(config)
    assert not _warp_rows(bound)
    assert CuteLaneUnrolledWarpRowsHeuristic.name not in spec.autotuner_heuristics


def test_a_row_written_in_place_rolls_with_a_hinted_load() -> None:
    # The barrier analysis proves that a row read and written in place stays
    # in its thread for a plain load only.  With an L1 hint on the row's
    # load it asks for a barrier, which the row held in registers has no
    # place for, so such a kernel keeps the rolled row, hint or not.
    bound = _bind(_inplace_norm, (1024, 768), (768,), dtype=torch.float32)
    spec = bound.config_spec
    base = spec.default_config().config
    slots = len(base["load_eviction_policies"])
    for policy in ("", "first"):
        config = spec.normalized_config(
            helion.Config.from_dict(
                base
                | {
                    "block_sizes": [4],
                    "num_threads": [4, 32],
                    "reduction_loops": [None],
                    "cute_lane_unroll": 4,
                    "load_eviction_policies": [policy] + [""] * (slots - 1),
                }
            )
        )
        assert config.config["reduction_loops"] != [None]
        assert "for roffset_" in bound.to_code(config)


def _center_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        row = x[tile, :]
        out[tile, :] = row - row.mean(-1, keepdim=True)
    return out


def test_arguments_on_one_buffer_roll_the_row_as_before() -> None:
    # Two arguments can be views of one buffer.  The lane split proves no
    # order between a read and a write of tensors that may share memory, so
    # the row it would refuse rolls as before; on two buffers the same
    # kernel keeps it in registers.
    buffer = torch.empty((2048, 768))
    shared = cpu_bind(_kernel(_center_into), (buffer[:1024], buffer[1024:]))
    spec = shared.config_spec
    assert not spec.cute_register_lane_rows
    config = _row_layout(shared, 32)
    assert config.config["reduction_loops"] != [None]
    assert spec.normalized_config(config) == config
    assert "for roffset_" in shared.to_code(config)
    assert not _warp_rows(shared)
    apart = cpu_bind(
        _kernel(_center_into), (torch.empty((1024, 768)), torch.empty((1024, 768)))
    )
    assert apart.config_spec.cute_register_lane_rows
    config = _row_layout(apart, 32)
    assert config.config["reduction_loops"] == [None]
    assert "for roffset_" not in apart.to_code(config)


def test_rows_whose_accesses_need_no_barrier_stay_in_registers() -> None:
    # One store per tensor and nothing read back: persistent and seeded, an
    # op with side effects in the body or not.
    shapes = ((1024, 768), (1024, 768), (768,))
    for fn in (_two_stream_norm, _printed_norm):
        plain = _bind(fn, *shapes, dtype=torch.float32)
        assert plain.config_spec.cute_register_lane_rows
        assert _row_layout(plain, 32).config["reduction_loops"] == [None]
        assert _warp_rows(plain)


def _row_scatter(x: torch.Tensor, bins: torch.Tensor, out: torch.Tensor) -> None:
    # Row totals added into ``out`` through atomics, its only accesses.
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [bins[tile]], x[tile, :].sum(-1))


def _row_scatter_and_read(
    x: torch.Tensor, bins: torch.Tensor, out: torch.Tensor
) -> None:
    # The same with a load of ``out`` beside the atomic.
    for tile in hl.tile(x.size(0)):
        total = x[tile, :].sum(-1)
        hl.atomic_add(out, [bins[tile]], total + out[bins[tile]])


def test_a_tensor_only_atomics_touch_keeps_rows_in_registers() -> None:
    # The barrier analysis orders no two atomics, so a tensor they alone
    # touch asks for no barrier; beside a load of it they may.
    arguments = (
        torch.empty((1024, 768)),
        torch.empty((1024,), dtype=torch.int64),
        torch.empty((64,)),
    )
    scatter = cpu_bind(_kernel(_row_scatter), arguments)
    assert scatter.config_spec.cute_register_lane_rows
    config = _row_layout(scatter, 32)
    assert config.config["reduction_loops"] == [None]
    assert "for roffset_" not in scatter.to_code(config)
    read = cpu_bind(_kernel(_row_scatter_and_read), arguments)
    assert not read.config_spec.cute_register_lane_rows
    assert _row_layout(read, 32).config["reduction_loops"] != [None]


# -- GPU numerics --------------------------------------------------------------


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0), reason="L2 cache policy loads need sm_90+")
@pytest.mark.parametrize(
    "rows,cta_rows,policies",
    [
        # 37 rows leave a partial last CTA of one row; 64 fill every CTA.
        (37, 4, ["", "", "", ""]),
        (64, 4, ["l1_l2_first", "l1_l2_last", "last", "last"]),
        (130, 4, ["", "l2_last", "", ""]),
        # Eight rows per CTA, the last CTA holding five.
        (37, 8, ["", "l1_l2_last", "", ""]),
    ],
)
def test_warp_rows_match_eager(rows: int, cta_rows: int, policies: list[str]) -> None:
    channels = 768
    generator = torch.Generator().manual_seed(rows)
    # Views with a storage offset: the rows start 3 rows into their buffers
    # (sliced on the device; a transfer of the slice would copy it to the
    # start of a new buffer).
    hidden = torch.randn((rows + 3, channels), generator=generator).to(DEVICE)[3:]
    pairs = torch.randn((rows + 3, channels, 2), generator=generator).to(DEVICE)[3:]
    assert hidden.storage_offset() == 3 * channels
    assert pairs.storage_offset() == 3 * channels * 2
    weight = torch.randn((channels,), generator=generator).to(DEVICE)
    bound = _kernel(_component_norm).bind((hidden, pairs, weight))
    spec = bound.config_spec
    (seed,) = _warp_rows(bound)
    slots = spec.load_eviction_policies.length
    layout = seed.config | {"block_sizes": [cta_rows], "num_threads": [cta_rows, 32]}
    config = spec.normalized_config(
        helion.Config.from_dict(
            layout
            | {
                "load_eviction_policies": (policies + [""] * slots)[:slots],
                "cute_scalar_load_hints": True,
            }
        )
    )
    code = bound.to_code(config)
    assert f"block=(32, {cta_rows}, 1)" in code and "for roffset_" not in code
    actual = bound.compile_config(config)(hidden, pairs, weight)
    x = hidden + pairs[:, :, 0]
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    expected = centered * torch.rsqrt(var + 1e-12) * weight[None, :]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # The hinted and the plain kernel sum in the same order: same bits.
    plain = spec.normalized_config(helion.Config.from_dict(layout))
    assert torch.equal(actual, bound.compile_config(plain)(hidden, pairs, weight))
