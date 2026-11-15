from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.autotune.heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._compiler.cute.resident_reductions import proven_resident_tensor_alignments
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_function import TensorArg
from helion._testing import import_example
from helion._testing.binding_state import with_binding_state
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec

pytestmark = pytest.mark.cpu_only

layer_norm_bwd = import_example("layer_norm").layer_norm_bwd
rms_norm_bwd = import_example("rms_norm").rms_norm_bwd


def _kernel_and_arguments(
    kind: str, dtype: torch.dtype, *, static_shapes: bool = False, hidden: int = 4096
) -> tuple[helion.Kernel, tuple[object, ...]]:
    x = torch.empty((64, hidden), dtype=dtype)
    dy = torch.empty_like(x)
    weight = torch.empty(hidden, dtype=dtype)
    if kind == "layer_norm":
        function = layer_norm_bwd.fn
        args = (dy, x, torch.empty(64), torch.empty(64), weight, True)
    else:
        function = rms_norm_bwd.fn
        args = (dy, x, weight, torch.empty((64, 1)))
    return helion.kernel(
        function,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ), args


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_original_normalization_seeds_survive_search_and_emit_resident_pipeline(
    kind: str, dtype: torch.dtype, static_shapes: bool
) -> None:
    kernel, args = _kernel_and_arguments(kind, dtype, static_shapes=static_shapes)
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(kernel, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        assert CuteResidentReductionHeuristic in get_heuristics("cute")
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert seeds and all(seed in spec.compiler_seed_configs for seed in seeds)
        geometries = {
            (
                tuple(seed.block_sizes),
                max(seed.num_threads),
                seed.config["cute_reduction_group_rows"],
                seed.config["cute_reduction_schedule"],
            )
            for seed in seeds
        }
        lattice = {
            ((32, 1), 256, 2, "pipelined"),
            ((32, 1), 256, 1, "resident"),
            ((32, 1), 128, 1, "pipelined"),
            ((16, 1), 512, 2, "pipelined"),
            ((32, 1), 512, 4, "pipelined"),
        }
        assert {g for g in geometries if g[0][0] >= 16} == lattice
        # 64 rows leave most SMs idle at 16-32 rows per CTA: the same thread /
        # row-group / schedule choices are repeated at an occupancy-filling
        # coarse tile (one row per CTA for a 128-SM device).
        occupancy = {g for g in geometries if g[0][0] < 16}
        assert occupancy
        assert {g[1:] for g in occupancy} <= {g[1:] for g in lattice}
        assert {g[0] for g in occupancy} == {(1, 1)}
        generation = ConfigGeneration(spec)
        surviving = [config for _flat, config in generation.seed_flat_config_pairs()]
        for seed in seeds:
            normalized = spec.normalized_config(seed)
            flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
            assert roundtrip in surviving
            assert generation.unflatten(flat) == roundtrip
            for key in (
                "block_sizes",
                "num_threads",
                "cute_vector_widths",
                "cute_reduction_schedule",
                "cute_reduction_group_rows",
                "cute_cluster_n",
                "cute_lane_layouts",
                "cute_reduction_reloads",
            ):
                assert roundtrip.config[key] == normalized.config[key]
            assert roundtrip.num_threads == seed.num_threads
            code = bound.to_code(roundtrip)
            assert "resident_state" in code
            assert "_cute_resident_sums" in code
            assert "_helion_lane_reduce" not in code
            threads = max(seed.num_threads)
            assert f"block=({threads}, 1, 1)" in code
            if seed.config["cute_reduction_schedule"] == "pipelined":
                assert "_cute_resident_copy_async" in code
                assert "cute.arch.cp_async_wait_group(1)" in code
                assert "cute.arch.cp_async_wait_group(0)" in code
            else:
                assert "_cute_resident_load_vector" in code
                assert "_cute_resident_copy_async" not in code


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_narrow_features_seed_one_8_wide_vector_per_thread(kind: str) -> None:
    # No lattice entry tiles 256 features: every seed carries one 8-wide
    # vector per thread (32 threads), over four- and two-row groups at the
    # lattice's coarse tile and over one row at the occupancy-filling tile of
    # one row that 64 rows get (a larger group would only stage masked rows).
    kernel, args = _kernel_and_arguments(kind, torch.bfloat16, hidden=256)
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(kernel, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert seeds and all(seed in spec.compiler_seed_configs for seed in seeds)
        assert {
            (
                tuple(seed.block_sizes),
                max(seed.num_threads),
                max(seed.config["cute_vector_widths"]),
                seed.config["cute_reduction_group_rows"],
                seed.config["cute_reduction_schedule"],
            )
            for seed in seeds
        } == {
            ((32, 1), 32, 8, 4, "pipelined"),
            ((32, 1), 32, 8, 2, "pipelined"),
            ((1, 1), 32, 8, 1, "pipelined"),
        }
        # The depth seeds deepen the lattice's one-row pipeline seeds, and
        # the lattice tiles nothing here: the 8-wide one-row seed keeps its
        # prefetch.
        assert not CuteResidentReductionHeuristic.pipeline_depth_seed_configs(
            bound.env, host.device_ir
        )
        generation = ConfigGeneration(spec)
        surviving = [config for _flat, config in generation.seed_flat_config_pairs()]
        for seed in seeds:
            _flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
            assert roundtrip in surviving
            assert (
                roundtrip.config["cute_vector_widths"]
                == (spec.normalized_config(seed).config["cute_vector_widths"])
            )
            code = bound.to_code(roundtrip)
            assert "resident_state" in code
            assert "_cute_resident_sums" in code
            assert "_cute_resident_copy_async" in code


def _carrier_twins(spec: ConfigSpec) -> dict[tuple[str, bool], helion.Config]:
    """The serial output carrier's row-schedule twins among the compiler
    seeds (the carrier itself is never one), by row schedule and packing."""
    twins = {
        (
            seed.config["cute_reduction_row_schedule"],
            seed.config.get("cute_reduction_pack_output", False),
        ): seed
        for seed in spec.compiler_seed_configs
        if "cute_reduction_row_schedule" in seed.config
    }
    assert set(twins) == {
        ("serial", False),
        ("serial_deferred", False),
        ("serial_deferred", True),
    }
    return twins


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.FastMathUnused],
)
def _split_rows_rms_norm_bwd(
    grad: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # The rows of each coarse index lie in a dim of their own:
    # ``x[tile_s, tile_r, :]`` under ``hl.tile(rows)``, not under
    # ``hl.tile(tile_s.begin, tile_s.end)`` as in the examples.
    splits, rows, hidden = x.shape
    grad_input = torch.empty_like(x)
    weight_partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        g = grad[()].to(torch.float32)
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            xf = x[tile_s, tile_r, :].to(torch.float32)
            r = rstd[tile_s, tile_r][:, :, None]
            gw = g * weight[None, None, :]
            row_dot = torch.sum(gw * xf, dim=-1, keepdim=True)
            scale = row_dot * -0.5 * r**3 / hidden
            grad_input[tile_s, tile_r, :] = (gw * r + scale * (xf * 2.0)).to(x.dtype)
            acc = acc + torch.sum(g * (xf * r), dim=1)
        weight_partial[tile_s, :] = acc
    return grad_input, weight_partial


def test_rows_in_their_own_dim_seed_one_coarse_index_per_cta() -> None:
    # A coarse tile wider than one index would wrap the row loop in a lane
    # loop over the coarse dim, which the resident lowering refuses: every
    # seed takes one coarse index per CTA, and the search reaches that tile
    # although 32768 coarse indices give it an occupancy floor above one
    # (``raise_grid_block_minimums``).
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(512),
        torch.empty((splits, 4)),
        torch.empty((splits, 4, 512), dtype=torch.bfloat16),
    )
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(_split_rows_rms_norm_bwd, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        coarse = spec.block_sizes.block_id_lookup(0)
        assert coarse._fragment(spec).low == 1
        heuristic = CuteResidentReductionHeuristic
        base = heuristic.get_seed_configs(bound.env, host.device_ir) or []
        depth = heuristic.pipeline_depth_seed_configs(bound.env, host.device_ir)
        assert depth
        # 512 features: the lattice's 128 threads of 4, and one 8-wide vector
        # per thread over four- and two-row groups.
        vector_rows = {
            seed.config["cute_reduction_group_rows"]: seed
            for seed in base
            if max(seed.num_threads) == 64
            and max(seed.config["cute_vector_widths"]) == 8
        }
        assert set(vector_rows) == {4, 2}
        twins = _carrier_twins(spec)
        generation = ConfigGeneration(spec)
        surviving = [config for _flat, config in generation.seed_flat_config_pairs()]
        for seed in (*base, *depth, *twins.values()):
            assert seed.block_sizes == [1, 1]
            normalized = spec.normalized_config(seed)
            flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
            assert generation.unflatten(flat) == roundtrip
            for key in (
                "block_sizes",
                "num_threads",
                "cute_vector_widths",
                "cute_reduction_schedule",
                "cute_reduction_group_rows",
                "cute_reduction_pipeline_depth",
            ):
                if key in normalized.config:
                    assert roundtrip.config[key] == normalized.config[key]
            assert roundtrip in surviving
        # One render per family: the 8-wide four-row seed, a depth seed and
        # the carrier's unpacked twins.
        for seed in (
            vector_rows[4],
            depth[0],
            twins["serial", False],
            twins["serial_deferred", False],
        ):
            code = bound.to_code(spec.normalized_config(seed))
            assert "resident_state" in code
            assert "_cute_resident_sums" in code
        # The packed twin pairs the row output's terminal FP32 product, and
        # this RMSNorm's output ends in a sum: the lowering refuses the twin,
        # as it does the examples' rms_norm_bwd's (the LayerNorm form below
        # packs at the same coarse tile).
        with pytest.raises(
            helion.exc.BackendUnsupported, match="independent FP32 carries"
        ):
            bound.to_code(spec.normalized_config(twins["serial_deferred", True]))
        # 32768 coarse indices give the grid an occupancy floor above one,
        # which the plan lowered: re-applying it raises the floor again.
        floor = coarse.autotuner_min
        try:
            spec.raise_grid_block_minimums()
            assert coarse._fragment(spec).low > 1
            # The plan undoes only that floor: one another surface raised
            # since stays, and no seed family is probed for it.
            occupancy = coarse.autotuner_min
            coarse.autotuner_min = 2 * occupancy
            with patch.object(
                heuristic, "_seeds_resident", side_effect=AssertionError("probed")
            ):
                heuristic.register_facts(bound.env, host.device_ir)
            assert coarse.autotuner_min == 2 * occupancy
            coarse.autotuner_min = occupancy
            heuristic.register_facts(bound.env, host.device_ir)
            assert coarse.autotuner_min == 1
        finally:
            coarse.autotuner_min = floor


def test_rows_in_their_own_dim_keep_the_floor_without_a_resident_seed() -> None:
    # 128 features: no lattice or 8-wide entry tiles them, the carrier wants
    # a multiple of 512 and the serial-row family does not lower resident in
    # this form, so no resident seed takes one coarse index per CTA, and the
    # coarse block keeps the occupancy floor its 32768 indices give it.
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(128),
        torch.empty((splits, 4)),
        torch.empty((splits, 4, 128), dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(_split_rows_rms_norm_bwd, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        heuristic = CuteResidentReductionHeuristic
        assert heuristic.get_seed_configs(bound.env, host.device_ir) is None
        assert heuristic.serial_row_seed_configs(bound.env, host.device_ir) == []
        assert heuristic.serial_output_carrier(bound.env, host.device_ir) is None
        coarse = spec.block_sizes.block_id_lookup(0)
        assert coarse.occupancy_floor is not None
        assert coarse._fragment(spec).low == coarse.occupancy_floor[1] > 1


def test_rows_in_their_own_dim_keep_the_floor_with_heuristics_disabled() -> None:
    # No compiler seed is planted with autotuner heuristics disabled: the
    # coarse block keeps the occupancy floor that 32768 indices give it,
    # although the resident families would seed one index per CTA.
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(512),
        torch.empty((splits, 4)),
        torch.empty((splits, 4, 512), dtype=torch.bfloat16),
    )
    kernel = helion.kernel(
        _split_rows_rms_norm_bwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.FastMathUnused],
        disable_autotuner_heuristics=True,
    )
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(kernel, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        assert spec.compiler_seed_configs == []
        assert CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        coarse = spec.block_sizes.block_id_lookup(0)
        assert coarse._fragment(spec).low > 1


def test_rows_in_their_own_dim_keep_the_floor_for_one_row_splits() -> None:
    # A split of one row matches the plan and gets one-index-per-CTA seeds,
    # but the lowering proves none of their carries ("could not prove
    # rectangular independent FP32 carries"): the coarse block keeps the
    # occupancy floor its 32768 indices give it.
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(512),
        torch.empty((splits, 1)),
        torch.empty((splits, 1, 512), dtype=torch.bfloat16),
    )
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(_split_rows_rms_norm_bwd, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        assert CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        coarse = spec.block_sizes.block_id_lookup(0)
        assert coarse.occupancy_floor is not None
        assert coarse._fragment(spec).low == coarse.occupancy_floor[1] > 1


def test_rows_in_their_own_dim_seed_one_row_groups_for_empty_splits() -> None:
    # Splits of no rows leave each CTA's row loop nothing to run: the 8-wide
    # entries' row groups stay at one row, a group the lowering accepts.
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(256),
        torch.empty((splits, 0)),
        torch.empty((splits, 0, 256), dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(_split_rows_rms_norm_bwd, args)
        host = bound.host_function
        assert host is not None
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert seeds
        for seed in seeds:
            assert seed.config["cute_reduction_group_rows"] == 1
            bound.config_spec.normalized_config(seed)


def test_rows_in_their_own_dim_fit_the_8_wide_groups_to_a_split() -> None:
    # Each CTA's row loop runs the two rows of its one split: the 8-wide
    # entries' four-row group shrinks to two (more would only stage masked
    # rows), leaving one 8-wide seed.
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(512),
        torch.empty((splits, 2)),
        torch.empty((splits, 2, 512), dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(_split_rows_rms_norm_bwd, args)
        host = bound.host_function
        assert host is not None
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert [
            seed.config["cute_reduction_group_rows"]
            for seed in seeds or ()
            if max(seed.config["cute_vector_widths"]) == 8
        ] == [2]


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.FastMathUnused],
)
def _split_rows_layer_norm_bwd(
    grad: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    row_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # better-benchmark's LayerNorm backward port, in the split-row form of
    # _split_rows_rms_norm_bwd; its row output ends in an FP32 product.
    splits, rows, hidden = x.shape
    grad_input = torch.empty_like(x)
    weight_partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        g = grad[()].to(torch.float32)
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            gw = g * weight[None, None, :]
            r = rstd[tile_s, tile_r][:, :, None]
            x_hat = (
                x[tile_s, tile_r, :].to(torch.float32)
                - mean[tile_s, tile_r][:, :, None]
            ) * r
            row_dot = torch.sum(gw * x_hat, dim=-1, keepdim=True)
            centered = gw * hidden - row_sum[tile_s, tile_r][:, :, None]
            grad_input[tile_s, tile_r, :] = (
                (r / hidden) * (centered - x_hat * row_dot)
            ).to(x.dtype)
            acc = acc + torch.sum(g * x_hat, dim=1)
        weight_partial[tile_s, :] = acc
    return grad_input, weight_partial


def test_rows_in_their_own_dim_pack_a_terminal_product() -> None:
    # Every row-schedule twin of the carrier takes one coarse index per CTA
    # and emits resident code, the packed one too when the row output ends in
    # an FP32 product.
    splits = 32768
    args = (
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(512),
        torch.empty((splits, 4, 512), dtype=torch.bfloat16),
        torch.empty((splits, 4)),
        torch.empty((splits, 4)),
        torch.empty((splits, 4)),
    )
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(_split_rows_layer_norm_bwd, args)
        spec = bound.config_spec
        for (_schedule, packed), seed in _carrier_twins(spec).items():
            assert seed.block_sizes == [1, 1]
            code = bound.to_code(spec.normalized_config(seed))
            assert "resident_state" in code
            assert "_cute_resident_sums" in code
            assert ("cute.arch.mul_packed_f32x2" in code) is packed


def test_resident_knobs_reject_unproved_config_values() -> None:
    kernel, args = _kernel_and_arguments("layer_norm", torch.bfloat16)
    bound = cpu_bind(kernel, args)
    host = bound.host_function
    assert host is not None
    seeds = CuteResidentReductionHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds
    for values in (
        {"cute_reduction_schedule": "unknown"},
        {"cute_reduction_group_rows": True},
        {"cute_reduction_group_rows": 5},
    ):
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalized_config(
                helion.Config.from_dict(seeds[0].config | values)
            )


@pytest.mark.parametrize("offset,alignment", [(0, 16), (1, 2), (2, 4), (4, 8), (8, 16)])
def test_fresh_allocation_view_alignment_accounts_for_storage_offset(
    offset: int, alignment: int
) -> None:
    tensor = torch.empty(32, dtype=torch.float16)[offset:]
    function = cast(
        "DeviceFunction", SimpleNamespace(arguments=[TensorArg("view", tensor, None)])
    )
    env = with_binding_state(
        SimpleNamespace(
            input_sources={}, compiler_fact_specialization_facts=frozenset()
        )
    )
    with patch.object(CompileEnvironment, "current", return_value=env):
        assert proven_resident_tensor_alignments(function) == {"view": alignment}
