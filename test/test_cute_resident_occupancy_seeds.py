"""Resident-reduction seeds at an occupancy-filling coarse tile for small row counts."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from test.test_cute_resident_serial_row_seeds import _bind
from test.test_cute_resident_serial_row_seeds import _cpu_only  # noqa: F401

from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    _occupancy_coarse,
)
from helion._compiler.autotuner_heuristics.cute_resident_reductions import _seed_vector
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

_UNITS = (
    "helion._compiler.autotuner_heuristics.cute_resident_reductions.num_compute_units"
)


def _geometry(config: object) -> tuple[int, int, str, int]:
    values = config.config  # type: ignore[attr-defined]
    return (
        values["block_sizes"][0],
        max(values["num_threads"]),
        values["cute_reduction_schedule"],
        values["cute_reduction_group_rows"],
    )


def test_occupancy_coarse_fills_the_machine_only_when_the_lattice_does_not() -> None:
    with patch(_UNITS, return_value=148):
        # 256 rows: 32 per CTA is 8 CTAs; one CTA per SM wants 2 rows.
        assert _occupancy_coarse(256, 32) == 2
        assert _occupancy_coarse(256, 16) == 2
        # 1024 rows: 7 per SM rounds up to 8, still below 32.
        assert _occupancy_coarse(1024, 32) == 8
        # 4096 rows already keep 128 CTAs busy at 32 rows; nothing to add.
        assert _occupancy_coarse(4096, 32) is None
        assert _occupancy_coarse(0, 32) is None
    with patch(_UNITS, return_value=1):
        assert _occupancy_coarse(256, 32) is None


def test_seed_vector_prefers_the_lattice_width_then_the_exact_quotient() -> None:
    assert _seed_vector(1024, 128, 4) == 4
    assert _seed_vector(1024, 256, 8) == 4
    assert _seed_vector(2048, 256, 8) == 8
    # Two features per thread is below the 16-byte seed vector; the lattice
    # left such feature counts alone and so does the occupancy variant.
    assert _seed_vector(1024, 512, 8) is None
    assert _seed_vector(256, 128, 4) is None
    assert _seed_vector(1000, 256, 8) is None


@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_small_row_counts_add_occupancy_seeds_after_the_lattice(kind: str) -> None:
    bound, _args = _bind(256, 1024, kind=kind)
    assert bound.host_function is not None
    with bound.env, patch(_UNITS, return_value=148):
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds is not None
    geometries = [_geometry(seed) for seed in seeds]
    # The lattice primary keeps its rank; 1024 features admit only its
    # 128-thread entry with the fixed vector widths.
    assert geometries[0] == (32, 128, "pipelined", 1)
    assert all(coarse == 32 for coarse, *_rest in geometries[:1])
    occupancy = [geometry for geometry in geometries if geometry[0] == 2]
    assert (2, 128, "pipelined", 1) in occupancy
    # 256 threads over 1024 features take the exact 4-wide vector the
    # lattice's 8 does not allow, for both schedules.
    assert (2, 256, "pipelined", 2) in occupancy
    assert (2, 256, "resident", 1) in occupancy
    assert not any(threads == 512 for _coarse, threads, *_rest in occupancy)
    for seed in seeds:
        values = seed.config
        if values["block_sizes"][0] == 2:
            # The reduction axis carries both the threads and the vector; the
            # lattice's 128 x 4 covers the features in two passes.
            assert (
                1024 % (max(values["cute_vector_widths"]) * max(values["num_threads"]))
                == 0
            )


def test_large_row_counts_keep_the_lattice_only() -> None:
    bound, _args = _bind(16384, 1024, kind="rms")
    assert bound.host_function is not None
    with bound.env, patch(_UNITS, return_value=148):
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds is not None
    assert {seed.config["block_sizes"][0] for seed in seeds} == {32}


def test_pipeline_depth_seeds_keep_an_occupancy_coarse_tile() -> None:
    bound, _args = _bind(256, 1024, kind="layer")
    assert bound.host_function is not None
    with bound.env, patch(_UNITS, return_value=148):
        deeper = CuteResidentReductionHeuristic.pipeline_depth_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    # The occupancy seed tiles 2 rows; its four-deep ring wants four, and the
    # four-row tile is also offered as one resident group.
    coarse_tiles = {seed.config["block_sizes"][0] for seed in deeper}
    assert coarse_tiles == {16, 4}
    assert all(seed.config["cute_reduction_pipeline_depth"] == 4 for seed in deeper)
    groups = {
        (seed.config["block_sizes"][0], seed.config["cute_reduction_group_rows"])
        for seed in deeper
    }
    assert groups == {(16, 1), (4, 1), (4, 4)}
