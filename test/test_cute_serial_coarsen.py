from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
from ._serial_lane_model import run
from .test_cute_serial_lane import _module
import helion
from helion import exc
from helion._compiler.cute.serial_lane_coarsen import KEY
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute import serial_lane_guard

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _recurrence(states, coefficient, out):
    batch, steps, rows, features = states.shape
    for bi, pi, ni in hl.tile([batch, rows, features], block_size=[1, None, None]):
        carry = hl.full([pi, ni], 0.25, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, pi, ni] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, pi, ni].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _dependent(states, coefficient, out):
    batch, steps, rows, features = states.shape
    for bi, pi, ni in hl.tile([batch, rows, features], block_size=[1, None, None]):
        carry = hl.zeros([pi, ni], dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, pi, ni] = carry.to(out.dtype)
            carry = (
                torch.exp(coefficient[bi.begin, qi, pi.begin].float()) * carry
                + states[bi.begin, qi, pi, ni]
            )
    return out


def args(dtype=torch.bfloat16, *, rows=8, steps=5, features=32, stride=1):
    generator = torch.Generator().manual_seed(893)
    states = torch.randn((2, steps, rows, features * stride), generator=generator)[
        ..., ::stride
    ]
    coefficient = -torch.rand((2, steps), generator=generator)
    out = torch.empty((2, steps, rows, features * stride), dtype=dtype)[..., ::stride]
    return states, coefficient, out


def config(factor=2, schedule="prefetch4", features=32, order=None):
    return helion.Config.from_dict(
        {
            "block_sizes": [4, features],
            "num_threads": [4, features // 4],
            "cute_vector_widths": [1, 1, 4, 1],
            "loop_orders": [[0, 2, 1] if order is None else order],
            "cute_serial_lane_schedule": "step_major_vector",
            "cute_serial_lane_load_schedule": schedule,
            KEY: factor,
        }
    )


def code(values=None, factor=2, schedule="prefetch4", *, order=None):
    values = args() if values is None else values
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(values)
        raw = config(factor, schedule, values[0].shape[-1], order)
        canonical = bound._normalized_config_copy(raw)
        source = bound.to_code(raw)
        assert source == bound.to_code(canonical)
        return source


def test_smoke_and_factor1_omission():
    old = code(factor=1)
    with _cpu_codegen():
        values = args()
        bound = _recurrence._bind_isolated(values)
        raw = config(1)
        raw.config.pop(KEY)
        assert bound.to_code(raw) == old
    new = code()
    assert "serial_pair_pid" in new and "_bank1" in new
    a, b = (ast.parse(s) for s in (old, new))
    assert ast.dump(
        next(n for n in a.body if isinstance(n, ast.FunctionDef))
    ) == ast.dump(next(n for n in b.body if isinstance(n, ast.FunctionDef)))


@pytest.mark.parametrize("schedule", ["prefetch2", "prefetch4"])
@pytest.mark.parametrize("steps", [1, 3, 5, 16])
def test_actual_ast_math_bijection_and_shared_coefficient(schedule, steps):
    values = args(steps=steps)
    old, old_count = run(code(values, 1, schedule), values, reordered=False)
    new, count = run(code(values, 2, schedule), values, reordered=True)
    assert torch.equal(old, new)
    assert count["state_reads"] == old_count["state_reads"] == values[0].numel()
    assert count["coefficient_reads"] * 8 == old_count["coefficient_reads"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("stride", [1, 2])
def test_output_precision_and_scalar_copy_fallback(dtype, stride):
    values = args(dtype, steps=3, stride=stride)
    old, _ = run(code(values, 1), values, reordered=False)
    new, counts = run(code(values), values, reordered=True)
    assert torch.equal(old, new)
    if stride == 2:
        assert counts["vector_load"] == counts["vector_store"] == 0


@pytest.mark.parametrize("value", [True, False, None, 0, 3, 1.0, "2"])
@pytest.mark.parametrize("repair", [False, True])
def test_bad_value_never_repairs(value, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args())
        raw = config().config | {KEY: value}
        with pytest.raises(exc.InvalidConfig, match="coarsen"):
            bound.config_spec.normalize(raw, _fix_invalid=repair)


@pytest.mark.parametrize("schedule", ["current", "group2", "group4"])
@pytest.mark.parametrize("repair", [False, True])
def test_missing_prefetch_never_repairs(schedule, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args())
        with pytest.raises(exc.InvalidConfig, match="coarsen"):
            bound.config_spec.normalize(config(schedule=schedule), _fix_invalid=repair)


@pytest.mark.parametrize("rows", [4, 12, 9])
def test_partial_or_unpaired_rows_rejected(rows):
    with pytest.raises(exc.BackendUnsupported, match="serial lane"):
        code(args(rows=rows))


def test_coefficient_dependent_axis_rejected():
    values = args()
    with _cpu_codegen():
        bound = _dependent._bind_isolated((values[0], torch.zeros(2, 5, 8), values[2]))
        with pytest.raises(exc.BackendUnsupported, match="serial lane"):
            bound.to_code(config())


def test_current_alias_grid_and_rebinding():
    source = code()
    calls = []
    module = _module(source, calls)
    values = args()
    alias = (
        values[0]
        .view(torch.bfloat16)
        .flatten()[: values[2].numel()]
        .view(values[2].shape)
    )
    with patch.object(serial_lane_guard, "_require_cuda"):
        module._recurrence(*values)
        module._recurrence(values[0], values[1], alias)
        module._recurrence(*(x.clone() for x in values))
    assert [fn.__name__.endswith("_serial_lane") for fn, *_ in calls] == [
        True,
        False,
        True,
    ]
    assert [grid for _, grid, *_ in calls] == [(2,), (4,), (2,)]


def test_paired_axis_not_outermost_grid():
    values = args(steps=3, features=64)
    old, _ = run(code(values, 1, order=[2, 1, 0]), values, reordered=False)
    new, _ = run(code(values, 2, order=[2, 1, 0]), values, reordered=True)
    assert torch.equal(old, new)


def test_first100_and_whole_old_seed_order():
    from torch._inductor.codecache import PyCodeCache

    from helion._compiler.autotuner_heuristics.serial_lane import serial_lane_seeds
    from helion.autotuner.base_search import PopulationBasedSearch
    from helion.autotuner.config_generation import ConfigGeneration

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args(features=64))
        assert bound.host_function is not None
        with bound.env:
            seeds = serial_lane_seeds(bound.env, bound.host_function.device_ir)
            old = [s for s in seeds if KEY not in s.config]
            assert len(old) == 12 and seeds[:12] == old and len(seeds) == 14
            for seed, parent in zip(seeds[12:], (old[5], old[7]), strict=True):
                assert seed.config == parent.config | {KEY: 2}
            assert KEY not in bound.config_spec.default_config().config
            generation = ConfigGeneration(bound.config_spec)
            flats = generation.random_population_flat(100)
            members = [
                PopulationBasedSearch.make_unbenchmarked(
                    cast(
                        "PopulationBasedSearch", SimpleNamespace(config_gen=generation)
                    ),
                    flat,
                )
                for flat in flats
            ]
            assert len(members) == 100
        admitted = [
            m.config for m in members if m is not None and m.config.config.get(KEY) == 2
        ]
        assert {p.config["cute_serial_lane_load_schedule"] for p in admitted} == {
            "prefetch2",
            "prefetch4",
        }
        for raw in admitted:
            canonical = bound._normalized_config_copy(raw)
            source = bound.to_code(raw)
            assert bound.to_code(canonical) == source

            class Stop(BaseException):
                pass

            with patch.object(PyCodeCache, "load", side_effect=Stop):
                for selected in (raw, canonical):
                    with pytest.raises(Stop):
                        bound.compile_config(selected, allow_print=False)


def test_width2_rejected_without_relaxing_vector_parent():
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args())
        raw = config().config | {
            "num_threads": [4, 16],
            "cute_vector_widths": [1, 1, 2, 1],
        }
        with pytest.raises(exc.BackendUnsupported, match="serial lane"):
            bound.to_code(helion.Config.from_dict(raw))


def test_user_seed_and_explicit_off_priority():
    from helion.autotuner.config_generation import ConfigGeneration

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args(features=64))
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            user = config(features=64).config | {"loop_orders": [[2, 1, 0]]}
            selected = helion.Config.from_dict(user)
            population = generation.random_population_flat(
                100, user_seed_configs=[selected]
            )
            assert len(population) == 100
            assert (
                generation.unflatten(population[0])
                == bound.config_spec.default_config()
            )
            assert population[1] == generation.flatten(selected)
            disabled = ConfigGeneration(bound.config_spec, overrides={KEY: 1})
            assert all(
                KEY not in candidate.config
                for _, candidate in disabled.seed_flat_config_pairs()
            )


def test_pattern_neighbors_retain_coarsening_across_other_knobs():
    from helion.autotuner.config_generation import ConfigGeneration

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args(features=64))
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            base = config(features=64)
            neighbors = generation.coordinate_neighbor_projections(
                generation.flatten(base)
            )
        admitted = set()
        rejected = set()
        for neighbor in neighbors:
            candidate = neighbor.config
            if (
                neighbor.outcome != "candidate"
                or candidate is None
                or candidate.config.get(KEY) != 2
                or neighbor.key == KEY
            ):
                continue
            try:
                source = bound.to_code(candidate)
            except (exc.InvalidConfig, exc.BackendUnsupported):
                rejected.add(neighbor.key)
            else:
                assert source == bound.to_code(bound._normalized_config_copy(candidate))
                admitted.add(neighbor.key)
        assert "block_sizes" in admitted
        assert "num_threads" in admitted
        assert len(admitted) >= 3
        # Admission failures remain visible; no fix-invalid repair is involved.
        assert rejected
