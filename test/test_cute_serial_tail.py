from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
from ._serial_lane_model import run
from .test_cute_host_fastpath import _pair
from .test_cute_host_fastpath import chain_args
from .test_cute_serial_coarsen import _recurrence
from .test_cute_serial_coarsen import args
from .test_cute_serial_coarsen import config
from .test_cute_serial_lane import _module
from helion import exc
from helion._compiler.cute.host_fastpath import KEY as FAST
from helion._compiler.cute.serial_lane_recurrence import TAIL_KEY as KEY
from helion._testing import skipUnlessBackends
from helion.runtime.cute import serial_lane_guard

pytestmark = skipUnlessBackends(["cute"])


def source(values, depth=4, factor=1, tail="peel_final_group", **extra):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(values)
        raw = config(factor, f"prefetch{depth}", values[0].shape[-1])
        if tail is not None:
            raw.config[KEY] = tail
        raw.config.update(extra)
        canonical = bound._normalized_config_copy(raw)
        result = bound.to_code(raw)
        assert result == bound.to_code(canonical)
        return result


@pytest.mark.parametrize("depth,steps", [(2, 2), (2, 6), (4, 4), (4, 12)])
@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_actual_ast_every_store_and_scalar_fallback(depth, steps, factor, stride):
    values = args(steps=steps, stride=stride)
    old, before = run(source(values, depth, 1, None), values, reordered=False)
    new, counts = run(source(values, depth, factor), values, reordered=True)
    assert torch.equal(old, new)
    assert counts["state_reads"] == before["state_reads"] == values[0].numel()
    assert counts["coefficient_reads"] * (4 * factor) == before["coefficient_reads"]
    if stride == 2:
        assert counts["vector_load"] == counts["vector_store"] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_precision_extremes(dtype):
    values = args(dtype, steps=4)
    values[1][0] = torch.tensor([-1000.0, 1000.0, -float("inf"), float("inf")])
    old, _ = run(source(values, tail=None), values, reordered=False)
    new, _ = run(source(values), values, reordered=True)
    torch.testing.assert_close(new, old, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("depth,steps", [(2, 2), (2, 6), (4, 4), (4, 12)])
def test_structure_final_stores_no_final_update_or_future_pointer(depth, steps):
    values = args(steps=steps)
    text = source(values, depth)
    module = ast.parse(text)
    variant = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name.endswith("_serial_lane")
    )
    # No remaining serial future-bound guard; memory alignment/range fallbacks
    # remain unchanged. Final group indices are literal, not dynamic selectors.
    assert not any(
        isinstance(n, ast.If) and "serial_future" in ast.unparse(n.test)
        for n in ast.walk(variant)
    )
    original = next(n for n in module.body if isinstance(n, ast.FunctionDef))
    serial_loop = next(
        n
        for n in ast.walk(original)
        if isinstance(n, ast.For)
        and ast.unparse(n.iter) == f"range(cutlass.Int32(0), cutlass.Int32({steps}))"
    )
    serial_name = ast.unparse(serial_loop.target)
    assert not any(
        isinstance(n, ast.If)
        and any(
            isinstance(v, ast.Name) and v.id == serial_name for v in ast.walk(n.test)
        )
        for n in ast.walk(variant)
    )
    scope = next(
        n
        for n in ast.walk(variant)
        if isinstance(n, (ast.FunctionDef, ast.For))
        and any(
            isinstance(s, ast.Assign)
            and ast.unparse(s.targets[0]) == "serial_carry_values"
            for s in n.body
        )
    )
    loops = [
        n
        for n in scope.body
        if isinstance(n, ast.For) and ast.unparse(n.target).startswith("serial_group")
    ]
    assert len(loops) == int(steps > depth)
    if loops:
        assert (
            ast.unparse(loops[0].iter)
            == f"range(cutlass.Int32(0), cutlass.Int32({steps // depth - 1}))"
        )
    final = scope.body[scope.body.index(loops[0]) + 1 :] if loops else scope.body
    # The actual carry-register writes count final updates, plus one init loop
    # when no dynamic group is present. Last incoming store still exists.
    writes = [
        n
        for stmt in final
        for n in ast.walk(stmt)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and "serial_carry_values" in ast.unparse(n.targets[0])
    ]
    assert len(writes) == depth - 1 + int(not loops)
    if loops:
        assert not any(
            isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith("serial_future")
            for stmt in final
            for n in ast.walk(stmt)
        )


@pytest.mark.parametrize("factor", [1, 2])
def test_guarded_missing_whole_source_and_fallback_identity(factor):
    values = args(steps=8)
    old = source(values, factor=factor, tail=None)
    assert old == source(values, factor=factor, tail="guarded")
    new = source(values, factor=factor)
    old_nodes = {
        n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)
    }
    new_nodes = {
        n.name: n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef)
    }
    for name, node in old_nodes.items():
        if not name.endswith("_serial_lane"):
            assert ast.dump(node) == ast.dump(new_nodes[name])


@pytest.mark.parametrize("value", [True, False, 1, None, "peel", [], {}])
@pytest.mark.parametrize("repair", [False, True])
def test_invalid_enum_never_repaired(value, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args(steps=8))
        with pytest.raises(exc.InvalidConfig, match="tail_schedule"):
            bound.config_spec.normalize(
                config().config | {KEY: value}, _fix_invalid=repair
            )


@pytest.mark.parametrize("mode", ["current", "group2", "group4"])
@pytest.mark.parametrize("repair", [False, True])
def test_unsupported_parent_never_repaired(mode, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(args(steps=8))
        with pytest.raises(exc.InvalidConfig, match="tail_schedule"):
            bound.config_spec.normalize(
                config(1, mode).config | {KEY: "peel_final_group"}, _fix_invalid=repair
            )


@pytest.mark.parametrize("depth,steps", [(2, 1), (2, 3), (4, 2), (4, 6)])
def test_partial_or_short_group_rejects(depth, steps):
    with pytest.raises(exc.BackendUnsupported, match="serial lane"):
        source(args(steps=steps), depth)


def test_actual_host_alias_fallback_grid():
    values = args(steps=8)
    calls = []
    module = _module(source(values, factor=2), calls)
    alias = (
        values[0]
        .view(torch.bfloat16)
        .flatten()[: values[2].numel()]
        .view(values[2].shape)
    )
    with patch.object(serial_lane_guard, "_require_cuda"):
        module._recurrence(*values)
        module._recurrence(values[0], values[1], alias)
    assert [(fn.__name__.endswith("_serial_lane"), grid) for fn, grid, *_ in calls] == [
        (True, (2,)),
        (False, (4,)),
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_raw_coefficient_type_preserved(dtype):
    values = list(args(steps=4))
    values[1] = values[1].to(dtype)
    text = source(tuple(values))
    expected = {
        torch.float16: "Float16",
        torch.bfloat16: "BFloat16",
        torch.float32: "Float32",
    }[dtype]
    assert f"cute.make_rmem_tensor((4,), cutlass.{expected})" in text
    old, _ = run(source(tuple(values), tail=None), tuple(values), reordered=False)
    new, _ = run(text, tuple(values), reordered=True)
    assert torch.equal(old, new)


@pytest.mark.parametrize("depth", [2, 4])
def test_optional_scalar_absent_and_width2(depth):
    from .test_cute_serial_lane import _config
    from .test_cute_serial_prefetch import _without_scalar

    values = (torch.empty(2, 4, 64), torch.empty(2, 4, 64, dtype=torch.bfloat16))
    with _cpu_codegen():
        bound = _without_scalar._bind_isolated(values)
        for width in (4, 2):
            raw = _config(width=width)
            raw.config.update(
                {
                    KEY: "peel_final_group",
                    "cute_serial_lane_load_schedule": f"prefetch{depth}",
                }
            )
            text = bound.to_code(raw)
            assert text == bound.to_code(bound._normalized_config_copy(raw))
            assert "serial_raw_coefficients" not in text


@pytest.mark.parametrize("depth", [2, 4])
def test_int32_refill_and_final_literal_domain(depth):
    for steps in (depth, depth * 3, (2**31 - 1) // depth * depth):
        groups = steps // depth
        for group in {0, groups - 2}:
            if 0 <= group < groups - 1:
                for slot in range(depth):
                    current = group * depth + slot
                    assert 0 <= current < steps - depth
                    assert depth <= current + depth < steps < 2**31
        assert list(range(steps - depth, steps))[-1] == steps - 1


@pytest.mark.parametrize("family", ["serial", "chained", "chained_cached"])
def test_old_pool_default_priority_and_actual_initial100(family):
    from torch._inductor.codecache import PyCodeCache

    from test.test_cute_chained_late_rhs import _args as cached_args
    from test.test_cute_chained_late_rhs import _pair as cached_pair

    from helion._compiler.autotuner_heuristics import cute_guarded_siblings
    from helion.autotuner.base_search import PopulationBasedSearch
    from helion.autotuner.config_generation import ConfigGeneration

    fn, values = (
        (_recurrence, args(steps=8, features=64))
        if family == "serial"
        else (
            (_pair, chain_args())
            if family == "chained"
            else (cached_pair, cached_args())
        )
    )
    with _cpu_codegen():
        with patch.object(
            cute_guarded_siblings, "guarded_schedule_siblings", return_value=[]
        ):
            old = fn._bind_isolated(values)
        bound = fn._bind_isolated(values)
        legacy = [
            p
            for p in bound.config_spec.compiler_seed_configs
            if KEY not in p.config and FAST not in p.config
        ]
        assert legacy == old.config_spec.compiler_seed_configs
        assert bound.config_spec.default_config() == old.config_spec.default_config()
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            flat = generation.random_population_flat(100)
            members = [
                PopulationBasedSearch.make_unbenchmarked(
                    cast(
                        "PopulationBasedSearch", SimpleNamespace(config_gen=generation)
                    ),
                    p,
                )
                for p in flat
            ]
            assert len(members) == 100
            disabled = (
                ConfigGeneration(
                    bound.config_spec, overrides={KEY: "guarded", FAST: False}
                )
                if family == "serial"
                else ConfigGeneration(bound.config_spec, overrides={FAST: False})
            )
            assert all(
                KEY not in p.config and FAST not in p.config
                for _, p in disabled.seed_flat_config_pairs()
            )
            user = bound.config_spec.compiler_seed_configs[-1]
            priority = generation.random_population_flat(100, user_seed_configs=[user])
            assert priority[1] == generation.flatten(user)
        selected = [
            m.config
            for m in members
            if m is not None and (m.config.config.get(FAST) or KEY in m.config.config)
        ]
        assert selected and any(p.config.get(FAST) for p in selected)
        if family == "serial":
            assert any(KEY in p.config for p in selected)
        else:
            assert len(selected) == (2 if family == "chained" else 1)
        for raw in selected:
            canonical = bound._normalized_config_copy(raw)
            if family == "chained":
                # The ranked donor explicitly requests a read cache, but this
                # generic expression has no eligible invariant reads. Preserve
                # that strict rejection for both donor and guarded sibling.
                assert raw.config["cute_chained_pointwise_read_cache"] is True
                assert raw.config["cute_chained_pointwise_inplace_async"] is False
                parent = type(raw).from_dict(
                    {key: value for key, value in raw.config.items() if key != FAST}
                )
                for candidate in (raw, canonical, parent):
                    with pytest.raises(
                        exc.BackendUnsupported, match="row-invariant typed reads"
                    ):
                        bound.to_code(candidate)
                continue
            text = bound.to_code(raw)
            assert text == bound.to_code(canonical)
            if raw.config.get(FAST):
                assert "_host_fast" in text

            class Stop(BaseException):
                pass

            with patch.object(PyCodeCache, "load", side_effect=Stop):
                for candidate in (raw, canonical):
                    with pytest.raises(Stop):
                        bound.compile_config(candidate, allow_print=False)
