from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
from ._serial_lane_model import run
from .test_cute_serial_lane import _args
from .test_cute_serial_lane import _config
from .test_cute_serial_lane import _module
from .test_cute_serial_lane import _recurrence
from .test_cute_serial_lane import _source
import helion
from helion import exc
from helion._compiler.cute.serial_lane_recurrence import LOAD_KEY
from helion._compiler.cute.serial_lane_recurrence import LOAD_SCHEDULES
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute import serial_lane_guard

pytestmark = skipUnlessBackends(["cute"])


def _code(mode="prefetch4", args=None, width=4):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args() if args is None else args)
        config = _config(width=width)
        config.config[LOAD_KEY] = mode
        canonical = bound._normalized_config_copy(config)
        raw = bound.to_code(config)
        assert raw == bound.to_code(canonical)
        return raw


def test_current_default_full_source_identity():
    assert _code("current") == _source()


@pytest.mark.parametrize("mode", LOAD_SCHEDULES)
@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 7, 64, 65])
def test_actual_ast_math_reads_ownership_and_serial_tails(mode, steps):
    generator = torch.Generator().manual_seed(193)
    args = (
        torch.randn(2, steps, 64, generator=generator),
        -torch.rand(2, steps, generator=generator),
        torch.empty(2, steps, 64, dtype=torch.bfloat16),
    )
    source = _code(mode, args)
    old, old_counts = run(source, args, reordered=False)
    new, counts = run(source, args, reordered=True)
    assert torch.equal(new, old)
    assert counts["state_reads"] == old_counts["state_reads"] == args[0].numel()
    assert counts["coefficient_reads"] * 4 == old_counts["coefficient_reads"]
    assert "sync_threads" not in source and "cp_async" not in source
    assert (
        "serial_prefetch_states" in source
        if mode.startswith("prefetch")
        else "serial_prefetch_states" not in source
    )


@pytest.mark.parametrize("mode", ["prefetch2", "prefetch4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width,stride", [(2, 1), (4, 2)])
def test_typed_output_and_guarded_scalar_transfers(mode, dtype, width, stride):
    args = _args(dtype, stride=stride)
    generator = torch.Generator().manual_seed(311)
    args[0].copy_(torch.randn(args[0].shape, generator=generator))
    args[1].copy_(-torch.rand(args[1].shape, generator=generator))
    source = _code(mode, args, width)
    old, _ = run(source, args, reordered=False)
    new, counts = run(source, args, reordered=True)
    assert torch.equal(new, old)
    if stride == 2:
        assert counts["vector_load"] == counts["vector_store"] == 0


@pytest.mark.parametrize("value", [True, False, 1, 2, None, "prefetch8", "other"])
@pytest.mark.parametrize("repair", [False, True])
def test_invalid_values_never_repair(value, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        config = _config().config | {LOAD_KEY: value}
        with pytest.raises(exc.InvalidConfig, match="serial"):
            bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("parent", [None, "lane_major", "step_major"])
@pytest.mark.parametrize("repair", [False, True])
def test_missing_vector_parent_never_repair(parent, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        config = _config(parent).config | {LOAD_KEY: "prefetch4"}
        with pytest.raises(exc.InvalidConfig, match="step_major_vector"):
            bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("mode", LOAD_SCHEDULES)
def test_original_fallback_and_host_are_unchanged(mode):
    original = ast.parse(_source())
    candidate = ast.parse(_code(mode))
    old_functions = [n for n in original.body if isinstance(n, ast.FunctionDef)]
    new_functions = [n for n in candidate.body if isinstance(n, ast.FunctionDef)]
    assert ast.dump(old_functions[0]) == ast.dump(new_functions[0])
    assert ast.dump(old_functions[-1]) == ast.dump(new_functions[-1])
    launches = []
    module = _module(ast.unparse(candidate), launches)
    args = _args()
    alias = (
        args[0].view(torch.bfloat16).flatten()[: args[2].numel()].view(args[2].shape)
    )
    with patch.object(serial_lane_guard, "_require_cuda"):
        module._recurrence(*args)
        module._recurrence(args[0], args[1], alias)
        module._recurrence(*_args(offset=True))
    assert [fn.__name__.endswith("_serial_lane") for fn, *_ in launches] == [
        True,
        False,
        True,
    ]


def test_old_sibling_block_and_default_preserved():
    from helion._compiler.autotuner_heuristics.serial_lane import serial_lane_seeds

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        assert bound.host_function is not None
        with bound.env:
            result = serial_lane_seeds(bound.env, bound.host_function.device_ir)
            old = [c for c in result if LOAD_KEY not in c.config]
            assert len(old) == 4 and result[:4] == old
            assert [c.config["cute_serial_lane_schedule"] for c in old] == [
                "step_major",
                "step_major_vector",
            ] * 2
            assert len(result) == 12
            for index, parent in enumerate((old[1], old[3])):
                assert [
                    c.config[LOAD_KEY] for c in result[4 + index * 4 : 8 + index * 4]
                ] == list(LOAD_SCHEDULES)
                for child in result[4 + index * 4 : 8 + index * 4]:
                    assert {
                        k: v for k, v in child.config.items() if k != LOAD_KEY
                    } == parent.config
            assert LOAD_KEY not in bound.config_spec.default_config().config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["prefetch2", "prefetch4"])
def test_raw_coefficient_dtype_preserved(dtype, mode):
    args = list(_args())
    args[0].fill_(0.125)
    args[1] = torch.full(args[1].shape, -0.25, dtype=dtype)
    source = _code(mode, tuple(args))
    old, _ = run(source, tuple(args), reordered=False)
    new, _ = run(source, tuple(args), reordered=True)
    assert torch.equal(old, new)
    expected = {
        torch.float32: "Float32",
        torch.float16: "Float16",
        torch.bfloat16: "BFloat16",
    }[dtype]
    assert f"cute.make_rmem_tensor(({mode[-1]},), cutlass.{expected})" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _without_scalar(states, out):
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], 0.25, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = carry * 0.75 + states[bi.begin, qi, fi]
    return out


@pytest.mark.parametrize("mode", LOAD_SCHEDULES)
def test_optional_scalar_stream_absent(mode):
    with _cpu_codegen():
        args = _args()
        bound = _without_scalar._bind_isolated((args[0], args[2]))
        config = _config()
        config.config[LOAD_KEY] = mode
        source = bound.to_code(config)
        assert "serial_raw_coefficients" not in source
        assert source == bound.to_code(bound._normalized_config_copy(config))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_production_initial100_and_preload_stops(dtype):
    from torch._inductor.codecache import PyCodeCache

    from helion.autotuner.base_search import PopulationBasedSearch
    from helion.autotuner.config_generation import ConfigGeneration

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args(dtype))
        with bound.env:
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
        candidates = [
            member.config
            for member in members
            if member is not None and LOAD_KEY in member.config.config
        ]
        assert {config.config[LOAD_KEY] for config in candidates} == set(LOAD_SCHEDULES)
        for mode in LOAD_SCHEDULES:
            config = next(c for c in candidates if c.config[LOAD_KEY] == mode)
            source = bound.to_code(config)
            canonical = bound._normalized_config_copy(config)
            assert bound.to_code(canonical) == source

            class StopBeforeLoad(BaseException):
                pass

            def stop(actual, expected=source, **kwargs):
                assert actual == expected
                raise StopBeforeLoad

            with (
                patch.object(PyCodeCache, "load", side_effect=stop),
                patch.object(type(bound.env.backend), "setup_compile_cache_dir"),
            ):
                for request in (config, canonical):
                    with pytest.raises(StopBeforeLoad):
                        bound.compile_config(request, allow_print=False)
            assert not bound._compile_cache


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 7, 63, 64, 65, 2**31 - 3, 2**31 - 1])
@pytest.mark.parametrize("depth", [2, 4])
def test_group_and_refill_integer_boundary_model(steps, depth):
    # Supplementary bound arithmetic, not a huge-tensor source/runtime claim.
    groups = (steps + depth - 1) // depth
    for group in {0, groups - 1}:
        base = group * depth
        assert 0 <= base < 2**31
        for slot in range(depth):
            if base < steps - slot:
                current = base + slot
                assert 0 <= current < steps
                if current < steps - depth:
                    assert current + depth < steps < 2**31
