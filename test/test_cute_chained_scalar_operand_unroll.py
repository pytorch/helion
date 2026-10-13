from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_root_typed_operands import _args
from .test_cute_chained_root_typed_operands import _chain
from .test_cute_chained_root_typed_operands import _tail_args
from .test_cute_chained_root_typed_operands import _tail_chain
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._compiler.cute.chained_scalar_operand_unroll import ScalarOperandUnroll
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


KEY = "cute_chained_scalar_operand_unroll"


def _config(factor: object = 0, typed: bool = False, **extra: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_typed_operands": typed,
            KEY: factor,
            **extra,
        }
    )


def _without_unroll(source: str) -> tuple[str, list[tuple[int, int]]]:
    module = ast.parse(source)
    loops = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and re.fullmatch(r"chain_\d+_[ab]_step", node.target.id)
        ):
            assert isinstance(node.iter, ast.Call)
            assert ast.unparse(node.iter.func) == "cutlass.range"
            (trips,) = node.iter.args
            assert isinstance(trips, ast.Constant)
            (keyword,) = node.iter.keywords
            assert keyword.arg == "unroll"
            assert isinstance(keyword.value, ast.Constant)
            assert isinstance(trips.value, int)
            assert isinstance(keyword.value.value, int)
            loops.append((trips.value, keyword.value.value))
            keyword.value = ast.Constant(value=1)
    return ast.dump(module, include_attributes=False), loops


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_only_scalar_loop_schedule_changes(typed: bool, factor: int) -> None:
    with _cpu_codegen():
        bound = _chain._bind_isolated(_args())
        old = bound.to_code(_config(typed=typed))
        new = bound.to_code(_config(factor, typed))
    old_tree, old_loops = _without_unroll(old)
    new_tree, new_loops = _without_unroll(new)
    assert old_tree == new_tree
    assert old_loops and all(unroll == 1 for _, unroll in old_loops)
    assert [trips for trips, _ in old_loops] == [trips for trips, _ in new_loops]
    assert all(unroll == min(factor, trips) for trips, unroll in new_loops)
    # The first direct-copy fallback is retained; typed aliases eliminate only
    # their original producers, never the surviving guarded scalar fallback.
    assert "_a_async_pointer" in new and "CopyG2SOp" in new
    assert ("chain_typed_" in new) == typed


@pytest.mark.parametrize("typed", [False, True])
def test_masked_strided_operands_keep_exact_scalar_fallback(typed: bool) -> None:
    values = list(_tail_args(33))
    left = values[0]
    assert isinstance(left, torch.Tensor)
    values[0] = torch.empty((*left.shape[:-1], 256), dtype=left.dtype)[..., ::2]
    with _cpu_codegen():
        bound = _tail_chain._bind_isolated(tuple(values))
        old = bound.to_code(_config(typed=typed))
        new = bound.to_code(_config(8, typed))
    assert _without_unroll(old)[0] == _without_unroll(new)[0]
    assert "operator.lt(" in new
    assert all(unroll == 8 for _, unroll in _without_unroll(new)[1])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _padded_root(
    a: torch.Tensor, b: torch.Tensor, rhs: torch.Tensor, initial: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(initial)
    for batch in hl.tile(a.size(0), block_size=1):
        idx = batch.id
        rows = hl.arange(64)
        reduction = hl.arange(16)
        middle = hl.arange(b.size(2))
        columns = hl.arange(32)
        first = hl.dot(
            hl.load(a, [idx, rows, reduction]),
            hl.load(b, [idx, reduction, middle]),
        )
        value = hl.dot(
            first.to(torch.float16),
            hl.load(rhs, [idx, middle, columns]),
            acc=hl.load(initial, [idx, rows, columns]),
        )
        hl.store(out, [idx, rows, columns], value)
    return out


def _padded_args(device: str = "cpu", width: int = 96) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((2, 64, 16), dtype=torch.bfloat16, device=device),
        torch.empty((2, 16, width), dtype=torch.bfloat16, device=device),
        torch.empty((2, width, 32), dtype=torch.float16, device=device),
        torch.empty((2, 64, 32), dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("width", [32, 96])
def test_padded_logical_tail_keeps_original_bounds(typed: bool, width: int) -> None:
    with _cpu_codegen():
        bound = _padded_root._bind_isolated(_padded_args(width=width))
        old = bound.to_code(_config(typed=typed))
        new = bound.to_code(_config(8, typed))
    old_tree, _ = _without_unroll(old)
    new_tree, loops = _without_unroll(new)
    assert new_tree == old_tree
    assert loops and all(factor == min(8, trips) for trips, factor in loops)
    if width == 96:
        assert "< 96" in new  # The physical 128-wide tile retains its logical tail.
    else:
        assert (4, 4) in loops


@pytest.fixture(scope="module")
def scalar_bound() -> BoundKernel:
    with _cpu_codegen():
        return _chain._bind_isolated(_args())


@pytest.mark.parametrize("value", [True, False, 1, 3, 16, -1, 8.0, "8", None])
def test_invalid_values_are_not_repaired(
    scalar_bound: BoundKernel, value: object
) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be 0, 2, 4 or 8"):
        scalar_bound.config_spec.normalized_config(_config(value))


def test_seed_inventory_roundtrip_and_default(scalar_bound: BoundKernel) -> None:
    with _cpu_codegen():
        spec = scalar_bound.config_spec
        field = spec._flat_fields()[KEY]
        assert isinstance(field, EnumFragment)
        assert field.search_values() == [0, 2, 4, 8] and field.default() == 0
        assert KEY not in spec.default_config()
        seeds = list(spec.compiler_seed_configs)
        assert scalar_bound.host_function is not None
        stages = []
        original = CuteChainedMatmulHeuristic._with_root_scalar_operand_unroll_seeds

        def observe(env, ir, parents):
            before = list(parents)
            result = original(env, ir, parents)
            stages.append((before, list(result)))
            return result

        with (
            scalar_bound.env,
            patch.object(
                CuteChainedMatmulHeuristic,
                "_with_root_scalar_operand_unroll_seeds",
                side_effect=observe,
            ),
        ):
            final = CuteChainedMatmulHeuristic.get_seed_configs(
                scalar_bound.env, scalar_bound.host_function.device_ir
            )
        assert final == seeds and len(stages) == 1
        base, stage = stages[0]
        added = stage[len(base) :]
        assert stage[: len(base)] == base and seeds[: len(stage)] == stage
        assert all(not seed.get(KEY, 0) for seed in base)
        assert added == [seed for seed in seeds if seed.get(KEY, 0)]
        assert len(added) == 2
        assert {bool(seed.get("cute_chained_typed_operands")) for seed in added} == {
            False,
            True,
        }
        rebuilt = CuteChainedMatmulHeuristic._with_root_scalar_operand_unroll_seeds(
            scalar_bound.env, scalar_bound.host_function.device_ir, base
        )
        assert rebuilt == stage
        for seed in added:
            parent = helion.Config.from_dict(
                {k: v for k, v in seed.items() if k != KEY}
            )
            assert parent in base
            selected = spec.normalized_config(
                _config(8, bool(seed.get("cute_chained_typed_operands")))
            )
            generation = spec.create_config_generation()
            assert generation.unflatten(generation.flatten(selected)) == selected
            assert _without_unroll(scalar_bound.to_code(seed))[1]
        original = helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")
        assert scalar_bound.to_code(original) == scalar_bound.to_code(_config())


@pytest.mark.parametrize("dimension", [32, 128])
def test_non_m64_geometry_is_not_advertised(dimension: int) -> None:
    with _cpu_codegen():
        bound = _chain._bind_isolated(_args(dimension))
        assert not bound.config_spec.cute_chained_scalar_operand_unroll_search_enabled
        assert KEY not in bound.config_spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="four-warp TCgen root producer"):
            bound.to_code(_config(8))


def test_single_dot_keeps_original_search_surface() -> None:
    from .test_cute_chained_pipeline import _one
    from .test_cute_chained_pipeline import _startup_m64_values

    with _cpu_codegen():
        bound = _one._bind_isolated(_startup_m64_values())
        assert not bound.config_spec.cute_chained_scalar_operand_unroll_search_enabled
        assert KEY not in bound.config_spec._flat_fields()
        assert all(KEY not in seed for seed in bound.config_spec.compiler_seed_configs)


def test_explicit_choice_cannot_fall_back_to_an_unrelated_emitter(
    scalar_bound: BoundKernel,
) -> None:
    with (
        _cpu_codegen(),
        patch(
            "helion._compiler.cute.chained_matmul.plan_chained_matmul",
            return_value=None,
        ),
        pytest.raises(exc.BackendUnsupported, match="admitted root chain"),
    ):
        scalar_bound.to_code(_config(8))


@pytest.mark.parametrize("trips", [0, 1, 3, 12, 17])
def test_scalar_activation_is_local_and_does_not_weaken_vector_proof(
    trips: int,
) -> None:
    scalar = ScalarOperandUnroll(8)
    assert scalar.loop_factor(trips) == min(8, max(1, trips))
    # A speculative fallback can be discarded by a successful vector emitter.
    with pytest.raises(exc.BackendUnsupported, match="emitted multi-trip"):
        scalar.validate()
    scalar.record(trips)
    if trips > 1:
        scalar.validate()
    else:
        with pytest.raises(exc.BackendUnsupported, match="emitted multi-trip"):
            scalar.validate()
    with pytest.raises(exc.BackendUnsupported, match="emitted multi-trip"):
        ScalarOperandUnroll(8).validate()
    with pytest.raises(exc.BackendUnsupported, match="whole number"):
        PointwiseUnroll(8).loop_factor(3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("typed", [False, True])
def test_padded_tail_unroll_gpu(typed: bool) -> None:
    args = _padded_args("cuda")
    generator = torch.Generator(device="cuda").manual_seed(809)
    for value in args:
        value.normal_(generator=generator).mul_(0.025)
    a, b, rhs, initial = args
    first = (a.double() @ b.double()).float().half().double()
    expected = (first @ rhs.double() + initial.double()).float()
    bound = _padded_root._bind_isolated(args)
    bound.set_config(_config(8, typed))
    torch.testing.assert_close(bound(*args), expected, atol=2e-5, rtol=2e-5)
