from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel
from test.test_cute_collective_tmem_seed_order import _initial_population

from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module", autouse=True)
def _forbid_cuda_init() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@pytest.mark.parametrize(
    "shape", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
def test_full_population_reaches_native_register_vector_loads(
    shape: tuple[int, int, int, int],
) -> None:
    batch, rows, reduction, columns = shape
    args = _jagged_inputs(batch=batch, rows=rows, k=reduction, n=columns)
    bound = _cpu_bind(_kernel(), args)
    population = _initial_population(bound, args)
    assert len(population) == len(set(population))
    assert population == _initial_population(bound, args)
    choices = [
        config
        for config in population
        if config.get("cute_collective_compute") == "tcgen05"
        and config.get("cute_collective_copy") == "scalar"
        and config.get("cute_collective_recipe") == "vector_unrolled"
    ]
    assert choices
    assert any(config.block_sizes == [1, 64, 32, 32] for config in choices)
    config = next(config for config in choices if config.block_sizes == [1, 64, 32, 32])
    source = bound.to_code(config)
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    global_vector_types = {
        ast.unparse(node.args[0])
        for node in calls
        if ast.unparse(node.func) == "cute.make_ptr"
        and len(node.args) >= 3
        and ast.unparse(node.args[2]) == "cute.AddressSpace.gmem"
        and any(
            keyword.arg == "assumed_align" and ast.literal_eval(keyword.value) == 16
            for keyword in node.keywords
        )
    }
    assert {"jagged.element_type", "dense.element_type"} <= global_vector_types
    assert "TmemAllocator" in source
    assert "cute.arch.cvt_f32_tf32" in source
    assert "collective_a_raw" not in source and "collective_b_raw" not in source
    assert "cutlass.Float16" not in source and "cutlass.BFloat16" not in source


@pytest.mark.parametrize(
    "shape", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
def test_dynamic_batch_metadata_keeps_existing_seed_admission(
    shape: tuple[int, int, int, int],
) -> None:
    # Recipe pairing does not broaden the proof for a symbolic batch axis.
    # Randomly generated native requests may still fall back to SIMT; those
    # requests are not evidence that a native compiler seed was reached.
    batch, rows, reduction, columns = shape
    args = _jagged_inputs(batch=batch, rows=rows, k=reduction, n=columns)
    bound = _cpu_bind(_kernel(static_shapes=False), args)
    assert bound.host_function is not None
    with bound.env:
        assert not CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


@pytest.mark.parametrize("count", [2, 7, 100])
def test_register_recipe_seeds_preserve_explicit_user_priority(count: int) -> None:
    args = _jagged_inputs(batch=256, rows=30856, k=128, n=128)
    bound = _cpu_bind(_kernel(), args)
    explicit = next(
        config
        for config in reversed(bound.config_spec.compiler_seed_configs)
        if config.get("cute_collective_compute") == "tcgen05"
        and config.get("cute_collective_copy") == "scalar"
        and config.get("cute_collective_recipe") == "vector_unrolled"
    )
    population = _initial_population(bound, args, count=count, user_seeds=[explicit])
    # The helper checks exactly count legacy rows plus declared coverage only.
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        _flat, normalized = generation.canonicalize_flat(generation.flatten(explicit))
    assert population[1] == normalized
