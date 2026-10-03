from __future__ import annotations

import ast

import numpy as np
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from .test_cute_bounded_cache_codegen import _cpu_target
from .test_cute_bounded_loop_cache import SOURCE
from .test_cute_bounded_loop_cache import _execute
from .test_cute_bounded_loop_cache import _prepare
import helion
from helion._compiler.autotuner_heuristics.cute_bounded_loop_cache import (
    CuteBoundedLoopCacheHeuristic,
)
from helion._compiler.cute.bounded_loop_cache import attach_variant
from helion._compiler.cute.bounded_loop_cache import cache_single_tile_loads
from helion._compiler.cute.bounded_loop_cache import specialize_single_tile_sweeps
from helion._testing import skipUnlessBackends
import helion.language as hl


def _admit(function: ast.FunctionDef):
    arguments = {argument.arg for argument in function.args.args}
    plan = specialize_single_tile_sweeps(
        function.body,
        integer_arguments=arguments - {"x", "out"},
        constexpr_values={"BLOCK": 8},
        launch_block=(1, 1, 1),
    )
    assert plan is not None
    cached = cache_single_tile_loads(
        plan.body,
        plan.plan,
        argument_names=arguments,
        constexpr_values={"BLOCK": 8},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset({"x", "out"})},
        rename_groups={},
    )
    return plan, cached


@pytest.mark.parametrize("depth", [0, 1])
def test_proof_coordinates_cannot_alias_real_boundary_arguments(depth):
    argument = f"__bounded_coordinate_{depth}"
    function = ast.parse(SOURCE.replace("row", argument)).body[0]
    assert isinstance(function, ast.FunctionDef)
    lane = "lane" if depth == 0 else "vec_lane_0"
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        if node.targets[0].id == "value":
            expression = f"(x.iterator + cutlass.Int32({lane})).load() if valid else cutlass.Float32(0)"
        elif node.targets[0].id == "other_value":
            expression = f"(x.iterator + cutlass.Int32({argument})).load() if other_valid else cutlass.Float32(0)"
        else:
            continue
        node.value = ast.parse(expression, mode="eval").body
    _, cached = _admit(function)
    assert cached is None
    values = [np.float32(index + 1) for index in range(8)]
    expected, _, _ = _execute(function, 8, 0, (8, 1), values)
    assert (
        np.frombuffer(expected, dtype=np.float32).tolist()
        == [12 if depth == 0 else 20] * 8
    )


@pytest.mark.parametrize(
    "argument",
    [
        "__bounded_coordinate_0",
        "__bounded_coordinate_1",
        "__bounded_recipe_1_base_remat",
        "__bounded_recipe_2_column_remat",
        "__bounded_recipe_3_valid_remat",
    ],
)
def test_colliding_user_names_preserve_valid_cache_and_exact_values(argument):
    function = ast.parse(SOURCE.replace("row", argument)).body[0]
    assert isinstance(function, ast.FunctionDef)
    _, cached = _admit(function)
    assert cached is not None and cached.replaced_loads == 1
    fast = ast.FunctionDef(
        name="kernel", args=function.args, body=cached.body, decorator_list=[]
    )
    values = [np.float32(index + 1) for index in range(8)]
    expected = _execute(function, 8, 0, (8, 1), values)
    actual = _execute(fast, 8, 0, (8, 1), values)
    assert actual[0] == expected[0]
    assert actual[1:] == (8, 8)


@pytest.mark.parametrize(
    "argument", ["_helion_bounded_cache", "_helion_cache_lane", "_helion_single_tile"]
)
def test_generated_cache_bindings_do_not_shadow_unused_arguments(argument):
    function = ast.parse(SOURCE).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.args.args.append(ast.arg(arg=argument))
    function.args.defaults.append(ast.Constant(17))
    _, cached = _admit(function)
    assert cached is not None
    assert all(
        not (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == argument
        )
        for statement in cached.body
        for node in ast.walk(statement)
    )
    fast = ast.FunctionDef(
        name="kernel", args=function.args, body=cached.body, decorator_list=[]
    )
    values = [np.float32(index + 1) for index in range(8)]
    assert (
        _execute(fast, 8, 0, (8, 1), values)[0]
        == _execute(function, 8, 0, (8, 1), values)[0]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "x, out, n, row, _launcher, kernel_bounded=17",
        "kernel_bounded, /, x, out, n, row, _launcher",
        "x, out, n, row, _launcher, *, kernel_bounded=17",
        "x, out, n, row, _launcher, *kernel_bounded",
        "x, out, n, row, _launcher, **kernel_bounded",
    ],
)
def test_host_variant_avoids_all_argument_kinds(arguments):
    original, specialized, cached = _prepare()
    assert specialized is not None and cached is not None
    wrapper = f"\ndef wrapper({arguments}):\n    _launcher(kernel, (1,), x, out, n, row, block=(1, 1, 1))\n"
    fast = ast.FunctionDef(
        name="kernel", args=original.args, body=cached.body, decorator_list=[]
    )
    merged = attach_variant(
        "BLOCK = 8\n" + ast.unparse(original) + wrapper,
        "BLOCK = 8\n" + ast.unparse(ast.fix_missing_locations(fast)) + wrapper,
        specialized.plan,
        kernel_name="kernel",
    )
    assert merged is not None
    functions = {
        node.name: node
        for node in ast.parse(merged).body
        if isinstance(node, ast.FunctionDef)
    }
    assert "kernel_bounded_" in functions and "kernel_bounded" not in functions
    assert ast.dump(functions["kernel"]) == ast.dump(original)


def name_collision(
    x: torch.Tensor, _helion_name_collision_bounded: int
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    columns = hl.register_block_size(n)
    for row in hl.tile(m):
        total = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(n, block_size=columns):
            value = x[row, column].to(torch.float32)
            total += torch.sum(value * value, dim=1)
        for column in hl.tile(n, block_size=columns):
            value = x[row, column].to(torch.float32)
            out[row, column] = value * total[:, None]
    return out


@skipUnlessBackends(["cute"])
def test_real_bound_unused_host_argument_cannot_replace_fast_kernel():
    with _cpu_target():
        kernel = helion.kernel(
            name_collision, backend="cute", static_shapes=False, autotune_effort="none"
        )
        x = torch.empty((5, 513), dtype=torch.float32)
        bound = kernel._bind_isolated((x, 17))
        assert bound.host_function is not None
        with bound.env:
            configs = CuteBoundedLoopCacheHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            assert configs
            config = configs[0]
        code = bound.to_code(config)
        module = PyCodeCache.load(code)
        calls = []

        def capture(implementation, grid, *args, **kwargs):
            calls.append(implementation)

        result = module.name_collision(x, 17, _launcher=capture)
        assert result.shape == x.shape
        assert calls == [module._helion_name_collision_bounded_]
        assert callable(calls[0])
