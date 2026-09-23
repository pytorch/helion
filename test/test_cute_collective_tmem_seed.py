from __future__ import annotations

import ast
from dataclasses import replace
from itertools import count
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes
from test.test_cute_collective_vector_epilogue_codegen import _mamba_config

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_tcgen05 import CollectiveTcgen05Plan
from helion._compiler.cute.collective_tcgen05 import CollectiveTmemResource
from helion._compiler.cute.collective_tmem_seed import NativeCollectiveLifetime
from helion._compiler.cute.collective_tmem_seed import _same_cell
from helion._compiler.cute.collective_tmem_seed import _seed_recipe
from helion._compiler.cute.collective_tmem_seed import bridge_collective_tmem_seeds
from helion._compiler.cute.collective_tmem_seed import prune_pure_scalar_statements
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from collections.abc import Callable


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _pair() -> tuple[
    list[ast.stmt], NativeCollectiveLifetime, NativeCollectiveLifetime
]:
    resource = CollectiveTmemResource("tmem", 64)
    source_plan = CollectiveTcgen05Plan(
        "first", "first_tid", 64, 64, 32, "cutlass.Float16", resource
    )
    target_plan = replace(
        source_plan, prefix="second", tid="second_tid", zero_seed=False
    )
    source_setup = source_plan.setup("thread")
    source_finish = source_plan.finish(_expr("0"), _expr("limit"))
    target_setup = target_plan.setup("thread")
    seed = ast.parse("value = first_c[(m0 + row) - m0, (n0 + col) - n0] * scale").body
    seed_nodes = [*seed, *target_plan.seed_from_shared()]
    source = NativeCollectiveLifetime(
        source_plan,
        "m0",
        "n0",
        "row",
        "col",
        (2, 64, 1),
        tuple(source_setup),
        (),
        (),
        ast.Constant(0),
        tuple(source_finish),
        _expr("0 < limit"),
    )
    target = NativeCollectiveLifetime(
        target_plan,
        "m0",
        "n0",
        "row",
        "col",
        (2, 64, 1),
        tuple(target_setup),
        tuple(seed_nodes),
        tuple(cast("list[ast.Assign]", seed)),
        _expr("value"),
        (),
        ast.Constant(True),
    )
    return [*source_setup, *source_finish, *target_setup, *seed_nodes], source, target


def _fresh() -> Callable[[str], str]:
    counter = count()
    return lambda name: f"{name}_{next(counter)}"


@pytest.mark.parametrize("index", ["row", "(m0 + row) - m0"])
def test_exact_integer_cell_is_accepted(index: str) -> None:
    assert _same_cell(_expr(index), "m0", "row")


@pytest.mark.parametrize(
    "index",
    [
        "col",
        "row + 1",
        "(m0 + row) - n0",
        "(n0 + row) - m0",
        "int(row)",
        "(m0 + row + 1) - m0",
        "(m0 + col) - m0",
    ],
)
def test_shift_alias_cast_or_transpose_is_rejected(index: str) -> None:
    assert not _same_cell(_expr(index), "m0", "row")


@pytest.mark.parametrize(
    "expression",
    [
        "first_c[row + 1, col]",
        "first_c[col, row]",
        "first_c.shape[0]",
        "alias[first_c]",
        "first_c[row, col] + cute.arch.thread_idx()[0]",
    ],
)
def test_seed_does_not_transplant_unknown_coordinates(expression: str) -> None:
    _, source, target = _pair()
    target = replace(target, seed_statements=(), seed_value=_expr(expression))
    assert _seed_recipe(source, target, "registers", "element") is None


@pytest.mark.parametrize("row", [0, 1, 31, 63])
@pytest.mark.parametrize("col", [0, 1, 31, 63])
def test_recipe_preserves_per_element_arithmetic(row: int, col: int) -> None:
    _, source, target = _pair()
    result = _seed_recipe(source, target, "registers", "element")
    assert result is not None
    statements, value = result
    environment = {"registers": [float(row * 64 + col)], "element": 0, "scale": 0.25}
    module = ast.fix_missing_locations(
        ast.Module(
            body=[*statements, ast.Assign([ast.Name("result", ast.Store())], value)],
            type_ignores=[],
        )
    )
    exec(compile(module, "<seed>", "exec"), environment)
    assert environment["result"] == (row * 64 + col) * 0.25


def test_seed_eliminates_dead_drain_and_captures_empty_k_before_setup() -> None:
    body, source, target = _pair()
    rebind = ast.parse("limit = other_limit").body[0]
    position = body.index(target.before_seed[0])
    body.insert(position, rebind)
    target = replace(target, before_seed=(rebind, *target.before_seed))
    assert bridge_collective_tmem_seeds(body, [source, target], _fresh()) == 1
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body, [])))
    assert "first_c" not in code
    assert "first_tmem" not in code
    assert code.index("initialized = 0 < limit") < code.index("limit = other_limit")
    assert "_values.fill(0.0)" in code
    assert "if collective_tmem_seed_0_initialized:" in code
    assert code.index("fence_view_async_tmem_load()") < code.index("St16x128bOp")
    assert code.index("fence_view_async_tmem_store()") < code.rindex("sync_threads()")


@pytest.mark.parametrize(
    "fanout", ["consume(first_c)", "consume(first_tmem_registers)"]
)
def test_additional_consumers_retain_shared_drain(fanout: str) -> None:
    body, source, target = _pair()
    body.extend(ast.parse(fanout).body)
    assert bridge_collective_tmem_seeds(body, [source, target], _fresh()) == 1
    assert all(statement in body for statement in source.finish)
    assert all(statement in body for statement in source.before_seed)


@pytest.mark.parametrize(
    "kind", ["other_effect", "other_resource", "other_tile", "other_offsets", "branch"]
)
def test_nonadjacent_or_incompatible_collectives_keep_original_ast(kind: str) -> None:
    body, source, target = _pair()
    if kind == "other_effect":
        body.insert(
            body.index(target.before_seed[0]), ast.parse("unknown_effect()").body[0]
        )
    elif kind == "other_resource":
        target = replace(
            target,
            plan=replace(target.plan, resource=CollectiveTmemResource("other", 64)),
        )
    elif kind == "other_tile":
        target = replace(target, plan=replace(target.plan, bm=128))
    elif kind == "other_offsets":
        target = replace(target, m_offset="other_row")
    else:
        index = body.index(target.before_seed[0])
        body[index:] = [ast.If(_expr("predicate"), body[index:], [])]
    before = ast.dump(ast.Module(body, []))
    assert bridge_collective_tmem_seeds(body, [source, target], _fresh()) == 0
    assert ast.dump(ast.Module(body, [])) == before


@pytest.mark.parametrize("condition", [False, True])
def test_pure_liveness_handles_branch_rebinding(condition: bool) -> None:
    original = ast.parse("""
dead = old
value = dead * 2
value = 3
if condition:
    value = value + 4
    unused = old + 9
else:
    unused = 2
if condition:
    result = value
else:
    result = value * 2
""").body
    pruned, live = prune_pure_scalar_statements(original, {"result"})
    assert live == {"condition"}
    for statements in (original, pruned):
        values = {"old": 100, "condition": condition}
        module = ast.fix_missing_locations(ast.Module(statements, []))
        exec(compile(module, "<liveness>", "exec"), values)
        assert values["result"] == (7 if condition else 6)
    assert "old" not in ast.unparse(ast.fix_missing_locations(ast.Module(pruned, [])))


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("block_m,block_n", [(64, 32), (64, 64), (128, 32), (128, 64)])
@skipUnlessBackends(["cute"])
def test_generated_mamba_has_direct_seed_and_live_final_drain(
    static_shapes: bool, block_m: int, block_n: int
) -> None:
    args = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2))
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), args)
    config = helion.Config.from_dict(
        _mamba_config("vector_unrolled").config
        | {
            "block_sizes": [block_m, block_n, 64],
            "num_threads": [128 // block_n, block_n, 1, 1],
            "cute_collective_tmem_seed": True,
        }
    )
    source = bound.to_code(config)
    assert "collective_tmem_seed_values" in source
    assert "collective_c[" not in source
    assert "collective_cp =" not in source
    assert "collective_1_c[" in source
    assert "collective_1_seed_source" not in source
    assert ("St16x128bOp" in source) == (block_m == 64)
    assert ("St32x32bOp" in source) == (block_m == 128)
    assert "mul.rn.f32" in source


@skipUnlessBackends(["cute"])
def test_tmem_seed_search_and_config_roundtrip() -> None:
    args = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2))
    bound = _cpu_bind(_mamba_kernel(static_shapes=False), args)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert any(seed.get("cute_collective_tmem_seed", False) for seed in seeds)
        generation = ConfigGeneration(bound.config_spec)
        for enabled in (False, True):
            config = helion.Config.from_dict(
                _mamba_config("vector_unrolled").config
                | {"cute_collective_tmem_seed": enabled}
            )
            flat, recovered = generation.canonicalize_flat(generation.flatten(config))
            assert recovered["cute_collective_tmem_seed"] is enabled


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
@skipUnlessBackends(["cute"])
def test_invalid_tmem_seed_flag_is_rejected(value: object) -> None:
    args = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(0))
    bound = _cpu_bind(_mamba_kernel(static_shapes=False), args)
    config = helion.Config.from_dict(
        _mamba_config("scalar").config | {"cute_collective_tmem_seed": value}
    )
    with pytest.raises(helion.exc.InvalidConfig, match="cute_collective_tmem_seed"):
        bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _dependent_contractions(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(0)), dtype=torch.float32, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(0)]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(lengths[0]):
            acc = hl.dot(a[row, reduction], b[column, reduction].T, acc=acc)
        acc = acc * scale[row, None]
        for reduction in hl.tile(lengths[1]):
            acc = hl.dot(a[row, reduction] * 0.5, b[column, reduction].T, acc=acc)
        out[row, column] = acc
    return out


def _dependent_config(block_m: int, block_n: int, enabled: bool) -> helion.Config:
    return helion.Config(
        block_sizes=[block_m, block_n, 32, 32],
        num_threads=[128 // block_n, block_n, 1, 1],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["blocked"] * 4,
        cute_collective_mma=True,
        cute_collective_compute="tcgen05",
        cute_collective_native_seeded=True,
        cute_collective_copy="async_cached",
        cute_collective_recipe="vector_unrolled",
        cute_collective_epilogue="vector_unrolled",
        cute_collective_tmem_seed=enabled,
    )


def _dependent_inputs(dtype: torch.dtype, device: str) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((130, 80), dtype=dtype, device=device)[:, :78],
        torch.empty((70, 80), dtype=dtype, device=device)[:, :78],
        torch.empty(130, dtype=torch.float32, device=device),
        torch.empty(2, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("block_m,block_n", [(64, 32), (128, 64)])
@skipUnlessBackends(["cute"])
def test_dynamic_empty_reduction_codegen(block_m: int, block_n: int) -> None:
    bound = _cpu_bind(_dependent_contractions, _dependent_inputs(torch.float16, "cpu"))
    source = bound.to_code(_dependent_config(block_m, block_n, True))
    assert "collective_tmem_seed_values" in source
    assert "collective_tmem_seed_initialized" in source
    # The contiguous N70 output cannot use the aligned vector epilogue. Its
    # original scalar consumer keeps the first drain alive conservatively.
    assert "collective_c[" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_m,block_n", [(64, 32), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_dependent_contractions_cuda_tail_empty_k_and_graph_reset(
    block_m: int, block_n: int, dtype: torch.dtype
) -> None:
    args = _dependent_inputs(dtype, "cuda")
    a, b, scale, lengths = args
    torch.manual_seed(915)
    for value in (a, b, scale):
        value.copy_(torch.randint(-4, 5, value.shape, device=CUDA_DEVICE) * 0.25)
    lengths.fill_(78)
    bound = _dependent_contractions._bind_isolated(args)
    config = _dependent_config(block_m, block_n, True)
    assert "collective_tmem_seed_values" in bound.to_code(config)
    bound.set_config(config)
    control = _dependent_contractions._bind_isolated(args)
    control.set_config(_dependent_config(block_m, block_n, False))
    replay = _make_cudagraph_replay(lambda: bound(*args))
    for first, second in ((78, 78), (0, 78), (78, 0), (0, 0), (17, 37), (78, 78)):
        lengths.copy_(
            torch.tensor([first, second], device=CUDA_DEVICE, dtype=torch.int32)
        )
        a.mul_(-1)
        scale.mul_(-1)
        expected = (a[:, :first].float() @ b[:, :first].float().T) * scale[:, None]
        expected += (a[:, :second].float() * 0.5) @ b[:, :second].float().T
        torch.testing.assert_close(control(*args), expected, rtol=0, atol=0)
        torch.testing.assert_close(bound(*args), expected, rtol=0, atol=0)
        replay().fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    # Bound kernels specialize input layouts. A pointer-replacement check must
    # preserve the original padded strides; compact clones need a fresh bind.
    replacements = _dependent_inputs(dtype, "cuda")
    for old, new in zip(args, replacements, strict=True):
        new.copy_(old)
    torch.testing.assert_close(bound(*replacements), expected, rtol=0, atol=0)
    compact = tuple(value.clone() for value in args)
    rebound = _dependent_contractions._bind_isolated(compact)
    rebound.set_config(config)
    torch.testing.assert_close(rebound(*compact), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_m,block_n", [(64, 32), (64, 64), (128, 32), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_mamba_cuda_tmem_seed_matches_existing_exactly(
    block_m: int, block_n: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(20260913)
    args = tuple(
        torch.randn(size, device=CUDA_DEVICE, dtype=torch.bfloat16).to(dtype)
        for size in _mamba_sizes(2)
    )
    args[2].uniform_()
    args[3].copy_(-torch.rand_like(args[2]).cumsum(-1))
    kernel = _mamba_kernel(static_shapes=False)
    config = helion.Config.from_dict(
        _mamba_config("vector_unrolled").config
        | {
            "block_sizes": [block_m, block_n, 64],
            "num_threads": [128 // block_n, block_n, 1, 1],
            "cute_collective_tmem_seed": True,
        }
    )
    bound = kernel._bind_isolated(args)
    assert "collective_tmem_seed_values" in bound.to_code(config)
    bound.set_config(config)
    control = kernel._bind_isolated(args)
    control.set_config(
        helion.Config.from_dict(config.config | {"cute_collective_tmem_seed": False})
    )
    replay = _make_cudagraph_replay(lambda: bound(*args))
    for factor in (1.0, 0.875, 1.125):
        args[1].mul_(factor)
        args[5].mul_(factor)
        expected = control(*args)
        torch.testing.assert_close(bound(*args), expected, rtol=0, atol=0)
        output = replay()
        output.fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = tuple(value.clone() for value in args)
    torch.testing.assert_close(
        bound(*replacements), control(*replacements), rtol=0, atol=0
    )
