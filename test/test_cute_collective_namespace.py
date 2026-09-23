from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_register_chain import _chain
from test.test_cute_collective_register_chain import _fresh

import helion
from helion._compiler.cute.collective_epilogue import _scalar_epilogue
from helion._compiler.cute.collective_register_chain import emit_register_chain
from helion._compiler.cute.collective_register_chain import plan_register_chain
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from pathlib import Path

CPU_DEVICE = torch.device("cpu")


def _named_kernel(tmp_path: Path, argument: str, *, scalar: bool) -> helion.Kernel:
    left = "a" if scalar else argument
    parameters = (
        f"a: torch.Tensor, b: torch.Tensor, {argument}: float"
        if scalar
        else f"{argument}: torch.Tensor, b: torch.Tensor"
    )
    epilogue = f"acc + {argument}" if scalar else "acc"
    source = f"""
def named_kernel({parameters}) -> torch.Tensor:
    m, k = {left}.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype={left}.dtype, device={left}.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, {left}[tm, tk] * 0.5, b[tk, tn])
        acc = acc * 0.5
        for tk2 in hl.tile(k):
            acc = torch.addmm(acc, {left}[tm, tk2], b[tk2, tn])
        out[tm, tn] = ({epilogue}).to(out.dtype)
    return out
"""
    path = tmp_path / "named_kernel.py"
    path.write_text(source)
    namespace = {"torch": torch, "hl": hl, "__name__": __name__}
    exec(compile(source, str(path), "exec"), namespace)
    return helion.kernel(
        namespace["named_kernel"],
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


def _device_function(source: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion_named_kernel"
    )


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize(
    "argument",
    [
        "normal_input",
        "collective_tid",
        "collective_a",
        "collective_tmem_ptr",
        "collective_operands_ap",
        "collective_tmem_seed_values",
    ],
)
@skipUnlessBackends(["cute"])
def test_collective_template_names_preserve_tensor_arguments(
    tmp_path: Path, argument: str, compute: str
) -> None:
    kernel = _named_kernel(tmp_path, argument, scalar=False)
    inputs = (
        torch.empty((64, 32), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((32, 32), dtype=torch.bfloat16, device=CPU_DEVICE),
    )
    bound = _cpu_bind(kernel, inputs)
    config = helion.Config(
        block_sizes=[64, 32, 32, 32],
        num_threads=[4, 32, 1, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_native_seeded=compute == "tcgen05",
        cute_collective_tmem_seed=compute == "tcgen05",
    )
    source = bound.to_code(config)
    assert "cute.gemm(" in source
    if compute == "tcgen05":
        assert "St16x128bOp" in source
    function = _device_function(source)
    assert argument in {item.arg for item in function.args.args}
    assert not any(
        isinstance(node, ast.Name)
        and node.id == argument
        and isinstance(node.ctx, ast.Store)
        for statement in function.body
        for node in ast.walk(statement)
    )
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
        and node.value.id == argument
        for statement in function.body
        for node in ast.walk(statement)
    )
    assert source == bound.to_code(config)


@pytest.mark.parametrize("value", [-5.5, 0.75, 3.25])
@pytest.mark.parametrize(
    "argument", ["normal_bias", "_helion_rc_0_tid", "_helion_rc_1_arena"]
)
@skipUnlessBackends(["cute"])
def test_register_chain_epilogue_keeps_scalar_argument_values(
    tmp_path: Path, argument: str, value: float
) -> None:
    kernel = _named_kernel(tmp_path, argument, scalar=True)
    inputs = (
        torch.empty((32, 32), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((32, 32), dtype=torch.bfloat16, device=CPU_DEVICE),
        value,
    )
    bound = _cpu_bind(kernel, inputs)
    config = helion.Config(
        block_sizes=[16, 16, 32, 32],
        num_threads=[4, 16, 1, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_register_chain=True,
    )
    source = bound.to_code(config)
    assert "cute.gemm(" in source
    function = _device_function(source)
    additions = [
        node
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and any(
            isinstance(child, ast.Name) and child.id == argument
            for child in ast.walk(node)
        )
    ]
    assert len(additions) == 1
    expression = additions[0]
    environment: dict[str, object] = {
        node.id: 2.0 for node in ast.walk(expression) if isinstance(node, ast.Name)
    }
    environment.update(
        {argument: value, "cutlass": SimpleNamespace(Float32=float, BFloat16=float)}
    )
    actual = eval(
        compile(ast.Expression(expression), "<epilogue>", "eval"), environment
    )
    assert actual == 2.0 + value
    assert source == bound.to_code(config)


@pytest.mark.parametrize("take_branch", [False, True])
def test_conditional_epilogue_reserves_boundaries_without_mutating_ast(
    take_branch: bool,
) -> None:
    original = ast.parse("""
m = m0
value = acc
if _helion_rc_0_tid:
    value = value + _helion_rc_1_arena
""")
    original_dump = ast.dump(original)
    expression = ast.Name("value", ast.Load())
    recipe = _scalar_epilogue(
        original.body,
        expression,
        m_index="m",
        n_index="n",
        dominating=[],
        boundaries={"m0", "acc", "_helion_rc_0_tid", "_helion_rc_1_arena"},
        fresh_name=_fresh(),
    )
    assert recipe is not None
    boundaries = {"acc", "_helion_rc_0_tid", "_helion_rc_1_arena"}
    assert boundaries <= recipe.boundary_names
    assert "value" not in recipe.boundary_names
    sites, epilogue, facts = _chain()
    plan = plan_register_chain(
        sites,
        replace(epilogue, value_recipe=recipe),
        thread_count=64,
        tensor_facts=facts,
    )
    assert not (boundaries & set(plan._generated_names))
    emit_register_chain(plan, prefix="chain", tid="actual_tid", fresh_name=_fresh())
    statements, value = recipe.emit({}, _fresh())
    module = ast.fix_missing_locations(
        ast.Module(
            [*statements, ast.Assign([ast.Name("result", ast.Store())], value)], []
        )
    )
    environment = {
        "acc": 2.0,
        "_helion_rc_0_tid": take_branch,
        "_helion_rc_1_arena": 3.25,
    }
    exec(compile(module, "<conditional-epilogue>", "exec"), environment)
    assert environment["result"] == (5.25 if take_branch else 2.0)
    assert ast.dump(original) == original_dump
    assert ast.dump(expression) == ast.dump(ast.Name("value", ast.Load()))
