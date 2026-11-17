"""Pointwise loop indices and masked column state in MMA recurrences."""

from __future__ import annotations

import ast
import re

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _masked_column_state(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bounds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, dim = q.shape
    tokens = hl.specialize(k.size(1))
    width = hl.specialize(v.size(2))
    hl.specialize(dim)
    out = torch.empty([rows, width], device=q.device, dtype=torch.float32)
    maximum_out = torch.empty([rows], device=q.device, dtype=torch.float32)
    denom_out = torch.empty_like(maximum_out)
    for tq in hl.tile(rows):
        maximum = hl.full([tq], float("-inf"), dtype=torch.float32)
        denominator = hl.zeros([tq], dtype=torch.float32)
        accumulator = hl.zeros([width, tq], dtype=torch.float32)
        for page in hl.grid(bounds[0], bounds[1]):
            for tk in hl.tile(tokens):
                score = hl.dot(k[page, tk, :], q[tq, :].T)
                position = page * tokens + tk.index
                visible = position[:, None] <= bounds[2] + tq.index[None, :]
                score = torch.where(visible, score, float("-inf"))
                nxt = torch.maximum(maximum, torch.amax(score, dim=0))
                safe = torch.where(nxt == float("-inf"), 0.0, nxt)
                alpha = torch.exp(maximum - safe)
                p = torch.exp(score - safe[None, :])
                denominator = denominator * alpha + torch.sum(p, dim=0)
                accumulator = hl.dot(
                    v[page, tk, :].T, p.to(q.dtype), acc=accumulator * alpha[None, :]
                )
                maximum = nxt
        out[tq, :] = accumulator.T
        maximum_out[tq] = maximum
        denom_out[tq] = denominator
    return out, maximum_out, denom_out


# The larger key block makes a 128x64 score fragment: its column-vector
# stores must exclude the extra 64 rows of the fragment's vector layout.
@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("key_block", [64, 128])
def test_masked_column_state_and_loop_index_recipe(mode: str, key_block: int) -> None:
    args = (
        torch.empty((64, 32), dtype=torch.bfloat16),
        torch.empty((2, 128, 32), dtype=torch.bfloat16),
        torch.empty((2, 128, 128), dtype=torch.bfloat16),
        torch.empty((3,), dtype=torch.int64),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_masked_column_state, backend="cute", static_shapes=True),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag._Matcher(bound.host_function.device_ir.graphs)._match()
            assert any("128*" in str(expr) for expr in program.aux_indices.values())
            assert all(
                step.node.meta["val"].dtype == torch.float32 for step in program.vec_ops
            )
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64, key_block],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=8 if mode == "warp" and key_block == 128 else 4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    assert "128 * md_i" in source or "md_i * 128" in source
    if mode == "tcgen05" and key_block == 128:
        assert re.search(r"if [^:\n]+ < 64:\n[^\n]*stg_f32", source)


def test_tcgen05_root_blocks_must_divide_the_extent() -> None:
    args = (
        torch.empty((47, 32), dtype=torch.bfloat16),
        torch.empty((2, 128, 32), dtype=torch.bfloat16),
        torch.empty((2, 128, 128), dtype=torch.bfloat16),
        torch.empty((3,), dtype=torch.int64),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_masked_column_state, backend="cute", static_shapes=True),
            args,
        )
        with (
            bound.env,
            pytest.raises(helion.exc.InvalidConfig, match="must divide the extent 47"),
        ):
            bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64, 128],
                    cute_mma_dag="tcgen05",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )


def _loop_index_operand_pass(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], device=a.device, dtype=torch.float32)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(2):
            value = a[tm, :] * (step * 2 + 1)
            acc = hl.dot(value, b[:, :], acc=acc)
        out[tm, :] = acc
    return out


def test_loop_index_recipe_does_not_enter_prefetched_operand_passes() -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_loop_index_operand_pass, backend="cute", static_shapes=True),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert dag.match_mma_dag(bound.host_function.device_ir.graphs) is None


def _iteration_table(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    m, n = a.size(1), hl.specialize(b.size(2))
    hl.specialize(a.size(2))
    out = torch.empty([m, n], device=a.device, dtype=torch.float32)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(4):
            alpha = torch.exp(g[step, tm] + 2 * step)
            projected = hl.dot(a[step, tm, :], b[step, :, :])
            second = hl.dot(a[step, tm, :], b[step, :, :], acc=projected)
            scaled = projected * alpha[:, None] + second
            acc = hl.dot(scaled.to(a.dtype), b[step, :, :], acc=acc)
        out[tm, :] = acc
    return out


def test_single_loop_tables_use_the_prefetched_iteration() -> None:
    args = (
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
        torch.empty((4, 64), dtype=torch.float32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_iteration_table, backend="cute", static_shapes=True), args
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag="tcgen05",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "md_tb" in source
    assert "2 * (md_i + 1)" in source


def _column_side_input(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n = b.size(2)
    hl.specialize(a.size(1))
    hl.specialize(a.size(2))
    out = torch.empty([n], device=a.device, dtype=torch.float32)
    total = torch.empty([1], device=a.device, dtype=torch.float32)
    for tn in hl.tile(n):
        maximum = hl.full([tn], float("-inf"), dtype=torch.float32)
        for step in hl.grid(2):
            value = hl.dot(a[step, :, :], b[step, :, tn])
            maximum = torch.maximum(maximum, torch.amax(value, dim=0))
        result = maximum + bias[tn]
        out[tn] = result
        total[0] = torch.sum(result, dim=0)
    return out, total


def test_column_vector_side_input_is_guarded_to_its_extent() -> None:
    args = (
        torch.empty((2, 128, 32), dtype=torch.bfloat16),
        torch.empty((2, 32, 64), dtype=torch.bfloat16),
        torch.empty((64,), dtype=torch.float32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_column_side_input, backend="cute", static_shapes=True), args
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag="tcgen05",
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("ldg_f32")
    ]
    assert loads
    for load in loads:
        node = load
        guards = []
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                guards.append(ast.unparse(node.test))
        assert any("md_row128 < 64" in guard for guard in guards), guards
