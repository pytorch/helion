"""CPU source coverage for state returned by nested MMA loops."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute.cute_mma_dag_tcgen05_regions import step_depends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _nested_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(3))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], device=a.device, dtype=torch.float32)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for outer in hl.grid(0, bounds[0]):
            for inner in hl.grid(0, bounds[1]):
                acc = hl.dot(a[tm, :], b[outer, inner, :, :], acc=acc)
        out[tm, :] = acc
    return out


def _nested_online(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bounds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, d = q.shape
    hl.specialize(k.size(3))
    n = hl.specialize(v.size(3))
    hl.specialize(d)
    out = torch.empty([m, n], device=q.device, dtype=torch.float32)
    out_max = torch.empty([m], device=q.device, dtype=torch.float32)
    out_sum = torch.empty([m], device=q.device, dtype=torch.float32)
    for tm in hl.tile(m):
        maximum = hl.full([tm], float("-inf"), dtype=torch.float32)
        denominator = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for outer in hl.grid(0, bounds[0]):
            for inner in hl.grid(0, bounds[1]):
                scores = hl.dot(q[tm, :], k[outer, inner, :, :])
                next_max = torch.maximum(maximum, torch.amax(scores, dim=1))
                alpha = torch.exp(maximum - next_max)
                weights = torch.exp(scores - next_max[:, None])
                acc = hl.dot(
                    weights.to(q.dtype), v[outer, inner, :, :], acc=acc * alpha[:, None]
                )
                denominator = denominator * alpha + torch.sum(weights, dim=1)
                maximum = next_max
        out[tm, :] = acc
        out_max[tm] = maximum
        out_sum[tm] = denominator
    return out, out_max, out_sum


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("kind", ["fragment", "online"])
def test_child_loop_results_feed_parent_carries(mode: str, kind: str) -> None:
    args: tuple[object, ...] = (
        torch.empty((64, 32), dtype=torch.bfloat16),
        torch.empty((2, 2, 32, 64), dtype=torch.bfloat16),
    )
    if kind == "online":
        args = (*args, torch.empty((2, 2, 64, 128), dtype=torch.bfloat16))
    args = (*args, torch.empty((2,), dtype=torch.int32))
    fn = _nested_matmul if kind == "fragment" else _nested_online
    with cpu_target():
        bound = cpu_bind(helion.kernel(fn, backend="cute", static_shapes=True), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            assert len(program.loops) == 2
            assert len(program.loops[0].carries) == 1
            if kind == "online":
                assert len(program.loops[0].vec_carries) == 2
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    if kind == "online":
        assert "md_vnext" in source
        if mode == "warp":
            # Alpha is computed in this iteration and may reuse the old max
            # slot. Rescaling the output at iteration entry would read -inf.
            # (``cute.math.exp``: the exact exponential, not exp2(x * log2(e)).)
            assert source.index("cute.math.exp(") < source.index("md_c0_0_0_0 *")


def _nested_swap(
    q: torch.Tensor, a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = q.size(0), hl.specialize(a.size(3))
    hl.specialize(q.size(1))
    out_a = torch.empty([m, n], device=q.device, dtype=torch.float32)
    out_b = torch.empty_like(out_a)
    for tm in hl.tile(m):
        left = hl.zeros([tm, n], dtype=torch.float32)
        right = hl.zeros([tm, n], dtype=torch.float32)
        for outer in hl.grid(0, bounds[0]):
            for inner in hl.grid(0, bounds[1]):
                left = hl.dot(q[tm, :], a[outer, inner, :, :], acc=left)
                right = hl.dot(q[tm, :], b[outer, inner, :, :], acc=right)
            left, right = right, left
        out_a[tm, :] = left
        out_b[tm, :] = right
    return out_a, out_b


def _nested_acc_vector(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = q.size(0), hl.specialize(v.size(3))
    hl.specialize(q.size(1))
    hl.specialize(k.size(3))
    out = torch.empty([m, n], device=q.device, dtype=torch.float32)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for outer in hl.grid(0, bounds[0]):
            for inner in hl.grid(0, bounds[1]):
                score = hl.dot(q[tm, :], k[outer, inner, :, :])
                alpha = torch.sum(score, dim=1) / 64 + 1
                acc = hl.dot(q[tm, :], v[outer, inner, :, :], acc=acc * alpha[:, None])
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("kind", ["swap", "acc_vector"])
def test_nested_outputs_preserve_simultaneous_values_and_vector_dependencies(
    mode: str, kind: str
) -> None:
    args = (
        torch.empty((64, 32), dtype=torch.bfloat16),
        torch.empty((2, 2, 32, 64), dtype=torch.bfloat16),
        torch.empty((2, 2, 32, 64 if kind == "swap" else 128), dtype=torch.bfloat16),
        torch.empty((2,), dtype=torch.int32),
    )
    fn = _nested_swap if kind == "swap" else _nested_acc_vector
    with cpu_target():
        bound = cpu_bind(helion.kernel(fn, backend="cute", static_shapes=True), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            if kind == "acc_vector":
                assert program.dots[1].acc_vecs
                assert step_depends(program, program.dots[1], program.dots[0])
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    if kind == "swap" and mode == "warp":
        assert "md_cnext" in source
    if kind == "swap" and mode == "tcgen05":
        copies = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.If) or ast.unparse(node.test) != "md_epi":
                continue
            operations = [
                call.func.attr
                for statement in node.body
                for call in ast.walk(statement)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in ("tcgen05_ld", "tcgen05_st")
            ]
            if operations.count("tcgen05_ld") >= 2 and "tcgen05_st" in operations:
                copies.append(operations)
        assert len(copies) == 1
        operations = copies[0]
        last_load = max(i for i, op in enumerate(operations) if op == "tcgen05_ld")
        assert last_load < operations.index("tcgen05_st")


def _nested_vector_swap(
    q: torch.Tensor, k: torch.Tensor, bounds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m = q.size(0)
    hl.specialize(q.size(1))
    hl.specialize(k.size(3))
    out_left = torch.empty([m], device=q.device, dtype=torch.float32)
    out_right = torch.empty_like(out_left)
    for tm in hl.tile(m):
        left = hl.full([tm], 1.0, dtype=torch.float32)
        right = hl.full([tm], 3.0, dtype=torch.float32)
        for outer in hl.grid(0, bounds[0]):
            for inner in hl.grid(0, bounds[1]):
                score = hl.dot(q[tm, :], k[outer, inner, :, :])
                update = torch.sum(score, dim=1)
                left = left + update
                right = right + update * 2
            left, right = right, left
        out_left[tm] = left
        out_right[tm] = right
    return out_left, out_right


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_vector_carry_cycle_reads_both_sources_before_writing(mode: str) -> None:
    args = (
        torch.empty((64, 32), dtype=torch.bfloat16),
        torch.empty((2, 2, 32, 64), dtype=torch.bfloat16),
        torch.empty((2,), dtype=torch.int32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_nested_vector_swap, backend="cute", static_shapes=True), args
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    reads = [
        line
        for line in source.splitlines()
        if "md_vnext" in line and line.strip().startswith("md_vnext")
    ]
    writes = [
        line
        for line in source.splitlines()
        if "md_vnext" in line and not line.strip().startswith("md_vnext")
    ]
    assert len(reads) >= 2 and len(writes) >= 2
    assert source.index(reads[-1]) < source.index(writes[0])


def _loaded_tile_vector_operand(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(1), hl.specialize(b.size(2))
    hl.specialize(a.size(2))
    out = torch.empty([m, n], device=a.device, dtype=torch.float32)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for page in hl.grid(2):
            seed = hl.dot(a[page, tm, :], b[page, :, :])
            scale = torch.sum(seed, dim=1)
            tiles = a[page, tm, :].float() + c[page, tm, :].float()
            value = tiles + scale[:, None]
            acc = hl.dot(value.to(a.dtype), b[page, :, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_a_tile_pass_chain_cannot_consume_a_program_vector(mode: str) -> None:
    args = tuple(torch.empty((2, 64, 64), dtype=torch.bfloat16) for _ in range(3))
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _loaded_tile_vector_operand, backend="cute", static_shapes=True
            ),
            args,
        )
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            value = next(step for step in program.derived if step.vecs)
            assert len(value.frags) == 1
            tiles = next(
                step for step in program.derived if step.node is value.frags[0]
            )
            # Both sources meet the tile-pass requirements, and this value's
            # only consumer is a dot in its own innermost loop. The vector
            # dependency is what keeps it in fragment registers.
            assert len(tiles.loads) == 2 and not tiles.frags and not tiles.vecs
            assert all(program.ring_derived_load(tiles, load) for load in tiles.loads)
            assert not program.pass_chains
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=8 if mode == "warp" else 4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
