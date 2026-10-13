from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_work_order_scalar import _ragged_sum
import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.work_order import UnsupportedWorkOrder
from helion._compiler.cute.work_order import discover_work_order
from helion._compiler.cute.work_order import emit_work_permutation
from helion._compiler.generate_ast import GenerateAST


def test_actual_ordinary_admission(tmp_path):
    original = GenerateAST._try_lower_direct_affine_root
    observed = []

    def observe(cg, grid, body):
        decision = discover_work_order(cg, grid.block_ids[0])
        decision.check(cg)
        observed.append((decision.count, decision.step, decision.range32))
        permutation = emit_work_permutation(
            cg,
            decision,
            {
                axis: ast.Name(id=grid.strategy.index_var(axis), ctx=ast.Load())
                for axis in decision.leaf_axes
            },
        )
        (tmp_path / "permutation.py").write_text(
            ast.unparse(ast.Module(body=list(permutation.statements), type_ignores=[]))
        )
        return original(cg, grid, body)

    args = (
        torch.empty((5, 64)),
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64) + 32,
        torch.empty(5),
    )
    with (
        _cpu_codegen(),
        patch.object(GenerateAST, "_try_lower_direct_affine_root", observe),
    ):
        bound = _ragged_sum._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_ragged_sum, args)):
            source = bound.to_code(
                helion.Config(block_sizes=[32], num_warps=4, pid_type="flat")
            )
    assert observed == [(5, 32, True)]
    (tmp_path / "source.py").write_text(source)


def test_actual_fused_admission(tmp_path):
    original = chain._install_chained_body
    observed = []

    def observe(cg, plan, lines):
        decisions = []
        for axis, _extent, block in plan.axes:
            if block == 1:
                try:
                    decision = discover_work_order(cg, axis)
                except UnsupportedWorkOrder as error:
                    if "does not depend" in str(error):
                        continue
                    raise
                decision.check(cg)
                decisions.append((decision.count, decision.step, decision.range32))
                permutation = emit_work_permutation(
                    cg,
                    decision,
                    {
                        axis: ast.Name(id=f"chain_origin_{axis}", ctx=ast.Load())
                        for axis in decision.leaf_axes
                    },
                )
                (tmp_path / "permutation.py").write_text(
                    ast.unparse(
                        ast.Module(body=list(permutation.statements), type_ignores=[])
                    )
                )
        observed.append(decisions)
        return original(cg, plan, lines)

    kernel, args = _kda_fixture()
    config = helion.Config(
        block_sizes=[128],
        num_warps=16,
        num_stages=2,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
        cute_chained_warp_mma_rows=32,
    )
    with _cpu_codegen(), patch.object(chain, "_install_chained_body", observe):
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(config)
    assert observed == [[(2, 32, False)]]
    (tmp_path / "source.py").write_text(source)


def test_original_alias_specialization_is_in_actual_bound_cache_key():
    args = (
        torch.empty((5, 64)),
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64) + 32,
        torch.empty(5),
    )
    aliased = (*args[:3], args[1].view(torch.float32)[:5])
    key = "cute_tensor_storage_disjoint_matrix_v1"
    with _cpu_codegen():
        before = _ragged_sum._bind_isolated(args)
        after = _ragged_sum._bind_isolated(aliased)
        assert before._base_spec_key == after._base_spec_key
        before_fact = before.env.bound_runtime_input_specialization_results[key]
        after_fact = after.env.bound_runtime_input_specialization_results[key]
        assert isinstance(before_fact, tuple) and isinstance(after_fact, tuple)
        assert all(before_fact)
        assert not all(after_fact)
        before_key = _ragged_sum._create_bound_kernel_cache_key(
            before, args, before._base_spec_key
        )
        after_key = _ragged_sum._create_bound_kernel_cache_key(
            after, aliased, after._base_spec_key
        )
        assert before_key != after_key


@pytest.mark.parametrize(
    "mutation", ("missing_scope", "aliased_view", "missing_matrix", "changed_bound")
)
def test_actual_runtime_alias_proof_rejects(mutation):
    original = GenerateAST._try_lower_direct_affine_root
    hits = []

    def observe(cg, grid, body):
        env = CompileEnvironment.current()
        key = "cute_tensor_storage_disjoint_matrix_v1"
        decision = discover_work_order(cg, grid.block_ids[0])
        values = dict(env.runtime_arg_values_by_name)
        if mutation == "missing_scope":
            context = env.use_runtime_arg_values({})
        elif mutation == "aliased_view":
            values["result"] = alias_result
            context = env.use_runtime_arg_values(values)
        elif mutation == "missing_matrix":
            context = patch.dict(env.runtime_input_specializations, {}, clear=True)
        else:
            context = patch.dict(
                env.bound_runtime_input_specialization_results, {key: ()}
            )
        with context, pytest.raises(UnsupportedWorkOrder, match="runtime disjointness"):
            decision.check(cg)
        hits.append(mutation)
        return original(cg, grid, body)

    args = (
        torch.empty((5, 64)),
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64) + 32,
        torch.empty(5),
    )
    alias_result = args[1].view(torch.float32)[:5]
    with (
        _cpu_codegen(),
        patch.object(GenerateAST, "_try_lower_direct_affine_root", observe),
    ):
        bound = _ragged_sum._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_ragged_sum, args)):
            bound.to_code(helion.Config(block_sizes=[32], num_warps=4, pid_type="flat"))
    assert hits == [mutation]
