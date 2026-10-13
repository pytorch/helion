from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_matmul import plan_chained_matmul
from helion._compiler.cute.chained_scan_input import plan_scan_inputs
from helion._compiler.generate_ast import GenerateAST
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_root_vector_cache import RootVectorCache


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _raw_root_with_prefix(raw, scale, left, right):
    rows = hl.specialize(left.size(1))
    columns = hl.specialize(right.size(2))
    output = torch.empty(
        (raw.size(0), rows, columns), device=raw.device, dtype=torch.float32
    )
    for batch in hl.tile(raw.size(0), block_size=1):
        index = hl.arange(rows)
        values = hl.load(raw, [batch.begin, index], index < raw.size(1)).float()
        nonlinear = torch.sigmoid(values)
        steps = torch.where(index < raw.size(1), scale[batch.begin] * nonlinear, 0.0)
        prefix = hl.cumsum(steps, 0)
        lhs = left[batch.begin, :, :]
        energy = (lhs.float() * lhs.float()).sum(1)
        result = hl.dot(lhs, right[batch.begin, :, :])
        output[batch.begin, :, :] = (
            result * torch.exp2(prefix[:, None]) + energy[:, None] * 0.125
        )
    return output


@pytest.mark.parametrize("rows", [64, 128])
def test_native_root_scan_input_keeps_joint_schedule_receipt(rows) -> None:
    args = (
        torch.empty((3, rows - 7), dtype=torch.bfloat16),
        torch.empty((3,), dtype=torch.float32),
        torch.empty((3, rows, 32), dtype=torch.bfloat16),
        torch.empty((3, 32, 64), dtype=torch.bfloat16),
    )
    with _cpu_codegen():
        bound = _raw_root_with_prefix._bind_isolated(args)
        config = bound.config_spec.normalized_config(
            {
                "num_warps": 4,
                "cute_chained_mma_schedule": "cp_async",
                "cute_chained_root_native": True,
                "cute_chained_scan_input_materialize": True,
            }
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            # Build only graph copies and the metadata context. No visitor or
            # source generation is needed to prove admission and receipt scope.
            context = GenerateAST(bound.host_function, config)
            with context.device_function:
                plan = plan_chained_matmul(context.codegen_graphs)
                assert plan is not None and plan.root_native is not None
                assert plan.strategy == "mixed_root" and len(plan.scan_inputs) == 1
                plan.root_native.check(plan)
                selected = plan.scan_inputs[0]
                selected.check(plan)
                assert not plan_scan_inputs(replace(plan, root_native=None))
                with pytest.raises(_UnsupportedChain, match="changed"):
                    selected.check(replace(plan, root_native=replace(plan.root_native)))
                with pytest.raises(_UnsupportedChain, match="changed"):
                    selected.check(
                        replace(
                            plan, root_vector_cache=cast("RootVectorCache", object())
                        )
                    )
            assert not context.host_statements and not context.module_statements
            assert not context.device_function.body
            assert not context.device_function.preamble
