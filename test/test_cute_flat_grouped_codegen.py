from __future__ import annotations

import ast
import dataclasses
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_flat_grouped_ir import _host
from test.test_cute_flat_grouped_ir import cuda_binding

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion._compiler.cute.tcgen05_flat_grouped_codegen import _bound_names
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_kernel_module,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    generate_flat_grouped_native_candidate,
)
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_rna_schedule
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandRolePlan

if TYPE_CHECKING:
    from helion._compiler.cute.tcgen05_flat_grouped_plan import FlatGroupedRnaSchedule
    from helion.runtime.kernel import BoundKernel


def _schedule(
    bound: BoundKernel, warps: int = 4, bk: int = 32
) -> FlatGroupedRnaSchedule | None:
    proof = prove_flat_grouped_rna(bound.env, _host(bound).device_ir)
    assert proof is not None
    return flat_grouped_rna_schedule(
        bound.env,
        proof,
        converter_warps=warps,
        block_k=bk,
        ab_stages={32: 6, 64: 3}[bk],
        shared_capacity=232448,
    )


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[1, 128, 64, 16],
        num_threads=[0, 2, 64, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_lane_layouts=["blocked", "strided", "strided", "strided"],
        cute_collective_mma=True,
        cute_collective_compute="tcgen05",
        cute_collective_stages=1,
        cute_collective_copy="scalar",
        cute_collective_recipe="vector_unrolled",
        cute_collective_epilogue="scalar",
    )


def _candidate(bound: BoundKernel, warps: int = 4, bk: int = 32) -> str:
    def generate(host, config, repro):
        return generate_flat_grouped_native_candidate(
            host,
            config,
            repro,
            converter_warps=warps,
            block_k=bk,
            ab_stages={32: 6, 64: 3}[bk],
            shared_capacity=232448,
        )

    with (
        bound.env.suspend(),
        patch("helion.runtime.kernel.generate_ast", side_effect=generate),
    ):
        return bound.to_code(_config())


def _ordinary(bound: BoundKernel) -> str:
    with bound.env.suspend():
        return bound.to_code(_config())


@pytest.mark.parametrize("warps", [1, 8])
@pytest.mark.parametrize("bk", [32, 64])
def test_work_record_release_joins_all_reader_lanes(warps: int, bk: int) -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        schedule = _schedule(bound, warps, bk)
        assert schedule is not None
        tree = ast.parse(flat_grouped_kernel_module(schedule))

    def is_call(node: ast.AST, target: str) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == target
        )

    releases = 0
    for node in ast.walk(tree):
        for _, statements in ast.iter_fields(node):
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements):
                if not (
                    isinstance(statement, ast.If)
                    and len(statement.body) == 1
                    and is_call(
                        statement.body[0], "tcgen05_sched_pipeline.consumer_release"
                    )
                ):
                    continue
                # A single lane acknowledges the whole warp. Its arrival must
                # follow all lanes' shared loads, including the terminal flag.
                assert ast.unparse(statement.test) == (
                    "cute.arch.lane_idx() == cutlass.Int32(0)"
                )
                assert index > 0 and is_call(
                    statements[index - 1], "cute.arch.sync_warp"
                )
                releases += 1
    # TMA, conversion, MMA and epilogue each have work and terminal releases.
    assert releases == 8


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
@pytest.mark.parametrize("bk", [32, 64])
def test_original_static_schedule_preserves_physical_budget_and_arrivals(
    warps: int, bk: int
) -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        schedule = _schedule(bound, warps, bk)
        assert schedule is not None
        assert schedule.thread_block == (32, ((7 + warps + 3) // 4) * 4, 1)
        assert schedule.roles.scheduler_empty_warp_arrivals == 6 + warps
        assert schedule.roles.converted_full_lane_arrivals == 32 * warps
        assert schedule.storage.bytes + 1024 == (232060 if bk == 32 else 231988)
        assert len(schedule.storage.placements) == 13
        assert (
            json.loads(json.dumps(schedule.cache_identity()))[0]
            == "flat_grouped_rna_schedule_v1"
        )
        assert "quack" not in flat_grouped_kernel_module(schedule).lower()
        with bound.env.use_runtime_arg_values({}):
            assert _schedule(bound, warps, bk) == schedule
        for construct in (
            lambda: dataclasses.replace(schedule, reserved_shared=0),
            lambda: dataclasses.replace(
                schedule, shared_capacity=schedule.storage.bytes + 1023
            ),
            lambda: dataclasses.replace(
                schedule,
                roles=Tcgen05OperandRolePlan(warps, scheduler_self_consumer=True),
            ),
            lambda: dataclasses.replace(schedule, ab_stages=5),
            lambda: dataclasses.replace(schedule, block_k=16),
        ):
            with pytest.raises(ValueError):
                construct()


@pytest.mark.parametrize(
    "case",
    [
        "dynamic",
        "misalign_a",
        "misalign_b",
        "misalign_bias",
        "offset_stride",
        "n_tail",
        "k_tail",
        "prefix_capacity",
        "ieee",
        "tf32x3",
    ],
)
def test_unsupported_bindings_retain_ordinary_schedule(case: str) -> None:
    with cuda_binding(
        static=case != "dynamic",
        original=True,
        groups=1024 if case == "prefix_capacity" else 3,
        rows=273,
        k=72 if case == "k_tail" else 128,
        n=130 if case == "n_tail" else 128,
        misalign={"misalign_a": 1, "misalign_b": 2, "misalign_bias": 3}.get(case),
        offsets_stride=2 if case == "offset_stride" else 1,
        dot_precision=case if case in ("ieee", "tf32x3") else "tf32",
    ) as bound:
        assert _schedule(bound) is None
        # These representative fallbacks compile through their actual binding,
        # including base-alignment specialization. No old aligned source is used.
        if case in (
            "misalign_a",
            "misalign_b",
            "misalign_bias",
            "offset_stride",
            "n_tail",
            "k_tail",
        ):
            ordinary = _ordinary(bound)
            assert _candidate(bound) == ordinary


def test_original_host_allocation_views_and_return_survive_launch_replacement() -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        ordinary = ast.parse(_ordinary(bound))
        candidate = ast.parse(_candidate(bound))
        name = _host(bound).name
        old_host = next(
            n
            for n in ordinary.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        new_host = next(
            n
            for n in candidate.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        old_launch = next(
            n
            for n in old_host.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "_launcher"
        )
        launch_index = next(
            i
            for i, n in enumerate(new_host.body)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "_launcher"
        )
        new_launch = new_host.body[launch_index]
        assert isinstance(new_launch, ast.Expr) and isinstance(
            new_launch.value, ast.Call
        )
        assert ast.unparse(new_launch.value.args[2]) == "seq_offsets"
        assert ast.unparse(new_launch.value.args[3]) == "jagged.view((30856, 128))"
        assert ast.unparse(new_launch.value.args[4]) == "dense.transpose(1, 2)"
        assert ast.unparse(new_launch.value.args[6]) == "output.view((30856, 128))"
        # Restore the one old launch in place of the three host scheduling and
        # workspace statements plus the new launch. Every original AST survives.
        new_host.body[launch_index - 3 : launch_index + 1] = [old_launch]
        assert ast.dump(new_host, include_attributes=False) == ast.dump(
            old_host, include_attributes=False
        )


def test_runtime_alignment_facts_rebind_before_reusing_a_native_schedule() -> None:
    with cuda_binding(
        static=True, original=True, groups=3, rows=273, k=128, n=128
    ) as bound:
        assert _schedule(bound) is not None
        names = tuple(bound.kernel.signature.parameters)
        values = []
        for name in names:
            value = bound._runtime_tensor_refs_by_name[name]()
            assert isinstance(value, torch.Tensor)
            values.append(value)
        with bound.env.suspend():
            fresh = tuple(
                value.clone().requires_grad_(value.requires_grad) for value in values
            )
            assert bound.kernel.bind(fresh) is bound
            left = fresh[1]
            shifted = torch.empty(left.numel() + 1, dtype=left.dtype)[1:].view(
                left.shape
            )
            shifted.requires_grad_(left.requires_grad)
            shifted_values = (fresh[0], shifted, *fresh[2:])
            other = bound.kernel.bind(shifted_values)
            assert other is not bound
            with other.env, _host(other):
                assert _schedule(other) is None
                assert "_helion_cute_native_grouped_plan" not in _candidate(other)
            assert bound.kernel.bind(fresh) is bound


def test_fresh_namespace_includes_arguments_and_non_name_bindings() -> None:
    tree = ast.parse("""
import torch as _helion_native_tensor_library
from helion.runtime import get_num_sm as _helion_native_get_num_sm
def owner(_helion_native_grouped_module, *, _helion_native_grid_z=0):
    global _helion_native_code_cache
    try:
        pass
    except Exception as _helion_native_tensormap_workspace:
        pass
""")
    assert {
        "_helion_native_tensor_library",
        "_helion_native_get_num_sm",
        "_helion_native_grouped_module",
        "_helion_native_grid_z",
        "_helion_native_code_cache",
        "_helion_native_tensormap_workspace",
    } <= _bound_names(tree)
