from __future__ import annotations

import ast
import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import sympy
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_native_matmul_metadata import _case
from .test_cute_native_matmul_metadata import _stage
from .test_cute_native_scan_input_metadata import _raw_root_with_prefix
from helion import exc
from helion._compiler.cute import chained_coordinate_bounds as bounds
from helion._compiler.cute import native_matmul_metadata as native
from helion._compiler.generate_ast import GenerateAST

SOURCE = """
thread = cutlass.Int32(cute.arch.thread_idx()[0])
block = cutlass.Int32(cute.arch.block_idx()[0])
if thread < 128 and block < 4:
    native_stage(thread, block)
if block * 3 + thread < 5:
    tail(thread, block)
"""


def _context():
    cg, plan = _case()
    plan.axes = ((0, 2, 1), (1, 5, 3))
    plan.strategy = "warp"
    plan.root_native = None
    plan.loop = plan.preparation_pipeline = None
    plan.warp_loop_carry = None
    cg.device_function.config.pid_type = "flat"
    cg.device_function.config.config["cute_chained_operand_coordinates"] = True
    cg.device_function._variable_renames = {}
    cg.device_function.cute_state = SimpleNamespace(
        chained_matmul_plan=plan,
        simt_cluster_n=1,
        direct_affine_plan=None,
        collective_register_chain_block_dims=None,
    )
    return cg, plan


def _execute(body, thread, block):
    calls = []
    namespace = {
        "cutlass": SimpleNamespace(Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (thread, 0, 0),
                block_idx=lambda: (block, 0, 0),
            )
        ),
        "native_stage": lambda *args: calls.append(("native", args)),
        "tail": lambda *args: calls.append(("tail", args)),
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<row-bounds>", "exec"),
        namespace,
    )
    return calls


def test_proved_mixed_root_keeps_actual_launch_and_native_owner() -> None:
    args = (
        torch.empty((3, 57), dtype=torch.bfloat16),
        torch.empty((3,), dtype=torch.float32),
        torch.empty((3, 64, 32), dtype=torch.bfloat16),
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
            context = GenerateAST(bound.host_function, config)
            with context.device_function:
                bound.env.backend.pre_codegen(
                    context.codegen_graphs,
                    config,
                    context.device_function.tile_strategy,
                )
                state = context.device_function.cute_state
                plan = state.chained_matmul_plan
                assert plan is not None and plan.root_native is not None
                assert plan.strategy == "mixed_root"
                assert bounds._launch(context, plan) == ((128, 1, 1), (3, 1, 1))
                assert bounds._launch(context, replace(plan, root_native=None)) is None
                assert bounds._launch(context, replace(plan, strategy="warp")) is None
                state.chained_matmul_plan = replace(plan)
                assert bounds._launch(context, plan) is None
                state.chained_matmul_plan = plan
                state.simt_cluster_n = 2
                assert bounds._launch(context, plan) is None
            assert not context.host_statements and not context.module_statements
            assert not context.device_function.body
            assert not context.device_function.preamble


def test_actual_launch_preserves_every_thread_and_partial_tail():
    cg, plan = _context()
    original = ast.parse(SOURCE).body
    before = ast.dump(ast.Module(body=original, type_ignores=[]))
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert receipt is not None and receipt.launch == ((128, 1, 1), (4, 1, 1))
    assert ast.dump(ast.Module(body=original, type_ignores=[])) == before
    rendered = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))
    assert "thread < 128" not in rendered
    assert "block * 3 + thread < 5" in rendered
    for block in range(4):
        for thread in range(128):
            assert _execute(original, thread, block) == _execute(
                rewritten, thread, block
            )


@pytest.mark.parametrize(
    "owner,key,value",
    [
        ("plan", "strategy", "tcgen05_tmem"),
        ("plan", "loop", object()),
        ("plan", "preparation_pipeline", object()),
        ("plan", "threads", True),
        ("plan", "threads", 2048),
        ("plan", "axes", ((0, sympy.Symbol("extent"), 1),)),
        ("plan", "axes", ((0, 4, 0),)),
        ("plan", "axes", ((0, 1 << 30, 1), (1, 8, 1))),
        ("state", "simt_cluster_n", 2),
        ("state", "chained_matmul_plan", None),
        ("state", "direct_affine_plan", object()),
        ("state", "collective_register_chain_block_dims", (32, 4, 1)),
        ("config", "pid_type", "persistent_blocked"),
        ("cg", "cute_wrapper_plans", [object()]),
    ],
)
def test_uncertain_or_unrequested_root_is_unchanged(owner, key, value):
    cg, plan = _context()
    objects = {
        "plan": plan,
        "state": cg.device_function.cute_state,
        "config": cg.device_function.config,
        "cg": cg,
    }
    setattr(objects[owner], key, value)
    original = ast.parse(SOURCE).body
    before = ast.dump(ast.Module(body=original, type_ignores=[]))
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert rewritten is original and receipt is None
    assert ast.dump(ast.Module(body=original, type_ignores=[])) == before


def test_pending_alias_groups_cannot_supply_early_proofs():
    cg, plan = _context()
    cg.device_function._variable_renames = {"renamed": ["thread", "block"]}
    original = ast.parse(SOURCE).body
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert rewritten is original and receipt is None


@pytest.mark.parametrize(
    "statement",
    [
        "import math as thread",
        "from sys import maxsize as thread",
        "if unknown:\n    import math as thread",
    ],
)
def test_imported_bindings_are_excluded_without_moving_imports(statement):
    cg, plan = _context()
    source = SOURCE + statement + "\n"
    original = ast.parse(source).body
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert receipt is not None
    rendered = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))
    assert "thread < 128" in rendered
    assert "block < 4" not in rendered
    assert ast.dump(rewritten[-1]) == ast.dump(original[-1])


@pytest.mark.parametrize(
    "statement",
    [
        "from module import *",
        "import module as cute",
        "from module import range",
        "import cutlass.something",
    ],
)
def test_unknown_import_namespace_declines(statement):
    cg, plan = _context()
    original = ast.parse(SOURCE + statement + "\n").body
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert rewritten is original and receipt is None


def test_proof_marker_does_not_rebind_an_import():
    cg, plan = _context()
    original = ast.parse(
        SOURCE + "import math as _cute_proven_loop_bounds_0_abi_version\n"
    ).body
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert receipt is not None
    assert (
        cast("ast.Name", cast("ast.Assign", rewritten[0]).targets[0]).id
        == "_cute_proven_loop_bounds_1_abi_version"
    )


@pytest.mark.parametrize(
    "mutation", ["before", "after", "config", "plan", "device", "launch", "renames"]
)
def test_native_commit_rejects_stale_or_reused_canonicalization(mutation):
    cg, plan = _context()
    original = ast.parse(SOURCE).body
    with native.body_attempt(cg):
        _stage(cg, plan, (SOURCE,))
        rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
        assert receipt is not None
        cg.device_function.body = rewritten
        if mutation == "before":
            original.append(ast.parse("wrong_before()").body[0])
        elif mutation == "after":
            rewritten.append(ast.parse("wrong_after()").body[0])
        elif mutation == "config":
            cg.device_function.config.config["cute_chained_operand_coordinates"] = False
        elif mutation == "plan":
            plan = copy.copy(plan)
            cg.device_function.cute_state.chained_matmul_plan = plan
        elif mutation == "device":
            cg.device_function = copy.copy(cg.device_function)
        elif mutation == "launch":
            plan.axes = ((0, 8, 1),)
        else:
            cg.device_function._variable_renames["thread"] = ["new"]
        with pytest.raises(exc.BackendUnsupported, match="coordinate bounds proof"):
            native.commit_body(
                cg, plan, (SOURCE,), coordinate_bounds=receipt, original_body=original
            )


def test_native_commit_retains_canonicalization_and_later_tamper_check():
    cg, plan = _context()
    original = ast.parse(SOURCE).body
    with native.body_attempt(cg):
        _stage(cg, plan, (SOURCE,))
        cg.device_function.body, receipt = bounds.canonicalize_coordinate_bounds(
            cg, plan, original
        )
        native.commit_body(
            cg, plan, (SOURCE,), coordinate_bounds=receipt, original_body=original
        )
    assert native.validate_native_metadata(cg)
    assert cg.cute_matmul_layouts.committed[0].coordinate_bounds is receipt
    cg.device_function.body.append(ast.parse("uncovered_effect()").body[0])
    with pytest.raises(exc.BackendUnsupported, match="body changed"):
        native.validate_native_metadata(cg)


def test_absent_false_true_preserve_effects_under_the_same_launch_proof():
    results = []
    for value in (None, False, True):
        cg, plan = _context()
        if value is None:
            cg.device_function.config.config.pop("cute_chained_operand_coordinates")
        else:
            cg.device_function.config.config["cute_chained_operand_coordinates"] = value
        original = ast.parse(SOURCE).body
        original_dump = ast.dump(ast.Module(body=original, type_ignores=[]))
        rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
        assert receipt is not None and receipt.launch == ((128, 1, 1), (4, 1, 1))
        assert ast.dump(ast.Module(body=original, type_ignores=[])) == original_dump
        for block in range(4):
            for thread in range(128):
                assert _execute(original, thread, block) == _execute(
                    rewritten, thread, block
                )
        results.append(ast.dump(ast.Module(body=rewritten, type_ignores=[])))
    assert results[0] == results[1] == results[2]


def test_runtime_and_overflow_predicates_remain():
    cg, plan = _context()
    source = (
        SOURCE
        + "if thread < runtime_limit and thread + 2147483647 > 0:\n    guarded(thread)\n"
    )
    original = ast.parse(source).body
    rewritten, receipt = bounds.canonicalize_coordinate_bounds(cg, plan, original)
    assert receipt is not None
    assert ast.dump(original[-1]) == ast.dump(rewritten[-1])


def test_existing_static_nonimage_rejects_explicit_request_normally():
    import torch

    from .test_cute_chained_coordinates import _split_view_sequence
    from .test_cute_chained_coordinates import _view_args

    with _cpu_codegen():
        bound = _split_view_sequence._bind_isolated(_view_args(torch.bfloat16))
        spec = bound.config_spec
        default = spec.default_config()
        ordinary = bound.to_code(default)
        assert (
            bound.to_code(
                spec.normalized_config(
                    default.config | {"cute_chained_completed_image": False}
                )
            )
            == ordinary
        )
        with pytest.raises(exc.BackendUnsupported, match="completed image"):
            bound.to_code(
                spec.normalized_config(
                    default.config | {"cute_chained_completed_image": True}
                )
            )
