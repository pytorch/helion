from __future__ import annotations

import ast
import copy
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_computed_scan import _args
from .test_cute_computed_scan import _prefix
import helion
from helion import exc
from helion._compiler.cute.computed_scan_root import ComputedScanEmitter
from helion._compiler.cute.computed_scan_root import ComputedScanPlan
from helion._compiler.cute.positional_root import _Unsupported
from helion._compiler.cute.positional_root import codegen_positional_root
from helion._compiler.cute.unroll_lane_loads import LaneUnrollNotApplied
from helion._compiler.cute.unroll_lane_loads import unroll_lane_loads
from helion._compiler.device_function import DeviceFunction
from helion._compiler.generate_ast import GenerateAST


def _late_unroll(body, df):
    # Exercise the actual late-pass block, without running whole-kernel emission.
    tree = ast.parse(inspect.getsource(inspect.getmodule(DeviceFunction)))
    statement = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "lane_unroll > 1 and (not self.config.config.get('cute_vloop_sink'))"
    )
    scope = {
        "self": df,
        "kernel_body": copy.deepcopy(body),
        "lane_unroll": df.config.config.get("cute_lane_unroll", 1),
        "unroll_lane_loads": unroll_lane_loads,
        "LaneUnrollNotApplied": LaneUnrollNotApplied,
        "proven_disjoint_tensor_pairs": frozenset(),
        "rename_groups": {},
    }
    exec(
        compile(ast.Module([copy.deepcopy(statement)], []), "<late-unroll>", "exec"),
        scope,
    )
    return scope["kernel_body"]


def check_active_path(context, plan, unroll):
    assert isinstance(plan, ComputedScanPlan) and plan.staging is not None
    df = context.device_function
    assert not df.cute_state.materialization_unroll_applied
    # The scalar expression is irrelevant to scheduling. Keep the real plan,
    # staging layout, emit_scan entry, installation, and late retry decision.
    expression = SimpleNamespace(lines=[], value=lambda node, coords: "input_value")
    with patch.object(ComputedScanEmitter, "expression", return_value=expression):
        emitter = ComputedScanEmitter(context, plan)
        lines = emitter.emit_scan(plan.scan)
        loops = [
            n for n in ast.walk(ast.parse("\n".join(lines))) if isinstance(n, ast.For)
        ]
        assert (
            next(k.value.value for k in loops[0].iter.keywords if k.arg == "unroll")
            == unroll
        )
        assert (
            next(k.value.value for k in loops[1].iter.keywords if k.arg == "unroll")
            == 1
        )
        export, coords = emitter.element_loop((64, 16))
        assert export[0].endswith("unroll=1):")
        copy_lines = emitter.copy_tile(plan.scan.args[1], "src", "dst")
        assert copy_lines[0].endswith("unroll=1):")
        assert not df.cute_state.materialization_unroll_applied
        separate = ComputedScanEmitter(context, plan)
        separate.copy_tile(plan.scan.args[1], "src", "dst")
        assert not separate.materialization_unroll_applied
        written = separate.write_tile(plan.scan.args[1], "dst")
        assert written[0].endswith(f"unroll={unroll}):")
        assert separate.materialization_unroll_applied == (unroll > 1)

        def failed(owner):
            owner.materialization_loop((64, 16))
            raise _Unsupported("failed after staging")

        with (
            patch.object(ComputedScanEmitter, "emit", failed),
            pytest.raises(exc.BackendUnsupported),
        ):
            codegen_positional_root(context, plan)
        assert not df.cute_state.materialization_unroll_applied
        # Only the ordinary body-installation entry publishes the receipt.
        with patch.object(
            ComputedScanEmitter, "emit", lambda owner: owner.emit_scan(plan.scan)
        ):
            codegen_positional_root(context, plan)
    assert df.cute_state.materialization_unroll_applied == (unroll > 1)
    before = ast.dump(ast.Module(df.body, []))
    assert ast.dump(ast.Module(_late_unroll(df.body, df), [])) == before
    if unroll > 1:
        # This is the old v2 failure: the same real staged body is discarded
        # when only grid-lane unrolling counts as having consumed the option.
        df.cute_state.materialization_unroll_applied = False
        with pytest.raises(LaneUnrollNotApplied):
            _late_unroll(df.body, df)
        df.cute_state.materialization_unroll_applied = True


@pytest.mark.parametrize("unroll", [1, 2, 4, 8])
def test_staged_scan_consumes_unroll_before_late_retry(unroll):
    with _cpu_codegen():
        bound = _prefix._bind_isolated((*_args(), 64, False))
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[16],
                num_threads=[4, 0],
                cute_computed_scan_schedule="staged_warp",
                cute_lane_unroll=unroll,
            )
        )
        with bound.env, bound.host_function, bound._runtime_arg_values_for_codegen():
            context = GenerateAST(bound.host_function, config)
            with context.device_function:
                bound.env.backend.pre_codegen(
                    context.codegen_graphs,
                    config,
                    context.device_function.tile_strategy,
                )
                check_active_path(
                    context,
                    context.device_function.cute_state.positional_root_plan,
                    unroll,
                )
