from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_positional_root import baddbmm_nonsquare
from .test_cute_positional_root import k_promoted_contractions
import helion
from helion import exc
from helion._compiler import generate_ast
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.positional_root import setup_positional_grid
from helion._compiler.program_id import FlatProgramIDs
import helion.language as hl
from helion.runtime.kernel import BoundKernel


def _cases():
    return [
        (
            baddbmm_nonsquare,
            (
                torch.empty(3, 8, 20),
                torch.empty(3, 20, 6),
                torch.empty(3, 8, 6),
                torch.empty(3, 8, 8),
            ),
        ),
        (k_promoted_contractions, (torch.empty(6, 20), torch.empty(6, 4, 4))),
    ]


@contextmanager
def _metadata(kernel, args):
    with (
        _cpu_codegen(),
        patch.object(BoundKernel, "to_code", side_effect=AssertionError("source")),
        patch.object(
            generate_ast, "generate_ast", side_effect=AssertionError("source")
        ),
        patch.object(
            generate_ast, "_generate_ast", side_effect=AssertionError("source")
        ),
    ):
        bound = kernel._bind_isolated(args)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [2]
        config = bound.config_spec.normalized_config(config.config)
        with bound.env, bound.host_function:
            cg = generate_ast.GenerateAST(bound.host_function, config)
            with cg.device_function:
                bound.env.backend.pre_codegen(
                    cg.codegen_graphs, config, cg.device_function.tile_strategy
                )
                plan = cg.device_function.cute_state.positional_root_plan
                assert plan is not None
                cg.current_root_graph_info = cg.get_graph(plan.root_graph_id)
                yield cg, list(bound.host_function.device_ir.grid_block_ids[0])
                assert not cg.device_function.body and not cg.module_statements


@pytest.mark.parametrize("kernel,args", _cases())
def test_positional_grid_binds_without_generic_thread_or_reduction_setup(kernel, args):
    with _metadata(kernel, args) as (cg, blocks), ExitStack() as stack:
        df = cg.device_function
        assert len(df.tile_strategy.strategies) == 4
        for strategy in df.tile_strategy.strategies:
            for name in ("codegen_grid", "codegen_preamble"):
                stack.enter_context(
                    patch.object(strategy, name, side_effect=AssertionError(name))
                )
        df.tile_strategy.codegen_grid(SimpleNamespace(codegen=cg), blocks)
        assert type(df.pid) is FlatProgramIDs
        plan = df.cute_state.positional_root_plan
        assert plan is not None
        assert [p.block_id for p in df.pid.pid_info] == [a for a, _, _ in plan.axes]
        assert [(p.numel, p.block_size_var) for p in df.pid.pid_info] == [
            (str(extent), str(block)) for _, extent, block in plan.axes
        ]
        assert not cg.active_device_loops and cg.current_grid_state is None
        assert not cg.host_statements


@pytest.mark.parametrize("changed", ["multiple_roots", "wrong_root", "wrong_axes"])
def test_positional_grid_rejects_changed_root_ownership_before_binding(changed):
    kernel, args = _cases()[0]
    with _metadata(kernel, args) as (cg, blocks), ExitStack() as stack:
        if changed == "multiple_roots":
            root_ids = cg.host_function.device_ir.root_ids
            stack.enter_context(
                patch.object(cg.host_function.device_ir, "root_ids", [*root_ids, 999])
            )
        elif changed == "wrong_root":
            cg.current_root_graph_info = None
        else:
            blocks = []
        with pytest.raises(exc.BackendUnsupported, match="grid ownership changed"):
            setup_positional_grid(cg, blocks)
        assert cg.device_function.pid is None
        assert not cg.active_device_loops and cg.current_grid_state is None


def test_absent_positional_plan_keeps_ordinary_dispatch_and_axis_guard():
    kernel, args = _cases()[0]
    with _metadata(kernel, args) as (cg, blocks):
        cg.device_function.cute_state.positional_root_plan = None
        strategy = cg.device_function.tile_strategy.block_id_to_strategy[tuple(blocks)]
        with (
            patch.object(
                strategy,
                "codegen_grid",
                side_effect=RuntimeError("ordinary grid reached"),
            ),
            pytest.raises(RuntimeError, match="ordinary grid reached"),
        ):
            cg.device_function.tile_strategy.codegen_grid(
                SimpleNamespace(codegen=cg), blocks
            )
        with pytest.raises(exc.BackendUnsupported, match="thread axis 3"):
            CuteBackend()._check_thread_axis(3)
        assert cg.device_function.pid is None


@pytest.mark.parametrize("kernel,args", _cases())
def test_source_positional_grid_replaces_the_four_axis_generic_setup(kernel, args):
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [2]
        source = bound.to_code(config)
    assert "ptp_thread =" in source
    assert "thread_idx()[3]" not in source
    assert "block=(128, 1, 1)" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_nonsquare(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    out = torch.empty_like(bias)
    for batch in hl.tile(a.size(0)):
        out[batch, :, :] = torch.baddbmm(
            bias[batch, :, :], a[batch, :, :], b[batch, :, :]
        )
    return out


def test_source_ordinary_four_axis_grid_still_rejects():
    args = _cases()[0][1][:3]
    with _cpu_codegen():
        bound = _ordinary_nonsquare._bind_isolated(args)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [2]
        with pytest.raises(exc.BackendUnsupported, match="thread axis 3"):
            bound.to_code(config)
