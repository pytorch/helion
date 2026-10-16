"""Scoped Inductor setup reuse preserves scalar codegen and context isolation."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch._inductor import config as inductor_config
from torch._inductor.virtualized import V

from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_register_region import _conditional_permutation
from test.test_cute_register_region import _plan

import helion
from helion._compiler.cute.row_fragment import RowFragmentEmitter
from helion._compiler.cute.scalar_lowering import ScalarLoweringContextCache
from helion._compiler.inductor_lowering import GenerateASTFromInductor
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_float(x: torch.Tensor, narrow_dtype: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :].reshape(32, 8)
        group = hl.arange(32)[:, None]
        peer = torch.gather(value, 0, (group ^ 1).expand(32, 8))
        small = (peer / 11).to(narrow_dtype)
        restored = small.to(torch.float32)
        out[row, :, :] = torch.where(
            value > 0,
            torch.sin(restored) * restored + torch.exp(restored / 256),
            torch.floor(restored / 3.5) + torch.remainder(restored, 2.5),
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_integer(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :].reshape(32, 8)
        group = hl.arange(32)[:, None]
        peer = torch.gather(value, 0, (group ^ 1).expand(32, 8))
        quotient = torch.div(peer, 3, rounding_mode="floor")
        second_quotient = torch.div(peer, 5, rounding_mode="floor")
        out[row, :, :] = torch.where(
            value > 0,
            quotient + torch.remainder(peer, 7),
            second_quotient + (peer >> 2),
        )
    return out


def _code(kernel, args):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(kernel, args)
        assert plan is not None
        source = bound.to_code(config)
    assert "register_output" in source
    return source


def _assert_source_unchanged(kernel, args):
    original_init = RowFragmentEmitter.__init__

    def init_without_reuse(self, *args, **kwargs):
        kwargs["reuse_scalar_lowering"] = False
        original_init(self, *args, **kwargs)

    with patch.object(RowFragmentEmitter, "__init__", init_without_reuse):
        baseline = _code(kernel, args)
    assert _code(kernel, args) == baseline


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scalar_context_preserves_float_rounding_source(dtype):
    _assert_source_unchanged(_mixed_float, (torch.zeros(3, 32, 8), dtype))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_scalar_context_preserves_integer_rounding_source(dtype):
    _assert_source_unchanged(_mixed_integer, (torch.zeros(3, 32, 8, dtype=dtype),))


def test_scalar_context_preserves_branch_source():
    _assert_source_unchanged(
        _conditional_permutation, (torch.zeros(3, 32, 4, dtype=torch.int32),)
    )


def test_scalar_context_isolates_programs_branches_and_arguments():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, _, _ = _plan(_mixed_integer, (torch.zeros(3, 32, 8, dtype=torch.int32),))
    cg = cast("GenerateAST", SimpleNamespace())
    cache = ScalarLoweringContextCache(cg)
    other_program = ScalarLoweringContextCache(cg)
    graph = torch.fx.Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")
    branch = torch.fx.Graph().placeholder("branch")
    a = ast.Name(id="a", ctx=ast.Load())
    b = ast.Name(id="b", ctx=ast.Load())

    with (
        bound.env,
        V.set_graph_handler(V.graph),
        V.set_kernel_handler(V.kernel),
        V.set_ops_handler(V.ops),
    ):
        initial = V.graph, V.kernel, V.ops
        initial_split_reductions = inductor_config.split_reductions
        with cache.install(first, {"arg": a}):
            first_handlers = V.graph, V.kernel, V.ops
            assert isinstance(V.ops, GenerateASTFromInductor)
            assert V.ops.input_name_lookup == {"arg": a}
            assert inductor_config.split_reductions is False
        assert (V.graph, V.kernel, V.ops) == initial
        assert inductor_config.split_reductions == initial_split_reductions

        with cache.install(second, {"arg": b}):
            assert V.graph is first_handlers[0]
            assert V.kernel is first_handlers[1]
            assert V.ops is not first_handlers[2]
            assert isinstance(V.ops, GenerateASTFromInductor)
            assert V.ops.input_name_lookup == {"arg": b}
            with cache.install(branch, {"arg": a}):
                assert V.graph is not first_handlers[0]
                assert V.kernel is not first_handlers[1]
            assert V.graph is first_handlers[0]
            assert V.kernel is first_handlers[1]
            assert V.ops.input_name_lookup == {"arg": b}

        with other_program.install(first, {"arg": a}):
            assert V.graph is not first_handlers[0]
            assert V.kernel is not first_handlers[1]

        with (
            pytest.raises(RuntimeError, match="context exit"),
            cache.install(first, {"arg": b}),
        ):
            raise RuntimeError("context exit")
        assert (V.graph, V.kernel, V.ops) == initial
        assert inductor_config.split_reductions == initial_split_reductions
