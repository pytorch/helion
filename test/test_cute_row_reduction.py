from __future__ import annotations

import ast
from contextlib import contextmanager
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from .test_cute_register_program import _VectorizeConditions
import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.row_fragment import RowFragment
from helion._compiler.cute.row_fragment import RowFragmentEmitter
from helion._compiler.cute.row_fragment import RowFragmentLayout
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.generate_ast import GenerateAST


_ROUNDING_CASES = [
    (torch.float32, torch.float16, 1.0001),
    (torch.float32, torch.bfloat16, 1.0001),
    (torch.float16, torch.bfloat16, 1.0009765625),
]


class _RowCodegen:
    def __init__(self) -> None:
        counter = itertools.count()
        self.device_function = SimpleNamespace(
            new_var=lambda prefix: f"{prefix}_{next(counter)}"
        )
        self.statements: list[ast.stmt] = []

    def add_statement(self, statement: ast.stmt) -> None:
        self.statements.append(statement)

    @contextmanager
    def set_statements(self, statements: list[ast.stmt]) -> Iterator[None]:
        previous = self.statements
        self.statements = statements
        try:
            yield
        finally:
            self.statements = previous


def _run_row_sum(
    values: torch.Tensor, output_dtype: torch.dtype, lanes: int
) -> torch.Tensor:
    """Execute the emitted casts, additions and subgroup exchanges on CPU."""

    def reduce(x: torch.Tensor) -> torch.Tensor:
        return x.sum(-1, keepdim=True, dtype=output_dtype)

    module = make_fx(reduce)(values)
    source = next(iter(module.graph.find_nodes(op="placeholder")))
    reduction = next(
        iter(
            module.graph.find_nodes(
                op="call_function", target=torch.ops.aten.sum.dim_IntList
            )
        )
    )
    cg = _RowCodegen()
    fragment = RowFragment(
        "selected", values.dtype, values.numel(), RowFragmentLayout(lanes, 1, "lane")
    )
    emitter = RowFragmentEmitter(
        cast("GenerateAST", cg), lambda node: fragment, layout=fragment.layout
    )
    emitter.bind(source, fragment)
    with patch.object(
        CompileEnvironment,
        "current",
        return_value=SimpleNamespace(backend=CuteBackend()),
    ):
        result = emitter.emit(reduction)

    def bfloat16(value):
        return torch.as_tensor(np.asarray(value).copy()).bfloat16().float().numpy()

    def uniform(value):
        assert np.all(value == np.asarray(value).flat[0])
        return bool(np.all(value))

    def shuffle(value, *, offset):
        assert 0 <= offset < lanes
        return value[np.arange(lanes) ^ offset]

    namespace = {
        "np": np,
        "warp_predicate": uniform,
        "lane": np.arange(lanes, dtype=np.int32),
        "selected": values.float().numpy().reshape(-1, lanes),
        "cutlass": SimpleNamespace(
            Float16=np.float16,
            BFloat16=bfloat16,
            Float32=np.float32,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda size, dtype: np.zeros(
                (size, lanes), dtype=np.float32 if dtype is bfloat16 else dtype
            ),
            arch=SimpleNamespace(shuffle_sync_bfly=shuffle),
        ),
    }
    tree = _VectorizeConditions().visit(ast.Module(body=cg.statements, type_ignores=[]))
    exec(compile(ast.fix_missing_locations(tree), "<row-sum-cpu>", "exec"), namespace)
    actual = torch.as_tensor(namespace[result.name]).to(output_dtype)
    assert torch.equal(actual, actual[:, :1].expand_as(actual))
    return actual[:, :1]


@pytest.mark.parametrize("source_dtype,output_dtype,first", _ROUNDING_CASES)
@pytest.mark.parametrize("lanes", [1, 2])
def test_row_sum_converts_inputs_before_accumulating(
    source_dtype: torch.dtype, output_dtype: torch.dtype, first: float, lanes: int
) -> None:
    selected = torch.tensor([[first, -1.0]], dtype=source_dtype)
    expected = selected.sum(-1, keepdim=True, dtype=output_dtype)
    assert not torch.equal(
        expected, selected.float().sum(-1, keepdim=True).to(output_dtype)
    )
    torch.testing.assert_close(
        _run_row_sum(selected, output_dtype, lanes), expected, rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _topk_explicit_sum(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty((x.size(0), 1), dtype=dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        values, indices = torch.topk(x[row, :], 2, dim=-1)
        out[row, :] = values.sum(-1, keepdim=True, dtype=dtype)
    return out


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source_dtype,output_dtype,first", _ROUNDING_CASES)
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_explicit_sum_native(
    source_dtype: torch.dtype, output_dtype: torch.dtype, first: float, layout: str
) -> None:
    x = torch.tensor(
        [[first, -1.0, -9.0, -9.0]], dtype=source_dtype, device=DEVICE
    ).repeat(5, 1)
    args = (x, output_dtype)
    bound = _topk_explicit_sum._bind_isolated(args)
    config = bound.config_spec.default_config()
    config.config.update(
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=16,
        cute_topk_selection_layout=layout,
    )
    source = bound.to_code(config)
    assert "row_reduce" in source
    actual = bound.compile_config(config)(*args)
    expected = x.topk(2).values.sum(-1, keepdim=True, dtype=output_dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
