from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
import inspect
import sys
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.jagged_dense_bmm import jagged_dense_bmm
import pytest
import torch

from test.test_cute_flat_grouped_codegen import _candidate
from test.test_cute_flat_grouped_codegen import _ordinary
from test.test_cute_flat_grouped_ir import _host
from test.test_cute_flat_grouped_ir import cuda_binding

from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.cute.tcgen05_flat_grouped_host import prove_flat_grouped_host
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.type_info import LiteralType

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel


_PREFIX = """
from __future__ import annotations
import torch
import helion.language as hl

def unknown_call(output, source):
    output.set_(source)

def unknown_device(source):
    source.as_strided_(source.shape, source.stride())
    return source.device
"""


@contextmanager
def rewritten_binding(
    tmp_path: Path, change: str, *, static: bool = True
) -> Iterator[BoundKernel]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(jagged_dense_bmm.fn)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    function.decorator_list = []
    allocation = next(
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "output"
            for target in statement.targets
        )
    )
    changes = {
        "output_set": "output.set_(jagged)",
        "output_set_value": "unused = output.set_(jagged)",
        "alias_then_set": "alias = output\nalias.set_(jagged)",
        "tuple_rebind": "(output,) = (jagged,)",
        "multiple_rebind": "unused = output = jagged",
        "branch_set": "if True:\n    output.set_(jagged)",
        "branch_rebind": "if True:\n    output = jagged",
        "metadata_mutation": "jagged.as_strided_(jagged.shape, jagged.stride())",
        "rhs_metadata_mutation": "dense.transpose_(1, 2)",
        "offset_mutation": "seq_offsets.add_(0)",
        "unknown_call": "unknown_call(output, jagged)",
        "subscript_store": "output[0] = 0",
        "while_loop": "while False:\n    output.set_(jagged)",
        "readonly_metadata": "unused = jagged.numel() + dense.size(1)",
        "literal_dead_effect": "if False:\n    output.set_(jagged)",
        "literal_readonly_branch": "if False:\n    unused = L * D\nelse:\n    unused = L * K",
    }
    if change in changes:
        function.body[allocation + 1 : allocation + 1] = ast.parse(changes[change]).body
    elif change == "post_grid_mutation":
        function.body[-1:-1] = ast.parse("output.set_(jagged)").body
    elif change == "allocation_argument_call":
        replacement = (
            "output = torch.empty((L, K), dtype=dtype, "
            "device=unknown_device(jagged)).view(-1)"
        )
        function.body[allocation] = ast.parse(replacement).body[0]
    elif change != "control":
        raise AssertionError(change)
    filename = tmp_path / f"host_{change}.py"
    filename.write_text(
        _PREFIX + "\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    )
    name = f"flat_grouped_host_{change}"
    spec = importlib.util.spec_from_file_location(name, filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    with (
        torch.no_grad(),
        patch.object(jagged_dense_bmm, "fn", module.jagged_dense_bmm),
        cuda_binding(
            static=static, original=True, groups=3, rows=273, k=128, n=128
        ) as bound,
    ):
        yield bound


@pytest.mark.parametrize(
    "change",
    [
        "output_set",
        "output_set_value",
        "alias_then_set",
        "tuple_rebind",
        "multiple_rebind",
        "branch_set",
        "branch_rebind",
        "metadata_mutation",
        "rhs_metadata_mutation",
        "offset_mutation",
        "unknown_call",
        "subscript_store",
        "while_loop",
        "post_grid_mutation",
        "allocation_argument_call",
        "literal_dead_effect",
    ],
)
def test_actual_typed_host_effects_retain_original_lowering(
    tmp_path: Path, change: str
) -> None:
    with rewritten_binding(tmp_path, change) as bound:
        host = _host(bound)
        assert prove_flat_grouped_host(host, "jagged", "output") is None
        assert prove_flat_grouped_rna(bound.env, host.device_ir) is None
        if change in {"output_set", "tuple_rebind", "branch_set"}:
            # Declining the native choice keeps this exact binding's original
            # host effects and ordinary device program, not another fallback.
            assert _candidate(bound) == _ordinary(bound)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize(
    "change", ["control", "readonly_metadata", "literal_readonly_branch"]
)
def test_typed_metadata_and_literal_metadata_branches_remain_admissible(
    tmp_path: Path, static: bool, change: str
) -> None:
    with rewritten_binding(tmp_path, change, static=static) as bound:
        host = _host(bound)
        assert prove_flat_grouped_host(host, "jagged", "output") == "jagged"
        proof = prove_flat_grouped_rna(bound.env, host.device_ir)
        assert proof is not None
        assert proof.a.untyped_storage()._cdata != proof.output.untyped_storage()._cdata


@pytest.mark.parametrize(
    "statement",
    [
        "output.requires_grad = False",
        "(unused := output.set_(jagged))",
        "if unknown_predicate:\n    output.set_(jagged)",
        "try:\n    output.set_(jagged)\nexcept RuntimeError:\n    pass",
    ],
)
def test_unknown_or_frontend_unsupported_host_writes_decline(
    statement: str,
) -> None:
    # Some host assignment forms are rejected before IR creation today. Keep a
    # direct-AST negative so this proof stays conservative if frontend support
    # is added later, without claiming these are currently executable kernels.
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        host = _host(bound)
        before = list(host.body)
        try:
            host.body[-2:-2] = ast.parse(statement).body
            assert prove_flat_grouped_host(host, "jagged", "output") is None
            assert prove_flat_grouped_rna(bound.env, host.device_ir) is None
        finally:
            host.body[:] = before


def test_allocation_callable_identity_does_not_prove_its_receiver_is_pure() -> None:
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        host = _host(bound)
        assignment = next(
            statement
            for statement in host.body
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "output"
        )
        view = assignment.value
        assert isinstance(view, ast.Call) and isinstance(view.func, ast.Attribute)
        allocation = view.func.value
        assert isinstance(allocation, ast.Call) and isinstance(
            allocation.func, ast.Attribute
        )
        receiver = allocation.func.value
        assert isinstance(receiver, ExtendedAST)
        before = receiver._type_info
        assert before is not None
        try:
            # The factory function still has the inferred identity torch.empty.
            # An unknown object's property lookup need not be effect-free.
            receiver._type_info = LiteralType(
                before.origin, SimpleNamespace(empty=torch.empty)
            )
            assert prove_flat_grouped_host(host, "jagged", "output") is None
        finally:
            receiver._type_info = before
