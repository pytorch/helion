"""The source of the paired attention templates as syntax trees.

The CPU tests of the paired body's options read a rule off the code that
states it: the definition of a template function, statement by statement,
and the calls in it. These find both.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from helion._compiler.cute import _paged_attention_paired_pipeline as paired_template

if TYPE_CHECKING:
    import types


def _functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def source_function(
    name: str, module: types.ModuleType = paired_template
) -> ast.FunctionDef:
    """The definition of function ``name`` in the source of ``module``, the
    paired body's template unless another is given: the one function of
    that name at any depth."""
    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    (found,) = [function for function in _functions(tree) if function.name == name]
    return found


def tree_function(tree: ast.AST, prefix: str) -> ast.FunctionDef:
    """The one function at any depth of the syntax tree ``tree`` whose name
    starts with ``prefix``."""
    (found,) = [
        function for function in _functions(tree) if function.name.startswith(prefix)
    ]
    return found


def source_calls(node: ast.AST, name: str) -> list[ast.Call]:
    """The calls of ``name`` in the syntax tree ``node``: of the callee as
    it is written (``cute.arch.atomic_add``) or of its last part, so a
    kernel's ``_helion_paged_paired.paged_attention_paired_body(...)`` is a
    call of ``paged_attention_paired_body``."""
    calls = []
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            callee = ast.unparse(call.func)
            if name in (callee, callee.rpartition(".")[2]):
                calls.append(call)
    return calls
