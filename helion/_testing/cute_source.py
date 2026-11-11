"""Source-visible functions for fixtures that transform a shared Helion kernel."""

from __future__ import annotations

import ast
import inspect
import linecache
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

import helion

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import Kernel


_ReturnT = TypeVar("_ReturnT")


def function_from_ast(
    function: ast.FunctionDef,
    namespace: dict[str, Any],
    filename: str,
    *,
    copy_namespace: bool = True,
) -> Callable[..., Any]:
    """Compile a fixture while retaining source for normal Helion tracing."""
    source = ast.unparse(function) + "\n"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    if copy_namespace:
        namespace = dict(namespace)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[function.name]


def transform_kernel(
    kernel: Kernel[_ReturnT],
    transform: ast.NodeTransformer,
    *,
    name: str,
    filename: str,
    namespace: dict[str, Any] | None = None,
) -> Kernel[_ReturnT]:
    """Rebuild a source-visible CuTe fixture from a shared kernel's AST.

    The caller retains its transform-specific rewrite-count assertions.
    An explicit namespace is used directly, so a real module's globals can
    retain importable helper origins. The default copies the source globals.
    """
    function = ast.parse(inspect.getsource(kernel.fn)).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.decorator_list = []
    function.name = name
    transformed = transform.visit(function)
    assert isinstance(transformed, ast.FunctionDef)
    return helion.kernel(
        function_from_ast(
            transformed,
            kernel.fn.__globals__ if namespace is None else namespace,
            filename,
            copy_namespace=namespace is None,
        ),
        backend="cute",
        static_shapes=True,
    )
