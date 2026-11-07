"""Source-visible functions for fixtures that transform a shared Helion kernel."""

from __future__ import annotations

import ast
import linecache
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable


def function_from_ast(
    function: ast.FunctionDef, namespace: dict[str, Any], filename: str
) -> Callable[..., Any]:
    """Compile a fixture while retaining source for normal Helion tracing."""
    source = ast.unparse(function) + "\n"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(namespace)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[function.name]
