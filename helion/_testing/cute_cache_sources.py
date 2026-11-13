"""Static import closure for compiled helper cache declarations."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import helion


def helion_imports(relative: str) -> set[str]:
    """Package-relative paths of the Helion modules ``relative`` imports.

    Follows the relative ``from . import`` statements transitively (type-only
    imports excluded), so a new Helion import in the schedule shows up here.
    """

    root = Path(cast("str", helion.__file__)).resolve().parent
    pending = [relative]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        package = Path(current).parent
        statements = list(ast.parse((root / current).read_text()).body)
        while statements:
            statement = statements.pop()
            if isinstance(statement, ast.If):
                test = statement.test
                if not (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"):
                    statements.extend(statement.body)
                statements.extend(statement.orelse)
                continue
            if not isinstance(statement, ast.ImportFrom) or not statement.level:
                continue
            base = package
            for _ in range(statement.level - 1):
                base = base.parent
            module = base.joinpath(*(statement.module or "").split("."))
            candidates = (
                [module.with_suffix(".py")]
                if statement.module
                else [module / f"{alias.name}.py" for alias in statement.names]
            )
            for candidate in candidates:
                if (root / candidate).is_file():
                    pending.append(candidate.as_posix())
    return seen
