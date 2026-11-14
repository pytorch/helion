"""Static import closure for compiled helper cache declarations."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import helion


def helion_imports(relative: str, *, prefix: str = "") -> set[str]:
    """Package-relative paths of the Helion modules ``relative`` imports.

    Follow relative and absolute Helion imports (excluding type-only imports).
    A prefix limits traversal to that subtree, but dependencies outside it
    remain in the result as boundary modules whose source must be declared.
    """

    root = Path(cast("str", helion.__file__)).resolve().parent
    pending = [relative]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if not current.startswith(prefix):
            continue
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
            if isinstance(statement, ast.Import):
                candidates = [
                    Path(*alias.name.split(".")[1:]).with_suffix(".py")
                    for alias in statement.names
                    if alias.name.startswith("helion.")
                ]
            elif isinstance(statement, ast.ImportFrom):
                if statement.level:
                    base = package
                    for _ in range(statement.level - 1):
                        base = base.parent
                    module = base.joinpath(*(statement.module or "").split("."))
                elif statement.module == "helion" or (
                    statement.module and statement.module.startswith("helion.")
                ):
                    module = Path(*statement.module.split(".")[1:])
                else:
                    continue
                candidates = [module.with_suffix(".py")] if module.name else []
                candidates.extend(
                    module / f"{alias.name}.py" for alias in statement.names
                )
            else:
                continue
            for candidate in candidates:
                if (root / candidate).is_file():
                    pending.append(candidate.as_posix())
    return seen
