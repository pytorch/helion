from __future__ import annotations

import ast
import pathlib
import sys
from typing import Iterable

ALLOWED_NAME = "DEVICE"


class DeviceKwargVisitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.errors: list[tuple[int, int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords or ():
            if kw.arg != "device":
                continue

            # Only disallow string literals, e.g., device="cuda" or device='cpu'
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                self.errors.append(
                    (
                        getattr(kw, "lineno", node.lineno),
                        getattr(kw, "col_offset", node.col_offset),
                        "device must not be a string literal like 'cuda'; use DEVICE",
                    )
                )

        # Continue walking children
        self.generic_visit(node)


def iter_python_files(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            yield from path.rglob("*.py")
        elif path.suffix == ".py" and path.exists():
            yield path


def should_check(path: pathlib.Path) -> bool:
    # Only check files under test/ or examples/
    try:
        parts = path.resolve().parts
    except FileNotFoundError:
        parts = path.parts
    # find directory names in the path
    return "test" in parts or "examples" in parts


def check_file(path: pathlib.Path) -> list[tuple[int, int, str]]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # let other hooks catch syntax errors
        return []

    visitor = DeviceKwargVisitor(str(path))
    visitor.visit(tree)

    # Allow inline opt-out using marker on the same line: @ignore-device-lint
    lines = source.splitlines()
    filtered_errors: list[tuple[int, int, str]] = []
    for lineno, col, msg in visitor.errors:
        if 1 <= lineno <= len(lines):
            if "@ignore-device-lint" in lines[lineno - 1]:
                continue
        filtered_errors.append((lineno, col, msg))
    return filtered_errors


def main(argv: list[str]) -> int:
    if len(argv) == 0:
        # pre-commit will pass file list; if not, scan the default dirs
        candidates = list(iter_python_files(["examples", "test"]))
    else:
        candidates = [pathlib.Path(a) for a in argv]

    errors_found = 0
    for path in candidates:
        if not should_check(path):
            continue
        for lineno, col, msg in check_file(path):
            print(f"{path}:{lineno}:{col}: {msg}")
            errors_found += 1

    return 1 if errors_found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
