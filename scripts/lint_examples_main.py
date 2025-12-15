from __future__ import annotations

import ast
from pathlib import Path
import sys


def has_main_function(filename: str) -> bool:
    text = Path(filename).read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as e:
        print(f"{filename} has syntax error: {e}")
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return True
    return False


def main() -> int:
    failed = False
    for filename in sys.argv[1:]:
        if not filename.startswith("examples/") or not filename.endswith(".py"):
            continue
        if Path(filename).name in ["__init__.py", "utils.py"]:
            continue
        if not has_main_function(filename):
            print(f"{filename} is missing a main() function.")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
