#!/usr/bin/env python3
"""Fail when Helion's backend-neutral core names the CuTe backend.

Core is ``helion/_compiler/*.py``, ``helion/_compiler/autotuner_heuristics/**``,
``helion/autotuner/**``, ``helion/language/**`` and ``helion/runtime/*.py``; the
sibling backend packages ``helion/_compiler/{flydsl,metal,pallas,triton}/**`` are
scanned by the same rules so a backend never reaches into another backend's
folder.  A violation is a ``== "cute"`` / ``!= "cute"`` comparison or an
``in ("cute", ...)`` membership test, a literal ``BackendUnsupported("cute", ...)``
(also when split over lines) or an import of ``helion._compiler.cute`` or
``helion.runtime.cute`` (relative or absolute; imports under
``if TYPE_CHECKING:`` are counted separately as ``type_import``).

Backend-specific behaviour belongs behind a ``Backend`` hook with a no-op
default; the backend package implements it.  ``ALLOWLIST`` records the sites
that remain while the refactor moves them out, per file and kind.  The check
fails when a file exceeds its allowance and reports allowances that can be
tightened (``--strict`` fails on those too).  ``--update`` rewrites the
allowlist from the current tree.

Usage: python scripts/check_backend_isolation.py [--update] [--strict] [--verbose]
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections import defaultdict
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parent.parent
CORE_GLOBS = (
    "helion/_compiler/*.py",
    "helion/_compiler/autotuner_heuristics/**/*.py",
    "helion/autotuner/**/*.py",
    "helion/language/**/*.py",
    "helion/runtime/*.py",
)
SIBLING_BACKEND_GLOBS = (
    "helion/_compiler/flydsl/**/*.py",
    "helion/_compiler/metal/**/*.py",
    "helion/_compiler/pallas/**/*.py",
    "helion/_compiler/triton/**/*.py",
)
# The CuTe packages: the compiler half and the runtime half (launcher,
# structural policy, host wrappers).
CUTE_PACKAGES = ("helion._compiler.cute", "helion.runtime.cute")
KINDS = ("name_literal", "unsupported_literal", "import", "type_import")
NAME_LITERAL = re.compile(
    r'(?:[=!]=\s*"cute"|"cute"\s*[=!]=|\bin\s*[(\[{][^()\[\]{}]*"cute")'
)
UNSUPPORTED_LITERAL = re.compile(r'BackendUnsupported\(\s*"cute"')

# ALLOWLIST-BEGIN
ALLOWLIST: dict[str, dict[str, int]] = {
    "helion/_compiler/backend.py": {
        "type_import": 2,
    },
    "helion/_compiler/backend_registry.py": {
        "import": 1,
    },
    "helion/_compiler/device_function.py": {
        "type_import": 2,
    },
    "helion/_compiler/device_ir.py": {
        "type_import": 1,
    },
    "helion/_compiler/generate_ast.py": {
        "type_import": 1,
    },
    "helion/_compiler/lane_loop_context.py": {
        "type_import": 1,
    },
    "helion/_compiler/metal/reduction.py": {
        "import": 2,
    },
    "helion/_compiler/metal/row_reduction_heuristic.py": {
        "import": 1,
    },
    "helion/_compiler/tile_dispatch.py": {
        "type_import": 1,
    },
    "helion/_compiler/tile_strategy.py": {
        "import": 12,
        "type_import": 4,
    },
    "helion/autotuner/aot_cache.py": {
        "import": 3,
    },
    "helion/autotuner/aot_kernel.py": {
        "type_import": 1,
    },
    "helion/autotuner/aot_structural_policy.py": {
        "import": 3,
    },
    "helion/autotuner/config_spec.py": {
        "import": 3,
        "type_import": 9,
    },
    "helion/autotuner/heuristic_generator.py": {
        "import": 1,
    },
    "helion/autotuner/local_cache.py": {
        "import": 1,
    },
    "helion/autotuner/surrogate_pattern_search.py": {
        "import": 14,
        "type_import": 2,
    },
    "helion/runtime/__init__.py": {
        "import": 3,
    },
    "helion/runtime/cute_structural_config.py": {
        "import": 1,
    },
    "helion/runtime/kernel.py": {
        "import": 3,
        "type_import": 2,
    },
    "helion/runtime/settings.py": {
        "import": 4,
        "type_import": 1,
    },
}
# ALLOWLIST-END


def _is_type_checking_test(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _resolve(module: str | None, level: int, package: str) -> str:
    if level == 0:
        return module or ""
    parts = package.split(".")
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, module] if module else base)


def _cute_package_of(module: str) -> str | None:
    """The CuTe package ``module`` is or lives under, if any."""
    for cute_package in CUTE_PACKAGES:
        if module == cute_package or module.startswith(cute_package + "."):
            return cute_package
    return None


def _imports(tree: ast.AST, package: str) -> list[tuple[int, str, str]]:
    """``(line, kind, module)`` for every import of a CuTe package."""
    found: list[tuple[int, str, str]] = []

    def visit(node: ast.AST, type_only: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking_test(child.test):
                for stmt in child.body:
                    visit_stmt(stmt, True)
                for stmt in child.orelse:
                    visit_stmt(stmt, type_only)
                continue
            visit_stmt(child, type_only)

    def visit_stmt(node: ast.AST, type_only: bool) -> None:
        kind = "type_import" if type_only else "import"
        if isinstance(node, ast.ImportFrom):
            module = _resolve(node.module, node.level, package)
            if _cute_package_of(module) is not None:
                found.append((node.lineno, kind, module))
            else:
                # ``from helion._compiler import cute`` / ``from . import cute``
                found.extend(
                    (node.lineno, kind, f"{module}.cute")
                    for alias in node.names
                    if alias.name == "cute" and f"{module}.cute" in CUTE_PACKAGES
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _cute_package_of(alias.name) is not None:
                    found.append((node.lineno, kind, alias.name))
        visit(node, type_only)

    visit(tree, False)
    return found


def scan(root: Path) -> dict[str, list[tuple[int, str, str]]]:
    """``{relative path: [(line, kind, detail), ...]}`` for every scanned file."""
    sites: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    globs = CORE_GLOBS + SIBLING_BACKEND_GLOBS
    files = sorted({p for g in globs for p in root.glob(g)})
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text()
        for pattern, kind in (
            (NAME_LITERAL, "name_literal"),
            (UNSUPPORTED_LITERAL, "unsupported_literal"),
        ):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                sites[rel].append((line, kind, match.group(0)))
        package = ".".join(path.relative_to(root).with_suffix("").parts[:-1])
        sites[rel].extend(_imports(ast.parse(text, filename=str(path)), package))
        sites[rel].sort()
    return {rel: found for rel, found in sites.items() if found}


def counts_of(
    sites: dict[str, list[tuple[int, str, str]]],
) -> dict[str, dict[str, int]]:
    return {
        rel: dict(sorted(Counter(kind for _, kind, _ in found).items()))
        for rel, found in sorted(sites.items())
    }


def rewrite_allowlist(counts: dict[str, dict[str, int]]) -> None:
    path = Path(__file__).resolve()
    text = path.read_text()
    begin = text.index("# ALLOWLIST-BEGIN\n") + len("# ALLOWLIST-BEGIN\n")
    end = text.index("# ALLOWLIST-END")
    lines = ["ALLOWLIST: dict[str, dict[str, int]] = {"]
    for rel, kinds in counts.items():
        lines.append(f'    "{rel}": {{')
        lines.extend(f'        "{kind}": {count},' for kind, count in kinds.items())
        lines.append("    },")
    lines.append("}")
    path.write_text(text[:begin] + "\n".join(lines) + "\n" + text[end:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true", help="rewrite ALLOWLIST")
    parser.add_argument(
        "--strict", action="store_true", help="also fail on stale allowances"
    )
    parser.add_argument("--verbose", action="store_true", help="list every site")
    args = parser.parse_args()

    sites = scan(REPO)
    counts = counts_of(sites)
    if args.update:
        rewrite_allowlist(counts)
        print(f"allowlist rewritten: {len(counts)} files")
        return 0

    failed = False
    stale: list[str] = []
    for rel in sorted(set(counts) | set(ALLOWLIST)):
        actual = counts.get(rel, {})
        allowed = ALLOWLIST.get(rel, {})
        for kind in KINDS:
            have, limit = actual.get(kind, 0), allowed.get(kind, 0)
            if have > limit:
                failed = True
                print(f"{rel}: {have} {kind} site(s), allowlist permits {limit}")
                for line, found_kind, detail in sites.get(rel, []):
                    if found_kind == kind:
                        print(f"    {rel}:{line}: {detail}")
            elif have < limit:
                stale.append(f"{rel}: {kind} {limit} -> {have}")
    if args.verbose:
        for rel, found in sorted(sites.items()):
            for line, kind, detail in found:
                print(f"{rel}:{line}: {kind}: {detail}")
    total = sum(sum(kinds.values()) for kinds in counts.values())
    print(f"{total} CuTe reference(s) in {len(counts)} scanned file(s) (allowlisted)")
    if stale:
        print("allowances that can be tightened (run with --update):")
        for item in stale:
            print(f"    {item}")
        if args.strict:
            failed = True
    if failed:
        print(
            "core and sibling backends must not name the CuTe backend: add a "
            "Backend hook with a no-op default instead (see the module docstring)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
