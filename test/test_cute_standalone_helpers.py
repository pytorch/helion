from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
import sys
import textwrap
import types
from typing import TYPE_CHECKING

import pytest

from helion._compiler.cute.standalone import _Helpers

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


def _load_helper_modules(
    sources: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, source in sources.items():
        path = tmp_path / f"{name.rsplit('.', 1)[-1]}.py"
        path.write_text(textwrap.dedent(source))
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)


@contextmanager
def _standalone_helpers(
    helpers: _Helpers, sources: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Any]]:
    tree = ast.fix_missing_locations(ast.Module(body=helpers.emit(), type_ignores=[]))
    code = compile(tree, "<standalone-helpers>", "exec")
    namespace: dict[str, Any] = {"types": types}
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "helion", None)
        for name in sources:
            isolated.setitem(sys.modules, name, None)
        exec(code, namespace)
        yield namespace


def test_late_dependency_keeps_every_required_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_shared": """
            def first():
                return 3

            def late():
                return 7
        """,
        "helion._standalone_test_consumer": """
            from helion._standalone_test_shared import late

            def consume():
                return late() * 2
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    shared = helpers.require("helion._standalone_test_shared", {"first"})
    consumer = helpers.require("helion._standalone_test_consumer", {"consume"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[shared].first() == 3
        assert exported[consumer].consume() == 14


def test_shared_dependency_initializes_before_its_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_dependency": """
            OFFSET = 5
        """,
        "helion._standalone_test_left": """
            from helion._standalone_test_dependency import OFFSET

            INITIAL_VALUE = OFFSET + 1

            def value():
                return INITIAL_VALUE
        """,
        "helion._standalone_test_right": """
            from helion._standalone_test_dependency import OFFSET

            INITIAL_VALUE = OFFSET + 2

            def value():
                return INITIAL_VALUE
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    dependency = helpers.require("helion._standalone_test_dependency", {"OFFSET"})
    left = helpers.require("helion._standalone_test_left", {"value"})
    right = helpers.require("helion._standalone_test_right", {"value"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[dependency].OFFSET == 5
        assert exported[left].value() == 6
        assert exported[right].value() == 7


@pytest.mark.parametrize(
    "import_statement",
    (
        "from helion._standalone_test_dependency import increment",
        "from ._standalone_test_dependency import increment",
    ),
)
def test_helper_local_import_has_no_helion_dependency(
    import_statement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_dependency": """
            def increment(value):
                return value + 1
        """,
        "helion._standalone_test_consumer": f"""
            def value():
                {import_statement}
                return increment(8)
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    consumer = helpers.require("helion._standalone_test_consumer", {"value"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[consumer].value() == 9


@pytest.mark.parametrize(
    ("module_name", "helper_name"),
    (
        ("helion.runtime.cute.paired_sum", "try_paired_sum_cast"),
        ("helion.runtime.cute.single_sum", "try_single_sum_cast"),
    ),
)
def test_runtime_dependent_launch_helper_is_rejected(
    module_name: str,
    helper_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "helion._standalone_test_consumer": f"""
            def value():
                from {module_name} import {helper_name}
                return {helper_name}
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    helpers.require("helion._standalone_test_consumer", {"value"})

    with pytest.raises(NotImplementedError, match="runtime-dependent launch plan"):
        helpers.emit()


def test_unaliased_dotted_helion_import_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_dependency": "VALUE = 9\n",
        "helion._standalone_test_consumer": """
            def value():
                import helion._standalone_test_dependency
                return helion._standalone_test_dependency.VALUE
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    helpers.require("helion._standalone_test_consumer", {"value"})

    with pytest.raises(NotImplementedError, match="helper imports require an alias"):
        helpers.emit()
