from __future__ import annotations

import ast

import pytest

from helion._compiler.output_code_utils import close_artifact_source
from helion._compiler.output_code_utils import dependency_free_runtime_source
from helion._compiler.triton.backend import TritonBackend

pytestmark = pytest.mark.cpu_only


def test_unused_source_import_is_removed() -> None:
    source = """
import original_kernel as _source_module
import torch

def kernel(x):
    return torch.empty_like(x)
"""
    closed = close_artifact_source(source)
    assert "original_kernel" not in closed
    assert "import torch" in closed
    assert close_artifact_source(closed) == closed


def test_shadowed_local_does_not_keep_source_import() -> None:
    source = """
import original_kernel as value

def kernel(value):
    def inner():
        return value
    return inner()
"""
    closed = close_artifact_source(source)
    assert "original_kernel" not in closed
    namespace = {}
    exec(closed, namespace)
    assert namespace["kernel"](7) == 7


@pytest.mark.parametrize(
    "body",
    [
        "def kernel(x):\n    return x * source.factor",
        "def kernel(x):\n    return source.helper(x)",
        "def kernel(x):\n    def inner():\n        return source.factor\n    return inner()",
        "@source.decorate\ndef kernel(x):\n    return x",
        "def kernel(x=source.default):\n    return x",
        "def kernel(*, x=source.default):\n    return x",
        "class Wrapper(source.Base):\n    pass",
        "class Wrapper:\n    value = source.factor",
        "def kernel(x):\n    return [v for v in source.values]",
        "def kernel(x: source.Tensor):\n    return x",
        'def kernel(x: "source.Tensor"):\n    return x',
        "def kernel(x) -> source.Tensor:\n    return x",
        "value: source.Tensor = None",
    ],
)
def test_live_external_binding_is_rejected(body: str) -> None:
    with pytest.raises(ValueError, match="external module"):
        close_artifact_source(
            "from __future__ import annotations\n"
            "import original_kernel as source\n" + body
        )


def test_live_from_import_is_rejected() -> None:
    with pytest.raises(ValueError, match="external module"):
        close_artifact_source(
            "from original_kernel import factor as scale\n"
            "def kernel(x):\n    return x * scale\n"
        )


def test_only_unused_aliases_are_dropped() -> None:
    closed = close_artifact_source(
        "import original_kernel as source, torch\n"
        "def kernel(x):\n    return torch.empty_like(x)\n"
    )
    assert "original_kernel" not in closed
    assert "import torch" in closed


@pytest.mark.parametrize(
    "source",
    [
        "from original_kernel import *",
        "from . import original_kernel",
        "def kernel(x):\n    import original_kernel\n    return x",
        "def unused_helper():\n    import original_kernel\n    return 1",
    ],
)
def test_unrelocatable_imports_are_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="imports|external module"):
        close_artifact_source(source)


@pytest.mark.parametrize(
    "source",
    [
        'def kernel():\n    return __import__("original_kernel")',
        'def kernel():\n    loader = __import__\n    return loader("original_kernel")',
        "from torch import __builtins__ as namespace\n",
        'import importlib as loader\ndef kernel():\n    return loader.import_module("original_kernel")',
        'from importlib import import_module as loader\ndef kernel():\n    return loader("original_kernel")',
        'def kernel():\n    return globals()["original_kernel"]',
        'def kernel():\n    return locals()["original_kernel"]',
        'def kernel():\n    return vars()["original_kernel"]',
        'def kernel():\n    return eval("original_kernel.factor")',
    ],
)
def test_dynamic_namespace_access_is_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="dynamic Python namespace"):
        close_artifact_source(source)


def test_unbound_global_is_rejected() -> None:
    with pytest.raises(ValueError, match="unbound global names: factor"):
        close_artifact_source("def kernel(x):\n    return x * factor\n")


def test_native_runtime_shim_remains_closed() -> None:
    source = (
        "from __future__ import annotations\nimport types\n"
        + dependency_free_runtime_source(TritonBackend().dependency_free_launcher_info)
        + "\ndef kernel(x):\n    return _default_launcher\n"
    )
    closed = close_artifact_source(source)
    assert "importlib.import_module('cuda.bindings.driver')" in closed
    assert "def default_launcher" in closed
    assert "def get_num_sm" in closed
    assert "def set_triton_allocator" in closed
    compile(closed, "<artifact>", "exec")
    assert close_artifact_source(closed) == closed


def test_modified_runtime_shim_does_not_get_runtime_exemption() -> None:
    source = "import types\n" + dependency_free_runtime_source(
        TritonBackend().dependency_free_launcher_info
    )
    tree = ast.parse(source)
    runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    runtime.body.insert(1, ast.parse("import original_kernel").body[0])
    with pytest.raises(ValueError, match="external module|dynamic Python namespace"):
        close_artifact_source(ast.unparse(tree))
